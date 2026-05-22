"""Dual-expert Mixture-of-Transformer-Experts backbone for Lance.

Architecture (VERIFIED against upstream source 2026-05-19 — supersedes the
original scaffold's open questions):

- 36 transformer layers (`num_hidden_layers=36`)
- Hidden 2048, intermediate 11008, 16 attention heads, 2 KV heads (GQA 8:1),
  `head_dim = 128`. Standard Qwen2.5-VL-3B dimensions.
- mRoPE with `rope_theta=1e6`, `mrope_section=[16, 24, 24]`. MaPE re-anchoring
  is applied to position_ids BEFORE the layer stack (see `mape.py`); no
  per-layer MaPE module.

Resolved questions (verified against `modeling/lance/qwen2_navit.py`):

1. **QKV projections are DUPLICATED per expert, NOT shared.** Each MoT layer
   holds two full attention substrates: `{q,k,v,o}_proj` for UND and
   `{q,k,v,o}_proj_moe_gen` for GEN. Upstream's `PackedAttentionMoT.__init__`
   creates the UND set via `super().__init__()` and ADDS the `_moe_gen`
   siblings. The shell flag `--copy_init_moe true` populates the GEN side
   from UND at load time.

2. **Per-expert QK-Norms: 4 RMSNorms per layer, 144 total across 36 layers.**
   `q_norm`, `k_norm`, `q_norm_moe_gen`, `k_norm_moe_gen` — each over
   `head_dim=128`. Tiny in params, but separate state-dict entries each.
   ⚠ NOTE: mlx-vlm's stock `Attention` does NOT have QK-norms — we add all
   four ourselves on top of the inherited q/k/v/o_proj.

   ⚠ Phase-1a empirical correction (2026-05-20): the **final** RMSNorm is
   ALSO per-expert. `model.norm` (UND) and `model.norm_moe_gen` (GEN) are
   BOTH present in the safetensors, each [2048]. Total RMSNorm count is
   therefore 146, not 144. Applied at the end of the layer stack, routed
   by `position_group` per-token (UND tokens → `self.norm`, GEN tokens
   → `self.norm_moe_gen`).

3. **Routing is strict per-token; NO cross-expert blending.** Each token
   passes through exactly one expert's input-layernorm → attention → MLP path
   and the result is written back via index assignment to a zero-init buffer
   which is then added to the residual. `freeze_und` optionally `.detach()`s
   UND outputs for fine-tuning GEN — not relevant for inference.

4. **LM head is UNTIED at runtime** despite `llm_config.json` saying
   `tie_word_embeddings: true`. `inference_lance.sh` passes
   `--tie_word_embeddings false`; the code calls `untie_lm_head()` after
   weight load. The safetensors contains a distinct `lm_head.weight` tensor
   (confirm in Phase-0 weight inspection).

5. **Per-expert prefixes in safetensors** follow the `_moe_gen` suffix
   pattern: UND keys carry no suffix (inherited from Qwen2 layer naming),
   GEN keys are siblings with `_moe_gen` appended (e.g. `q_proj_moe_gen`,
   `mlp_moe_gen`, `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`).

v1 implementation strategy (correctness-first, 2026-05-20):

Both expert paths are computed on ALL tokens at each routing point, then
merged with `mx.where`. This produces the same numerical output as upstream's
gather/scatter pattern but avoids scatter-assignment which complicates MLX's
functional autograd. The cost is 2× FLOPs on the dominant MLP — for inference
on Apple Silicon this is currently dwarfed by attention SDP at typical Lance
sequence lengths (8K–20K tokens for image/video gen). Optimization to
gather/scatter (or sorted-modality slicing) is a Phase 5 task once the
correctness baseline is validated against the Phase 0 oracle.

Subclassing strategy (verified to be feasible):

- We subclass `mlx_vlm.models.qwen2_5_vl.language.{Attention, Qwen2VLDecoderLayer}`
  with small deltas. Upstream's commit is pinned in `pyproject.toml`.
- `apply_multimodal_rotary_pos_emb` in mlx-vlm is a free function consuming
  `position_ids` — the clean seam for MaPE (pre-shift `position_ids` before
  the layer stack; no need to override the rotary embedding itself).

Class layout:

- `LanceMoTAttention(Attention)`: adds `q_proj_moe_gen`/`k_proj_moe_gen`/
  `v_proj_moe_gen`/`o_proj_moe_gen` and 4 QK-norms; routed `__call__`
  that takes `position_group` and merges per-token via `mx.where`.

- `LanceMoTLayer(Qwen2VLDecoderLayer)`: adds `mlp_moe_gen`,
  `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`; routed
  forward that dispatches by `position_group`.

- `LanceModel`: full backbone — 36 LanceMoTLayer + embeddings + UNTIED
  lm_head + per-expert final RMSNorms + flow head + VAE bridge + latent
  pos embed + timestep embedder. NOT IMPLEMENTED THIS SESSION (Phase 1d).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_vlm.models.qwen2_5_vl.config import TextConfig
from mlx_vlm.models.qwen2_5_vl.language import (
    MLP,
    Attention,
    Qwen2VLDecoderLayer,
    apply_multimodal_rotary_pos_emb,
)

from .flow_head import FlowHead
from .latent_pos_embed import LatentPosEmbed
from .routing import expert_mask_from_position_group
from .time_embedder import TimestepEmbedder
from .vae_bridge import VAEInputProjection


def _broadcast_mask(position_group: mx.array, target_dtype) -> mx.array:
    """(T,) int position_group → (1, T, 1) bool mask for per-token routing.

    True == route to GEN expert; False == route to UND expert.
    Reshape lets it broadcast cleanly against (B, T, D)-shaped projections.
    """
    e_mask = expert_mask_from_position_group(position_group)  # (T,) int 0/1
    return (e_mask.reshape(1, -1, 1) == 1)


class LanceMoTAttention(Attention):
    """mlx-vlm Attention + `_moe_gen` projection siblings + 4 per-expert QK-norms.

    On top of stock Attention (q/k/v/o_proj + rotary_emb), adds:
        - q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen, o_proj_moe_gen
        - q_norm, k_norm, q_norm_moe_gen, k_norm_moe_gen (each RMSNorm over head_dim)

    The routed `__call__` takes a `position_group` tensor (per-token modality
    bucket). Tokens where `position_group >= CLEAN_VAE` (i.e., 2 or 3) route to
    GEN-side projections and norms; tokens 0/1 route to UND.

    Attention SDP itself is SHARED — there is one packed sequence and all
    tokens attend to all tokens. Only the *projections* and *norms* are
    duplicated per expert.
    """

    def __init__(self, args: TextConfig):
        super().__init__(args)  # q/k/v/o_proj + rotary_emb
        dim = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv_heads = args.num_key_value_heads or n_heads
        head_dim = dim // n_heads
        eps = args.rms_norm_eps

        # GEN-side projections (mirror UND with same dims/biases)
        self.q_proj_moe_gen = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(n_heads * head_dim, dim, bias=False)

        # 4 per-expert QK-norms (added on top of stock Attention, which has none).
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.q_norm_moe_gen = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm_moe_gen = nn.RMSNorm(head_dim, eps=eps)

        # P0a candidate (issue #2 / Phase 5g): when True, compute cos/sin and
        # the q*cos + rotate_half(q)*sin rotation in fp32 instead of inheriting
        # mlx-vlm's `Qwen2RotaryEmbedding`'s downcast (language.py:73 does
        # `cos.astype(x.dtype)` → bf16 in our run). Hypothesis: bf16 rotation
        # error perturbs high-frequency channels of the flow-matching velocity
        # field, manifesting as soft water/fur/paws. Set via
        # `LanceModel.set_rope_fp32(True)` which iterates all 36 layers.
        # Default: False (preserves legacy behavior).
        self._rope_fp32 = False

    def __call__(
        self,
        x: mx.array,                       # (B, L, D)
        position_group: mx.array,          # (T=L,) modality bucket
        mask: mx.array | None = None,
        cache=None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        B, L, D = x.shape
        # (1, L, 1) bool — True = GEN expert
        gen_mask = _broadcast_mask(position_group, x.dtype)

        # --- Per-expert Q/K/V projection (both paths, merged) -----------------
        queries = mx.where(gen_mask, self.q_proj_moe_gen(x), self.q_proj(x))
        keys    = mx.where(gen_mask, self.k_proj_moe_gen(x), self.k_proj(x))
        values  = mx.where(gen_mask, self.v_proj_moe_gen(x), self.v_proj(x))

        # Reshape to (B, n_heads_or_kv_heads, L, head_dim)
        queries = queries.reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        keys    = keys.reshape   (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values  = values.reshape (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # --- Per-expert QK-norm (over head_dim, applied to (B, H, L, head_dim)) ---
        # Reshape mask for the new layout: (1, 1, L, 1)
        gen_mask_qk = gen_mask.reshape(1, 1, L, 1)
        queries = mx.where(gen_mask_qk, self.q_norm_moe_gen(queries), self.q_norm(queries))
        keys    = mx.where(gen_mask_qk, self.k_norm_moe_gen(keys),    self.k_norm(keys))

        # --- Position-aware rotary (uses post-MaPE position_ids from upstream) ----
        kv_seq_len = keys.shape[-2]
        if position_ids is None:
            offset = cache.offset if cache is not None else 0
            kv_seq_len += offset + (1 if cache is not None else 0)
            position_ids = mx.arange(L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += (cache.offset + 1) if cache is not None else 0

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        if self._rope_fp32:
            # P0a: pass fp32 dummy to rotary_emb so its `cos.astype(x.dtype)`
            # downcast at language.py:73 becomes a no-op fp32→fp32. Then upcast
            # q/k to fp32 for the rotation arithmetic and downcast result back.
            values_fp32 = values.astype(mx.float32)
            cos, sin = self.rotary_emb(values_fp32, position_ids)  # both fp32
            q_fp32 = queries.astype(mx.float32)
            k_fp32 = keys.astype(mx.float32)
            q_rot, k_rot = apply_multimodal_rotary_pos_emb(
                q_fp32, k_fp32, cos, sin, unqueeze_dim=1
            )
            queries = q_rot.astype(queries.dtype)
            keys = k_rot.astype(keys.dtype)
        else:
            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin, unqueeze_dim=1
            )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # --- Shared SDP attention (full sequence, no per-expert split) -----------
        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # --- Per-expert output projection (both paths, merged) -------------------
        return mx.where(gen_mask, self.o_proj_moe_gen(output), self.o_proj(output))


class LanceMoTLayer(Qwen2VLDecoderLayer):
    """mlx-vlm Qwen2VLDecoderLayer + `_moe_gen` siblings.

    Adds to the stock decoder layer:
        - `self.self_attn` replaced by LanceMoTAttention (per-expert q/k/v/o + QK-norms)
        - `self.mlp_moe_gen`, sibling SwiGLU MLP for GEN tokens
        - `self.input_layernorm_moe_gen`, second pre-attention RMSNorm
        - `self.post_attention_layernorm_moe_gen`, second post-attention RMSNorm

    The routed forward pattern (mirrors upstream `Qwen2MoTDecoderLayer.forward_train`):

        r = self_attn(  mx.where(gen, input_layernorm_moe_gen(x),     input_layernorm(x)),
                        position_group, ...)
        h = x + r
        r = mx.where(gen, mlp_moe_gen(post_attention_layernorm_moe_gen(h)),
                          mlp        (post_attention_layernorm(h))             )
        return h + r
    """

    def __init__(self, args: TextConfig):
        super().__init__(args)  # self_attn (Attention), mlp, input_layernorm, post_attention_layernorm

        # Replace stock Attention with our routed subclass (uses same args).
        self.self_attn = LanceMoTAttention(args)

        # GEN-side delta.
        self.mlp_moe_gen = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm_moe_gen        = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,                     # (B, T, D)
        position_group: mx.array,        # (T,) modality bucket
        mask: mx.array | None = None,
        cache=None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        # (1, T, 1) bool — True = GEN expert
        gen_mask = _broadcast_mask(position_group, x.dtype)

        # === Pre-attention: per-expert input_layernorm, then routed attention ===
        h_norm = mx.where(
            gen_mask,
            self.input_layernorm_moe_gen(x),
            self.input_layernorm(x),
        )
        r = self.self_attn(h_norm, position_group, mask, cache, position_ids)
        h = x + r

        # === Post-attention: per-expert post_attention_layernorm + MLP ===========
        h_norm2 = mx.where(
            gen_mask,
            self.post_attention_layernorm_moe_gen(h),
            self.post_attention_layernorm(h),
        )
        mlp_out = mx.where(
            gen_mask,
            self.mlp_moe_gen(h_norm2),
            self.mlp(h_norm2),
        )
        return h + mlp_out


class LanceModel(nn.Module):
    """Full Lance LLM backbone.

    Layout (matches the safetensors keys from `scripts/02_convert.py`):

        embed_tokens                  Embedding(vocab, hidden)
        layers[0..N-1]                LanceMoTLayer × num_hidden_layers
        norm                          RMSNorm(hidden)        — UND final
        norm_moe_gen                  RMSNorm(hidden)        — GEN final
        lm_head                       Linear(hidden, vocab, bias=False) — UNTIED
        vae_in_proj.vae2llm           Linear(48, hidden, bias=True)  — VAE → LLM
        latent_pos_embed.pos_embed    (4096|126976, hidden) parameter
        time_embedder.proj_in/out     Linear pair for sinusoidal-timestep MLP
        llm2vae                       Linear(hidden, 48, bias=True)  — LLM → VAE velocity (flow head)

    Does NOT include the ViT — that lives at the Pipeline orchestrator level
    (see `notes/phase1b_converter_design.md` for the placement rationale).

    `__call__` runs the transformer stack and per-expert final norm. The
    output heads (`self.lm_head`, `self.llm2vae`) are exposed as attributes;
    callers apply them on the position subsets they care about. This avoids
    burning the lm_head matmul (311 M params, expensive) on GEN positions
    where its output is discarded.

    Caller patterns (pipeline modules, not LanceModel itself):

        # x2t_image (VQA): all UND, take logits at last position.
        h = model(input_ids=tokens, position_ids=pids, position_group=groups)
        logits = model.lm_head(h[:, -1:, :])

        # t2i (image-gen flow step): mixed UND + NOISY_VAE.
        # Caller pre-builds inputs_embeds = text_emb || (vae_in_proj(latents) + latent_pos_embed + time_embedder(t)).
        h = model(inputs_embeds=embeds, position_ids=pids, position_group=groups)
        velocity = model.llm2vae(h[:, vae_idx, :])   # (B, n_vae, 48)
    """

    def __init__(self, args: TextConfig, num_latent_positions: int = 4096):
        """
        Args:
            args: mlx-vlm's TextConfig (matches Qwen2.5-VL-3B dimensions for Lance_3B).
            num_latent_positions: size of the `latent_pos_embed.pos_embed` table.
                4096 for Lance_3B (image, 64x64 spatial grid).
                126976 for Lance_3B_Video (4096 × 31 temporal slots).
                On load from a converted checkpoint, this gets overwritten with
                the actual tensor; the value here only sizes the fresh-init buffer.
        """
        super().__init__()
        self.args = args
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [LanceMoTLayer(args) for _ in range(args.num_hidden_layers)]

        # Per-expert final RMSNorms (146 RMSNorms total per Phase 1a inspection).
        self.norm         = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm_moe_gen = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Untied LM head — runtime override of llm_config.json's tie_word_embeddings: true.
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # Phase-1a empirical additions (not in original handoff).
        self.vae_in_proj      = VAEInputProjection(latent_channels=48, hidden_size=args.hidden_size)
        self.latent_pos_embed = LatentPosEmbed(num_positions=num_latent_positions, hidden_size=args.hidden_size)
        self.time_embedder    = TimestepEmbedder(hidden_size=args.hidden_size)
        self.llm2vae          = FlowHead(hidden_size=args.hidden_size, latent_channels=48)

    # ----- Phase 5g (issue #2 P0a) — runtime RoPE precision toggle ----------
    def set_rope_fp32(self, enabled: bool) -> None:
        """Toggle fp32 RoPE rotation across all 36 LanceMoTAttention modules.

        When True, each LanceMoTAttention computes cos/sin and applies the
        rotation `q*cos + rotate_half(q)*sin` in fp32 instead of inheriting
        mlx-vlm's bf16 downcast at `qwen2_5_vl/language.py:73`. This is the
        P0a candidate from the Phase 5e research brief — hypothesized to
        recover high-frequency precision in the flow-matching velocity field
        (manifests as soft water/fur/paws when off).
        """
        for layer in self.layers:
            layer.self_attn._rope_fp32 = bool(enabled)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        *,
        position_ids: mx.array,
        position_group: mx.array,
        mask: mx.array | None = None,
        cache: list | None = None,
    ) -> mx.array:
        """Run embeddings → 36 LanceMoTLayers → per-expert final norm.

        Args:
            input_ids:      (B, T) int token IDs, OR None if inputs_embeds is given.
            inputs_embeds:  (B, T, hidden_size) pre-built embeddings, OR None if input_ids.
                            For mixed-modality (text + ViT + VAE), caller builds these
                            using `model.embed_tokens`, `model.vae_in_proj`,
                            `model.latent_pos_embed`, `model.time_embedder` as helpers.
            position_ids:   (3, B, T) post-MaPE position coordinates for mRoPE.
            position_group: (T,) int per-token modality bucket {0:TEXT, 1:VIT, 2:CLEAN_VAE, 3:NOISY_VAE}.
            mask:           optional attention mask (causal by default if None).
            cache:          optional list of KVCache per layer (decoder steps).

        Returns:
            (B, T, hidden_size) — final hidden states with per-expert RMSNorm applied.
            Caller applies `self.lm_head(h[:, und_idx, :])` for next-token logits
            and `self.llm2vae(h[:, gen_idx, :])` for flow-matching velocity.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("LanceModel.__call__: provide either input_ids or inputs_embeds")
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, position_group, mask, c, position_ids)

        # Per-expert final norm.
        gen_mask = _broadcast_mask(position_group, h.dtype)
        h = mx.where(gen_mask, self.norm_moe_gen(h), self.norm(h))
        return h
