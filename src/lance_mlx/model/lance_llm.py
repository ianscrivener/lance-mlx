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

Recommended implementation strategy (verified to be feasible):

- **Subclass `mlx_vlm.models.qwen2_5_vl.language.{Attention, Qwen2VLDecoderLayer}`
  with a small delta**, rather than vendoring a snapshot. mlx-vlm's recent
  Qwen2.5-VL plumbing has churned (multiple mRoPE commits within the last 30
  days), so pin a known-good commit in `pyproject.toml`. The `apply_multimodal_
  rotary_pos_emb` function in mlx-vlm is a free function consuming `position_ids`
  — that's the clean seam for MaPE (pre-shift `position_ids` before the layer
  stack; no need to override the rotary embedding itself).

- The class delta per layer:
    `LanceMoTAttention(Attention)`: adds `q_proj_moe_gen`/`k_proj_moe_gen`/
        `v_proj_moe_gen`/`o_proj_moe_gen` and 4 QK-norms; routed `__call__`
        that takes `und_idx` and `gen_idx`.
    `LanceMoTDecoderLayer(Qwen2VLDecoderLayer)`: adds `mlp_moe_gen`,
        `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`; routed
        forward that dispatches by index.

Still requires Phase-0 weight inspection to enumerate exact tensor names
(`scripts/01_inspect_keys.py`) before the converter (`scripts/02_convert.py`)
can map HF safetensors → MLX module tree unambiguously.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .routing import build_index_tensors_from_position_group


class LanceMoTLayer(nn.Module):
    """One dual-expert transformer layer (DUPLICATED attention + DUPLICATED MLP).

    To be implemented as a subclass of `mlx_vlm.models.qwen2_5_vl.language.
    Qwen2VLDecoderLayer` with the verified `_moe_gen` siblings added, per the
    module docstring above. Kept as `NotImplementedError` until Phase-0 key
    inspection confirms the exact safetensors tensor names.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        # UND side: inherited from mlx-vlm's Qwen2VLDecoderLayer in the eventual
        # subclass (`self.self_attn.{q,k,v,o}_proj`, `self.mlp`,
        # `self.input_layernorm`, `self.post_attention_layernorm`, plus the
        # per-expert `q_norm`/`k_norm` we'll add into `self_attn`).
        #
        # GEN side (the delta we add):
        # self.self_attn.q_proj_moe_gen = nn.Linear(hidden, num_heads*head_dim, bias=True)
        # self.self_attn.k_proj_moe_gen = nn.Linear(hidden, num_kv_heads*head_dim, bias=True)
        # self.self_attn.v_proj_moe_gen = nn.Linear(hidden, num_kv_heads*head_dim, bias=True)
        # self.self_attn.o_proj_moe_gen = nn.Linear(num_heads*head_dim, hidden, bias=False)
        # self.self_attn.q_norm_moe_gen = nn.RMSNorm(head_dim, eps=rms_eps)
        # self.self_attn.k_norm_moe_gen = nn.RMSNorm(head_dim, eps=rms_eps)
        # self.mlp_moe_gen = SwiGLUMLP(config)
        # self.input_layernorm_moe_gen = nn.RMSNorm(hidden, eps=rms_eps)
        # self.post_attention_layernorm_moe_gen = nn.RMSNorm(hidden, eps=rms_eps)

    def __call__(
        self,
        h: mx.array,                # (B, T, D) packed sequence
        position_ids: mx.array,     # (B, 3, T) post-MaPE
        position_group: mx.array,   # (T,) modality bucket — derives und/gen indexes
        attention_mask: mx.array | None = None,
        kv_cache: tuple | None = None,
    ) -> mx.array:
        """Strict index-routed forward (no soft mixing).

        Reference pattern (from upstream `Qwen2MoTDecoderLayer.forward_train`):

            und_idx, gen_idx = build_index_tensors_from_position_group(position_group)
            h_norm = mx.zeros_like(h)
            h_norm[..., und_idx, :] = self.input_layernorm(h[..., und_idx, :])
            h_norm[..., gen_idx, :] = self.input_layernorm_moe_gen(h[..., gen_idx, :])
            attn_out = self.self_attn(h_norm, position_ids, und_idx, gen_idx, ...)
            h = h + attn_out
            h_norm2 = mx.zeros_like(h)
            h_norm2[..., und_idx, :] = self.post_attention_layernorm(h[..., und_idx, :])
            h_norm2[..., gen_idx, :] = self.post_attention_layernorm_moe_gen(h[..., gen_idx, :])
            mlp_out = mx.zeros_like(h)
            mlp_out[..., und_idx, :] = self.mlp(h_norm2[..., und_idx, :])
            mlp_out[..., gen_idx, :] = self.mlp_moe_gen(h_norm2[..., gen_idx, :])
            return h + mlp_out
        """
        und_idx, gen_idx = build_index_tensors_from_position_group(position_group)
        raise NotImplementedError(
            "LanceMoTLayer: implement after Phase-0 weight inspection. See module "
            "docstring for the verified architecture and recommended subclass pattern."
        )


class LanceModel(nn.Module):
    """Full Lance LLM backbone — 36 LanceMoTLayer + embeddings + UNTIED lm_head
    + per-expert final RMSNorms.

    Does NOT include the ViT or VAE — those are vendored from mlx-vlm and
    mlx-video respectively (see the pipeline modules for orchestration).

    Critical: do NOT tie `lm_head.weight` to `embed_tokens.weight`. Load both
    as independent tensors from the safetensors — the JSON's
    `tie_word_embeddings: true` is overridden at runtime by `untie_lm_head()`.

    Phase-1a empirical additions to the layout (not in original scaffold):
      - `self.norm_moe_gen` — second final RMSNorm for GEN tokens; sibling of
        `self.norm`. Routed by `position_group` at the end of the layer stack.
      - `self.vae_in_proj` (VAEInputProjection, vae_bridge.py) — applied to
        VAE-latent token features at input time before they join the stream.
      - `self.latent_pos_embed` (LatentPosEmbed, latent_pos_embed.py) — added
        to VAE-latent token hidden states for spatial-grid positional info.
      - `self.time_embedder` (TimestepEmbedder, time_embedder.py) — broadcast
        added to ALL token positions in the stream during flow-matching steps.
      - The flow head `self.llm2vae` (FlowHead, flow_head.py) hangs off this
        model; called on hidden states at noisy-VAE positions after `self.norm_moe_gen`.
    """

    def __init__(self, config: dict):
        super().__init__()
        # self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        # self.layers = [LanceMoTLayer(config) for _ in range(config["num_hidden_layers"])]
        # self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])           # UND
        # self.norm_moe_gen = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])   # GEN (Phase-1a empirical)
        # self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        # # Phase-1a empirical additions:
        # self.vae_in_proj = VAEInputProjection(latent_channels=48, hidden_size=config["hidden_size"])
        # self.latent_pos_embed = LatentPosEmbed(max_latent_size=64, hidden_size=config["hidden_size"])
        # self.time_embedder = TimestepEmbedder(hidden_size=config["hidden_size"])
        # self.llm2vae = FlowHead(hidden_size=config["hidden_size"], latent_channels=48)
        ...

    def __call__(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        position_group: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Returns:
            logits: (B, T, vocab_size) — over UND positions (autoregressive next-token).
            hidden_states: (B, T, hidden_size) — fed to `flow_head.llm2vae` at GEN
                positions for velocity prediction.
        """
        raise NotImplementedError(
            "LanceModel: implement after Phase-0 weight inspection (scripts/01_inspect_keys.py)."
        )
