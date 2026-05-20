"""End-to-end tests for LanceModel — instantiation, fresh-init forward,
and (if a converted checkpoint is available locally) load-from-disk + forward.

The disk-load tests are skipped when the converted checkpoint isn't present —
useful for CI environments without the 12 GB model file.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model import LanceModel
from lance_mlx.model.routing import PositionGroup


# Default checkpoint location matches what scripts/02_convert.py writes by
# convention; override via LANCE_MLX_WEIGHTS env var.
DEFAULT_WEIGHTS_DIR = Path(
    os.environ.get(
        "LANCE_MLX_WEIGHTS",
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16",
    )
)


def _small_config(num_layers: int = 2) -> TextConfig:
    """Tiny TextConfig for fast instantiation tests."""
    return TextConfig(
        model_type="qwen2_5_vl",
        hidden_size=2048,
        num_hidden_layers=num_layers,
        intermediate_size=11008,
        num_attention_heads=16,
        rms_norm_eps=1e-6,
        vocab_size=151936,
        num_key_value_heads=2,
        max_position_embeddings=128000,
        rope_theta=1_000_000.0,
        rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
        tie_word_embeddings=False,
    )


def _mixed_position_group(T: int) -> mx.array:
    half = T // 2
    return mx.array(
        [PositionGroup.TEXT] * half + [PositionGroup.NOISY_VAE] * (T - half),
        dtype=mx.int32,
    )


def _mrope_position_ids(B: int, T: int) -> mx.array:
    # mlx-vlm's mRoPE expects (3, B, T) — same indices across the 3 axes is fine
    # for fresh-init smoke tests; production code uses get_rope_index().
    base = mx.arange(T, dtype=mx.int32).reshape(1, 1, T)
    return mx.broadcast_to(base, (3, B, T))


# --------------------------- Instantiation ----------------------------------

def test_lance_model_instantiates_image_size():
    """4096 latent positions = 64x64 spatial grid (Lance_3B image variant)."""
    args = _small_config(num_layers=2)
    m = LanceModel(args, num_latent_positions=4096)
    assert m.latent_pos_embed.pos_embed.shape == (4096, 2048)
    assert len(m.layers) == 2


def test_lance_model_instantiates_video_size():
    """126976 latent positions for the Lance_3B_Video variant (4096 × 31)."""
    args = _small_config(num_layers=2)
    m = LanceModel(args, num_latent_positions=126976)
    assert m.latent_pos_embed.pos_embed.shape == (126976, 2048)


def test_lance_model_has_all_expected_attributes():
    args = _small_config(num_layers=2)
    m = LanceModel(args)
    for attr in (
        "embed_tokens", "layers", "norm", "norm_moe_gen", "lm_head",
        "vae_in_proj", "latent_pos_embed", "time_embedder", "llm2vae",
    ):
        assert hasattr(m, attr), f"LanceModel missing attribute: {attr}"


def test_lance_model_param_keys_match_converter_output_image():
    """LanceModel param tree keys MUST match scripts/02_convert.py output for
    Lance_3B exactly. This is the load-without-modification contract.

    We don't have the converted file in CI necessarily, but the model itself
    enumerates predictable keys — we can spot-check a representative subset
    against the rules in scripts/02_convert.py."""
    args = _small_config(num_layers=2)
    m = LanceModel(args)
    keys = set(k for k, _ in tree_flatten(m.parameters()))

    # Top-level
    for k in (
        "embed_tokens.weight", "lm_head.weight",
        "norm.weight", "norm_moe_gen.weight",
        "llm2vae.weight", "llm2vae.bias",
        "vae_in_proj.vae2llm.weight", "vae_in_proj.vae2llm.bias",
        "latent_pos_embed.pos_embed",
        "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
    ):
        assert k in keys, f"Expected param key missing: {k}"

    # Per-layer
    for i in (0, 1):
        for k in (
            f"layers.{i}.self_attn.q_proj.weight",
            f"layers.{i}.self_attn.q_proj_moe_gen.weight",
            f"layers.{i}.self_attn.q_norm.weight",
            f"layers.{i}.self_attn.q_norm_moe_gen.weight",
            f"layers.{i}.mlp.gate_proj.weight",
            f"layers.{i}.mlp_moe_gen.gate_proj.weight",
            f"layers.{i}.input_layernorm.weight",
            f"layers.{i}.input_layernorm_moe_gen.weight",
            f"layers.{i}.post_attention_layernorm.weight",
            f"layers.{i}.post_attention_layernorm_moe_gen.weight",
        ):
            assert k in keys, f"Expected per-layer param key missing: {k}"


# --------------------------- Fresh-init forward -----------------------------

def test_lance_model_fresh_forward_shape():
    """Forward on randomly-initialized 2-layer LanceModel produces correct shape."""
    args = _small_config(num_layers=2)
    m = LanceModel(args)
    B, T = 1, 16
    input_ids = mx.random.randint(0, args.vocab_size, (B, T))
    pos_group = _mixed_position_group(T)
    pos_ids = _mrope_position_ids(B, T)

    h = m(
        input_ids=input_ids,
        position_ids=pos_ids,
        position_group=pos_group,
    )
    mx.eval(h)
    assert h.shape == (B, T, args.hidden_size)
    assert mx.all(mx.isfinite(h)).item()


def test_lance_model_inputs_embeds_path():
    """Forward via pre-built inputs_embeds works (the multi-modal path)."""
    args = _small_config(num_layers=2)
    m = LanceModel(args)
    B, T = 1, 8
    inputs_embeds = mx.random.normal((B, T, args.hidden_size))
    pos_group = _mixed_position_group(T)
    pos_ids = _mrope_position_ids(B, T)

    h = m(
        inputs_embeds=inputs_embeds,
        position_ids=pos_ids,
        position_group=pos_group,
    )
    mx.eval(h)
    assert h.shape == (B, T, args.hidden_size)
    assert mx.all(mx.isfinite(h)).item()


def test_lance_model_rejects_both_inputs_and_embeds_none():
    args = _small_config(num_layers=1)
    m = LanceModel(args)
    pos_group = _mixed_position_group(4)
    pos_ids = _mrope_position_ids(1, 4)
    with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
        m(position_ids=pos_ids, position_group=pos_group)


def test_lance_model_output_heads():
    """lm_head + llm2vae produce correct shapes on subset positions."""
    args = _small_config(num_layers=1)
    m = LanceModel(args)
    B, T = 1, 8
    h = mx.random.normal((B, T, args.hidden_size))

    # 4 UND positions, 4 GEN positions
    und_idx = mx.array([0, 1, 2, 3], dtype=mx.int32)
    gen_idx = mx.array([4, 5, 6, 7], dtype=mx.int32)
    logits = m.lm_head(h[:, und_idx, :])
    velocity = m.llm2vae(h[:, gen_idx, :])
    mx.eval(logits, velocity)

    assert logits.shape == (B, 4, args.vocab_size)
    assert velocity.shape == (B, 4, 48)


# --------------------------- Load-from-disk ---------------------------------
#
# These are gated on the converted checkpoint actually being present locally.
# Skip gracefully in environments without the 12 GB safetensors.

WEIGHTS_PATH = DEFAULT_WEIGHTS_DIR / "model.safetensors"
HAS_LANCE_3B = WEIGHTS_PATH.exists()


@pytest.mark.skipif(not HAS_LANCE_3B,
                    reason=f"Lance_3B checkpoint not at {WEIGHTS_PATH}")
def test_load_lance_3b_from_disk_keys_match():
    """Load the converted Lance_3B checkpoint and confirm every model key is
    present in the safetensors and vice-versa."""
    import json
    cfg_path = DEFAULT_WEIGHTS_DIR / "config.json"
    cfg = json.loads(cfg_path.read_text())
    args = TextConfig(
        model_type=cfg["model_type"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"],
        num_key_value_heads=cfg.get("num_key_value_heads"),
        max_position_embeddings=cfg.get("max_position_embeddings", 128000),
        rope_theta=cfg.get("rope_theta", 1e6),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )

    saved = mx.load(str(WEIGHTS_PATH))
    num_latent_positions = saved["latent_pos_embed.pos_embed"].shape[0]
    m = LanceModel(args, num_latent_positions=num_latent_positions)

    model_keys = set(k for k, _ in tree_flatten(m.parameters()))
    ckpt_keys = set(saved.keys())

    missing = model_keys - ckpt_keys
    extra = ckpt_keys - model_keys
    assert not missing, f"Model has keys not in checkpoint: {sorted(missing)[:5]}"
    assert not extra, f"Checkpoint has keys not in model: {sorted(extra)[:5]}"
    assert len(model_keys) == 1021, f"Expected 1021 keys, got {len(model_keys)}"


@pytest.mark.skipif(not HAS_LANCE_3B,
                    reason=f"Lance_3B checkpoint not at {WEIGHTS_PATH}")
def test_load_lance_3b_from_disk_forward_finite():
    """Full load + dummy forward against the real bf16 Lance_3B checkpoint."""
    import json
    cfg_path = DEFAULT_WEIGHTS_DIR / "config.json"
    cfg = json.loads(cfg_path.read_text())
    args = TextConfig(
        model_type=cfg["model_type"], hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        rms_norm_eps=cfg["rms_norm_eps"], vocab_size=cfg["vocab_size"],
        num_key_value_heads=cfg.get("num_key_value_heads"),
        max_position_embeddings=cfg.get("max_position_embeddings", 128000),
        rope_theta=cfg.get("rope_theta", 1e6),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )

    saved = mx.load(str(WEIGHTS_PATH))
    num_latent_positions = saved["latent_pos_embed.pos_embed"].shape[0]
    m = LanceModel(args, num_latent_positions=num_latent_positions)
    m.load_weights(list(saved.items()))
    mx.eval(m.parameters())

    B, T = 1, 16
    input_ids = mx.random.randint(0, args.vocab_size, (B, T))
    pos_group = _mixed_position_group(T)
    pos_ids = _mrope_position_ids(B, T)

    h = m(input_ids=input_ids, position_ids=pos_ids, position_group=pos_group)
    mx.eval(h)
    assert h.shape == (B, T, args.hidden_size)
    assert mx.all(mx.isfinite(h)).item()

    # Heads on subset positions
    und_idx = mx.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=mx.int32)
    gen_idx = mx.array([8, 9, 10, 11, 12, 13, 14, 15], dtype=mx.int32)
    logits = m.lm_head(h[:, und_idx, :])
    velocity = m.llm2vae(h[:, gen_idx, :])
    mx.eval(logits, velocity)
    assert logits.shape == (B, 8, args.vocab_size)
    assert velocity.shape == (B, 8, 48)
    assert mx.all(mx.isfinite(logits)).item()
    assert mx.all(mx.isfinite(velocity)).item()
