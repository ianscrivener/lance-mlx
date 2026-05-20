#!/usr/bin/env python3
"""Phase 1d — load a converted Lance MLX checkpoint into LanceModel + dummy forward.

This is the "did everything connect right" gate. It does NOT validate
numerical correctness against the Phase 0 oracle (that's Phase 2+); it
confirms:

  1. The converted safetensors loads into the LanceModel param tree with
     zero unmapped / missing keys.
  2. A small dummy forward (random input_ids, mixed-modality position_group)
     produces the expected output shape and finite values.
  3. The lm_head and llm2vae output heads are reachable + produce the right
     shapes on subset positions.

Usage:
    uv run python scripts/03_load_smoke.py \\
        --weights /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16

Add `--variant lance_3b_video` to test the video checkpoint instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model import LanceModel
from lance_mlx.model.routing import PositionGroup, expert_mask_from_position_group


def text_config_from_json(path: Path) -> TextConfig:
    """Construct a TextConfig from the converter's config.json."""
    cfg = json.loads(path.read_text())
    return TextConfig(
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True,
                    help="Converter output dir (contains model.safetensors + config.json)")
    # (num_latent_positions is auto-detected from the checkpoint header below)
    args = ap.parse_args()

    weights_dir = args.weights
    safetensors_path = weights_dir / "model.safetensors"
    config_path = weights_dir / "config.json"

    for p in (safetensors_path, config_path):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1

    # --- 1. Build the model from config -----------------------------------
    print(f"=== Loading config from {config_path} ===")
    text_cfg = text_config_from_json(config_path)
    print(f"  hidden={text_cfg.hidden_size}, layers={text_cfg.num_hidden_layers}, "
          f"heads={text_cfg.num_attention_heads}, kv={text_cfg.num_key_value_heads}, "
          f"vocab={text_cfg.vocab_size}")

    # Auto-detect latent_pos_embed size from the safetensors header.
    print(f"\n=== Building LanceModel ===")
    t0 = time.perf_counter()
    saved = mx.load(str(safetensors_path))
    lpe_shape = saved["latent_pos_embed.pos_embed"].shape
    num_latent_positions = lpe_shape[0]  # 4096 for Lance_3B, 126976 for Video
    print(f"  num_latent_positions (auto-detected): {num_latent_positions}")

    model = LanceModel(text_cfg, num_latent_positions=num_latent_positions)
    t1 = time.perf_counter()
    print(f"  instantiated in {t1-t0:.1f}s")

    n_params_in_model = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  param tree: {n_params_in_model/1e9:.3f} B params")

    # --- 2. Load checkpoint weights ---------------------------------------
    print(f"\n=== Loading weights from {safetensors_path} ===")
    t0 = time.perf_counter()
    model_keys = set(k for k, _ in tree_flatten(model.parameters()))
    ckpt_keys = set(saved.keys())

    missing = model_keys - ckpt_keys
    extra = ckpt_keys - model_keys
    if missing:
        print(f"  ✗ MODEL has {len(missing)} keys not in checkpoint (first 5):", file=sys.stderr)
        for k in sorted(missing)[:5]:
            print(f"     {k}", file=sys.stderr)
        return 2
    if extra:
        print(f"  ✗ CHECKPOINT has {len(extra)} keys not in model (first 5):", file=sys.stderr)
        for k in sorted(extra)[:5]:
            print(f"     {k}", file=sys.stderr)
        return 2
    print(f"  ✓ {len(model_keys)} keys, all match (0 missing, 0 extra)")

    # mlx-vlm convention: load_weights takes a list of (name, array) pairs.
    model.load_weights(list(saved.items()))
    mx.eval(model.parameters())  # force materialization
    t1 = time.perf_counter()
    print(f"  loaded + eval'd in {t1-t0:.1f}s")

    # --- 3. Dummy forward (mixed UND + GEN) ------------------------------
    print(f"\n=== Dummy forward (B=1, T=32, mixed UND/GEN) ===")
    B, T = 1, 32
    input_ids = mx.random.randint(0, text_cfg.vocab_size, (B, T))
    # Half text (UND), half noisy-VAE (GEN) — exercises both expert paths.
    position_group = mx.array(
        [PositionGroup.TEXT] * (T // 2) + [PositionGroup.NOISY_VAE] * (T - T // 2),
        dtype=mx.int32,
    )
    # Simple 3D position_ids: same values for t, h, w axes (text-like layout).
    position_ids = mx.broadcast_to(
        mx.arange(T, dtype=mx.int32).reshape(1, 1, T),
        (3, B, T),
    )

    t0 = time.perf_counter()
    h = model(
        input_ids=input_ids,
        position_ids=position_ids,
        position_group=position_group,
    )
    mx.eval(h)
    t1 = time.perf_counter()
    print(f"  forward: {t1-t0:.2f}s  →  shape={tuple(h.shape)}  dtype={h.dtype}")
    assert h.shape == (B, T, text_cfg.hidden_size), \
        f"unexpected hidden state shape {h.shape}"
    assert mx.all(mx.isfinite(h)).item(), "non-finite values in output"

    # --- 4. Output heads on subset positions -----------------------------
    # Build index arrays from the known position_group structure (MLX 0.31
    # doesn't support single-arg mx.where(cond); use Python-level enumeration).
    pg_list = position_group.tolist()
    und_idx = mx.array([i for i, v in enumerate(pg_list) if v < PositionGroup.CLEAN_VAE], dtype=mx.int32)
    gen_idx = mx.array([i for i, v in enumerate(pg_list) if v >= PositionGroup.CLEAN_VAE], dtype=mx.int32)
    print(f"\n=== Output heads ===")
    logits = model.lm_head(h[:, und_idx, :])
    velocity = model.llm2vae(h[:, gen_idx, :])
    mx.eval(logits, velocity)
    print(f"  lm_head on  {und_idx.size} UND positions → logits {tuple(logits.shape)}")
    print(f"  llm2vae on  {gen_idx.size} GEN positions → velocity {tuple(velocity.shape)}")
    assert mx.all(mx.isfinite(logits)).item(), "non-finite logits"
    assert mx.all(mx.isfinite(velocity)).item(), "non-finite velocity"

    print(f"\n✓ Load + forward smoke test PASSED for {weights_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
