#!/usr/bin/env python3
"""Phase 5b — quantize Lance LLM weights to 8-bit (groupwise affine).

Quantizes the LLM bulk (Linear layers + lm_head + embed_tokens) while skipping
numerics-sensitive small modules (time_embedder, llm2vae flow head). The VAE
input projection (vae_in_proj.vae2llm) auto-skips because its input_dim=48 is
not divisible by the group_size.

Note: the bundled Wan2.2 VAE (vae.safetensors) and Qwen2.5-VL ViT (vit.safetensors)
are NOT in LanceModel and remain bf16 — they have their own loaders.

Output layout mirrors the bf16 source:
  <out-dir>/
    model.safetensors          quantized LLM weights
    config.json                bf16 source config + 'quantization' block added
    conversion_report.json     provenance + quantization stats

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/16_quantize.py \\
        --lance-weights /Volumes/.../Lance-3B-bf16 \\
        --out-dir       /Volumes/.../Lance-3B-8bit \\
        --bits 8 --group-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import quantize_model
from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model import LanceModel


# Module path patterns we explicitly DO NOT quantize.
SKIP_PATTERNS_ALWAYS = (
    "time_embedder.proj_in",
    "time_embedder.proj_out",
    "llm2vae",
    # latent_pos_embed.pos_embed is nn.Embedding — auto-handled
    # vae_in_proj.vae2llm has in_dim=48 (not %64) — auto-skipped
)

# The GEN tower (`*_moe_gen` projections + MLPs) is numerically sensitive at
# 8-bit for video. Empirically (Phase 5b 2026-05-21): quantizing both towers
# produces gray-gradient noise at 256² × 17f t2v. UND-only quantization
# preserves quality while still cutting LLM weights ~25%. For image tasks
# (Lance_3B), full quantization (both towers) is fine — t2i quality stays
# photorealistic at 8-bit.
SKIP_PATTERNS_GEN_TOWER = (
    "_moe_gen",                 # _proj_moe_gen, mlp_moe_gen.*
)


def make_quant_predicate(skip_gen_tower: bool):
    """Return a predicate that skips small/sensitive modules and optionally
    the entire GEN expert tower."""
    skip = list(SKIP_PATTERNS_ALWAYS)
    if skip_gen_tower:
        skip += list(SKIP_PATTERNS_GEN_TOWER)
    skip_tuple = tuple(skip)
    def pred(path: str, module: nn.Module) -> bool:
        return not any(p in path for p in skip_tuple)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-weights", type=Path, required=True,
                    help="Path to bf16 Lance directory.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write the quantized model.safetensors + config.")
    ap.add_argument("--bits", type=int, default=8,
                    help="Bits per weight. 8 (safer) or 4 (smaller; gen-tower-sensitive).")
    ap.add_argument("--group-size", type=int, default=64,
                    help="Group size. 64 default; try 32 for 4-bit gen-path stability.")
    ap.add_argument("--skip-gen-tower", action="store_true",
                    help="Skip the *_moe_gen projections + MLPs (the GEN expert). "
                         "Required for Lance_3B_Video at 8-bit (preserves t2v quality).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = args.lance_weights

    print(f"┏━━ Phase 5b — Lance LLM quantization ━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ source      : {src}")
    print(f"┃ output      : {args.out_dir}")
    print(f"┃ bits        : {args.bits}")
    print(f"┃ group_size  : {args.group_size}")
    print(f"┃ mode        : affine (mlx-lm default)")
    print(f"┃ skip GEN    : {args.skip_gen_tower} "
          f"({'preserves t2v quality' if args.skip_gen_tower else 'full quant'})")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # --- 1. Load source ----------------------------------------------------
    print(f"\n=== Loading bf16 source ===")
    t0 = time.perf_counter()
    cfg = json.loads((src / "config.json").read_text())
    text_cfg = TextConfig(
        model_type=cfg["model_type"], hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"], intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"], rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"], num_key_value_heads=cfg.get("num_key_value_heads"),
        max_position_embeddings=cfg.get("max_position_embeddings", 128000),
        rope_theta=cfg.get("rope_theta", 1e6),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
    )
    saved = mx.load(str(src / "model.safetensors"))
    n_lat_positions = saved["latent_pos_embed.pos_embed"].shape[0]
    model = LanceModel(text_cfg, num_latent_positions=n_lat_positions)
    model.load_weights(list(saved.items()))
    mx.eval(model.parameters())
    print(f"  loaded {len(saved)} tensors in {time.perf_counter()-t0:.1f}s")

    # --- 2. Tally bf16 footprint ------------------------------------------
    bf16_bytes = sum(int(v.nbytes) for v in saved.values())
    print(f"  bf16 footprint: {bf16_bytes / 1e9:.2f} GB")

    # --- 3. Quantize -------------------------------------------------------
    print(f"\n=== Quantizing ===")
    t0 = time.perf_counter()
    # Build a minimal config dict that quantize_model recognizes.
    quant_config = dict(cfg)
    quantized_model, quantized_config = quantize_model(
        model=model,
        config=quant_config,
        group_size=args.group_size,
        bits=args.bits,
        mode="affine",
        quant_predicate=make_quant_predicate(args.skip_gen_tower),
    )
    # Stash skip-policy in config so the loader can mirror it on the receiving end.
    quantized_config["quantization"]["skip_gen_tower"] = args.skip_gen_tower
    mx.eval(quantized_model.parameters())
    print(f"  quantized in {time.perf_counter()-t0:.1f}s")

    # --- 4. Tally quantized footprint -------------------------------------
    # The new safetensors will have *_scales and *_biases tensors alongside
    # the quantized weight; sum all parameter bytes.
    quant_state = dict(quantized_model.parameters())

    def total_bytes(tree):
        if isinstance(tree, mx.array):
            return int(tree.nbytes)
        elif isinstance(tree, dict):
            return sum(total_bytes(v) for v in tree.values())
        elif isinstance(tree, list):
            return sum(total_bytes(v) for v in tree)
        else:
            return 0

    quant_bytes = total_bytes(quant_state)
    ratio = quant_bytes / bf16_bytes
    print(f"  quantized footprint: {quant_bytes / 1e9:.2f} GB  "
          f"({ratio:.1%} of bf16)")

    # --- 5. Write safetensors ----------------------------------------------
    print(f"\n=== Writing quantized weights ===")
    t0 = time.perf_counter()

    # Flatten nested parameter dict into safetensors-style keys.
    def flatten(prefix, tree, out):
        if isinstance(tree, mx.array):
            out[prefix] = tree
        elif isinstance(tree, dict):
            for k, v in tree.items():
                flatten(f"{prefix}.{k}" if prefix else k, v, out)
        elif isinstance(tree, list):
            for i, v in enumerate(tree):
                flatten(f"{prefix}.{i}" if prefix else str(i), v, out)

    flat: dict[str, mx.array] = {}
    flatten("", quant_state, flat)

    mx.save_safetensors(str(args.out_dir / "model.safetensors"), flat)
    print(f"  wrote {len(flat)} tensors in {time.perf_counter()-t0:.1f}s")

    # --- 6. Write config + report ------------------------------------------
    out_cfg_path = args.out_dir / "config.json"
    out_cfg_path.write_text(json.dumps(quantized_config, indent=2))
    print(f"  wrote {out_cfg_path.name} (with 'quantization' block)")

    report = {
        "source_dir": str(src),
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": "affine",
        "bf16_bytes": bf16_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": ratio,
        "n_tensors_bf16": len(saved),
        "n_tensors_quant": len(flat),
        "skip_gen_tower": args.skip_gen_tower,
        "skip_patterns": list(SKIP_PATTERNS_ALWAYS) + (
            list(SKIP_PATTERNS_GEN_TOWER) if args.skip_gen_tower else []
        ),
    }
    (args.out_dir / "quantization_report.json").write_text(json.dumps(report, indent=2))
    print(f"  wrote quantization_report.json")

    # --- 7. Copy auxiliary files ------------------------------------------
    print(f"\n=== Copying tokenizer + auxiliary files ===")
    import shutil
    for fname in ["tokenizer.json", "vocab.json", "tokenizer_config.json",
                  "generation_config.json", "llm_config.json",
                  "vit.safetensors", "vae.safetensors"]:
        src_path = src / fname
        if src_path.exists():
            shutil.copy(src_path, args.out_dir / fname)
            print(f"  copied {fname}")

    print(f"\n✓ Quantization complete. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
