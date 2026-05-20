#!/usr/bin/env python3
"""Phase 1e — convert Lance HF safetensors to MLX format.

Gated on Phase 1a (01_inspect_keys.py) producing notes/lance_architecture.md.
Reads the verdicts from that file to drive key remapping decisions.

Usage:
    uv run python scripts/02_convert.py \\
        --src ~/models/Lance \\
        --variant 3B \\
        --dst ~/models/mlx/Lance-3B-bf16 \\
        --dtype bf16

Variants:
    3B        — Lance_3B/model.safetensors (image)
    3B-Video  — Lance_3B_Video/model.safetensors (video)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_phase1a_passed(notes_dir: Path) -> bool:
    report = notes_dir / "lance_architecture.md"
    if not report.exists():
        print(f"ERROR: {report} not found. Run scripts/01_inspect_keys.py first.", file=sys.stderr)
        return False
    return True


def convert_safetensors(src_file: Path, dst_dir: Path, dtype: str = "bf16") -> None:
    """Convert HF safetensors → MLX safetensors with key remapping.

    TODO(claude-code): full implementation. Steps:
        1. Load HF safetensors with safetensors.numpy.load_file
        2. Apply key remapping per the architecture verdicts in Phase 1a
           (e.g., rename `model.layers.{i}.mlp_und.gate_proj` to whatever
            MLX module path LanceMoTLayer expects)
        3. Apply per-layer transposes if Phase 1a reveals any conv weights
           (unlikely for an LLM-style model, but check VAE if bundled)
        4. Cast to target dtype (bf16/fp16/fp32)
        5. mx.save_safetensors(dst, dict(tree_flatten(weights)))
        6. Copy tokenizer.json, llm_config.json, generation_config.json verbatim
        7. Write a `config.json` for mlx-vlm/mlx-video discovery

    The remapping should be data-driven from `notes/lance_keys_summary.md` —
    don't hardcode 36 layer indices, use the actual keys discovered.
    """
    print(f"  src: {src_file}")
    print(f"  dst: {dst_dir}")
    print(f"  dtype: {dtype}")
    print(f"  TODO(claude-code): implement after Phase 1a verdicts are in")


def convert_wan_vae(src_pth: Path, dst_safetensors: Path) -> None:
    """Convert the bundled Wan2.2_VAE.pth to .safetensors.

    The upstream Lance repo ships the VAE as a pickled .pth file (security-flagged
    on HF). Before any mlx-community upload we need to convert to safetensors.

    TODO(claude-code): straightforward — torch.load with map_location='cpu',
    then safetensors.torch.save_file. Verify shapes match what mlx-video's
    WanVAE expects.
    """
    print(f"  Wan2.2 VAE: {src_pth} → {dst_safetensors}")
    print(f"  TODO(claude-code): implement")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True,
                        help="Path to Lance HF download (containing Lance_3B/ and Lance_3B_Video/)")
    parser.add_argument("--variant", choices=["3B", "3B-Video"], default="3B")
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--force", action="store_true",
                        help="Skip Phase 1a gate check")
    parser.add_argument("--convert-vae", action="store_true",
                        help="Also convert the bundled Wan2.2_VAE.pth to safetensors")
    args = parser.parse_args()

    if not args.force and not check_phase1a_passed(args.notes_dir):
        return 1

    variant_dir = {"3B": "Lance_3B", "3B-Video": "Lance_3B_Video"}[args.variant]
    src_file = args.src / variant_dir / "model.safetensors"
    if not src_file.exists():
        print(f"ERROR: {src_file} not found", file=sys.stderr)
        return 1

    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"=== Converting Lance {args.variant} → MLX {args.dtype} ===")
    convert_safetensors(src_file, args.dst, args.dtype)

    if args.convert_vae:
        vae_pth = args.src / "Wan2.2_VAE.pth"
        if vae_pth.exists():
            convert_wan_vae(vae_pth, args.dst / "Wan2.2_VAE.safetensors")
        else:
            print(f"WARN: {vae_pth} not found; skipping VAE conversion", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
