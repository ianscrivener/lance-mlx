#!/usr/bin/env python3
"""Phase 5a — quantize converted bf16 weights to q8 and q4.

Lance has a unique opportunity: the two expert towers can be quantized
INDEPENDENTLY. Image quality is GEN-tower-sensitive; understanding quality
is UND-tower-sensitive. Mixed-precision (e.g., UND=q4, GEN=q8) is worth
exploring as a quality knob.

Usage:
    # Uniform 4-bit
    uv run python scripts/05_quantize.py \\
        --mlx-path ~/models/mlx/Lance-3B-bf16 \\
        --output ~/models/mlx/Lance-3B-4bit \\
        --bits 4

    # Mixed precision (advanced)
    uv run python scripts/05_quantize.py \\
        --mlx-path ~/models/mlx/Lance-3B-bf16 \\
        --output ~/models/mlx/Lance-3B-mixed \\
        --und-bits 4 --gen-bits 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def quantize_uniform(src: Path, dst: Path, bits: int, group_size: int = 32) -> None:
    """Standard mlx-lm-style group quantization at uniform bit width.

    TODO(claude-code): wire to mlx.nn.quantize or mlx_lm.utils.quantize_model
        from mlx_lm.utils import quantize_model
        quantize_model(model, group_size=group_size, bits=bits)
    """
    print(f"  src: {src}")
    print(f"  dst: {dst}")
    print(f"  bits: {bits}, group_size: {group_size}")


def quantize_mixed(src: Path, dst: Path, und_bits: int, gen_bits: int, group_size: int = 32) -> None:
    """Per-expert mixed-precision quantization.

    TODO(claude-code): walk the module tree, applying quantize() with the
    appropriate bit width per submodule. Routing is straightforward because
    LanceMoTLayer has explicit _und / _gen attributes.

        for layer in model.layers:
            for name in ["ffn_und", "qk_norm_und", "o_proj_und"]:
                if hasattr(layer, name):
                    quantize(getattr(layer, name), bits=und_bits, group_size=group_size)
            for name in ["ffn_gen", "qk_norm_gen", "o_proj_gen"]:
                if hasattr(layer, name):
                    quantize(getattr(layer, name), bits=gen_bits, group_size=group_size)
            # Shared attention can use either or its own bit width
    """
    print(f"  src: {src}")
    print(f"  dst: {dst}")
    print(f"  und_bits: {und_bits}, gen_bits: {gen_bits}, group_size: {group_size}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-path", type=Path, required=True,
                        help="Source bf16 MLX checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=[4, 6, 8], default=None,
                        help="Uniform bit width (mutually exclusive with --und-bits/--gen-bits)")
    parser.add_argument("--und-bits", type=int, choices=[4, 6, 8], default=None,
                        help="Bit width for LLM_UND tower (mixed precision)")
    parser.add_argument("--gen-bits", type=int, choices=[4, 6, 8], default=None,
                        help="Bit width for LLM_GEN tower (mixed precision)")
    parser.add_argument("--group-size", type=int, default=32)
    args = parser.parse_args()

    if args.bits is not None and (args.und_bits or args.gen_bits):
        print("ERROR: --bits is mutually exclusive with --und-bits / --gen-bits", file=sys.stderr)
        return 1
    if (args.und_bits is None) != (args.gen_bits is None):
        print("ERROR: --und-bits and --gen-bits must be specified together", file=sys.stderr)
        return 1

    if not args.mlx_path.exists():
        print(f"ERROR: {args.mlx_path} not found", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    if args.bits is not None:
        print(f"=== Uniform {args.bits}-bit quantization ===")
        quantize_uniform(args.mlx_path, args.output, args.bits, args.group_size)
    else:
        print(f"=== Mixed precision: UND={args.und_bits}-bit, GEN={args.gen_bits}-bit ===")
        quantize_mixed(args.mlx_path, args.output, args.und_bits, args.gen_bits, args.group_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
