#!/usr/bin/env python3
"""Convert Lance's bundled Wan2.2_VAE.pth (PyTorch pickle) → MLX safetensors.

Lance ships a 48-channel Wan2.2 VAE alongside its LLM checkpoints. The
PyTorch checkpoint TREE mostly mirrors mlx-video's `Encoder3d`/`Decoder3d`
attribute trees, with three known divergences (handled by this converter):

1. **Tensor format**:
   - Conv3d weight: (out, in, kT, kH, kW) → (out, kT, kH, kW, in)
   - Conv2d weight: (out, in, kH, kW)     → (out, kH, kW, in)
   - RMS_norm gamma: (dim, 1, 1, 1)        → (dim,)  (or (dim, 1, 1) → (dim,))

2. **Sequential indices → layer_N**:
   PyTorch's `nn.Sequential` stores submodules by integer index in the
   state-dict (e.g., `head.0.gamma`, `residual.6.weight`). mlx-video uses
   explicit attribute names (`layer_0`, `layer_2`, …). Renames applied:
     - `head.<N>.X`               → `head.layer_<N>.X`
     - `*.residual.<N>.X`         → `*.residual.layer_<N>.X`

3. **AttentionBlock flattened attrs**:
   PyTorch has `proj.weight/bias` and `to_qkv.weight/bias` as nested
   `nn.Conv2d` submodules. mlx-video stores them as flat parameters:
     - `*.proj.weight`   → `*.proj_weight`   (+ 2D conv transpose)
     - `*.proj.bias`     → `*.proj_bias`
     - `*.to_qkv.weight` → `*.to_qkv_weight` (+ 2D conv transpose)
     - `*.to_qkv.bias`   → `*.to_qkv_bias`

Output goes into `--output` (default `~/.../Wan22-VAE-bf16/`) so the t2i
pipeline can find it alongside the LLM safetensors.

Usage:
    HF_HUB_DISABLE_XET=1 \\
    uv run python scripts/06_convert_wan_vae.py \\
        --src ~/.cache/huggingface/hub/.../Wan2.2_VAE.pth \\
        --output /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx


# Patterns are matched with `re.search` (not anchored) so keys like
# `decoder.head.0.gamma` are handled regardless of their `encoder.`/`decoder.`
# prefix. Patterns use `\b` or specific structural context to avoid
# false positives.
_RENAMES = [
    # Sequential-index → layer_N: `head.N.X` (head's nn.Sequential layers).
    (re.compile(r"(\bhead)\.(\d+)\.([^.]+)$"),                 r"\1.layer_\2.\3"),
    # `residual.N.X` (ResidualBlockLayers' nn.Sequential layers).
    (re.compile(r"(\bresidual)\.(\d+)\.([^.]+)$"),             r"\1.layer_\2.\3"),
    # Resample's nn.Sequential — `.resample.1.X` → `.resample_X`. The N=0 entry
    # is Upsample/ZeroPad2d (no params); only N=1 (the Conv2d) appears in the
    # state dict.
    (re.compile(r"\.resample\.1\.(weight|bias)$"),             r".resample_\1"),
    # AttentionBlock — flatten `to_qkv.X`, `proj.X` to flat attribute names.
    (re.compile(r"\.to_qkv\.(weight|bias)$"),                  r".to_qkv_\1"),
    (re.compile(r"\.proj\.(weight|bias)$"),                    r".proj_\1"),
]


def rename_key(k: str) -> str:
    """Apply key renames in sequence. Returns the new key (possibly unchanged).

    Each key gets at most ONE rename applied (the first matching pattern). The
    patterns are orthogonal in practice — no Wan2.2 key matches two of them.
    """
    for pat, repl in _RENAMES:
        if pat.search(k):
            return pat.sub(repl, k)
    return k


def needs_2d_conv_transpose(k: str) -> bool:
    """True for 2D conv weights — both in 'resample' layers and in
    AttentionBlock's flattened to_qkv/proj."""
    return (
        k.endswith("resample_weight")
        or k.endswith("to_qkv_weight")
        or k.endswith("proj_weight")
    )


def convert(src_pth: Path, output_dir: Path, dtype: str = "bf16") -> int:
    """Load PyTorch pickle, transform, save as MLX safetensors."""
    print(f"Loading {src_pth} ({src_pth.stat().st_size/1e9:.2f} GB) ...")
    import torch
    sd = torch.load(str(src_pth), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    dtype_mlx = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[dtype]
    out: dict[str, mx.array] = {}
    counts = {"conv3d": 0, "conv2d": 0, "rms_gamma": 0, "attn_norm_gamma": 0,
              "renamed": 0, "other": 0}

    for k, v in sd.items():
        arr_np = v.detach().to(torch.float32).numpy()
        ndim = arr_np.ndim

        new_k = rename_key(k)
        if new_k != k:
            counts["renamed"] += 1

        if ndim == 5:
            # Conv3d weight: (O, I, kT, kH, kW) → (O, kT, kH, kW, I)
            arr_np = arr_np.transpose(0, 2, 3, 4, 1)
            counts["conv3d"] += 1
        elif ndim == 4 and needs_2d_conv_transpose(new_k):
            # Conv2d weight: (O, I, kH, kW) → (O, kH, kW, I)
            arr_np = arr_np.transpose(0, 2, 3, 1)
            counts["conv2d"] += 1
        elif ndim == 4 and new_k.endswith(".gamma"):
            # RMS_norm gamma in main path: (dim, 1, 1, 1) → (dim,)
            arr_np = arr_np.reshape(-1)
            counts["rms_gamma"] += 1
        elif ndim == 3 and new_k.endswith(".gamma"):
            # AttentionBlock's norm.gamma: (dim, 1, 1) → (dim,)
            arr_np = arr_np.reshape(-1)
            counts["attn_norm_gamma"] += 1
        else:
            counts["other"] += 1

        out[new_k] = mx.array(arr_np).astype(dtype_mlx)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "vae.safetensors"
    print(f"Writing {out_path} ({len(out)} tensors) ...")
    mx.save_safetensors(str(out_path), out)

    total_params = sum(t.size for t in out.values())
    print(f"  {total_params/1e9:.3f} B params, dtype={dtype}")
    print(f"  shape conversions: {counts}")

    # Provenance metadata.
    report = {
        "source_pth": str(src_pth),
        "n_tensors": len(out),
        "total_params": total_params,
        "dtype": dtype,
        "shape_conversions": counts,
        "expected_loader": "mlx_vlm.models.wan_2.vae22.Wan22VAEDecoder + Wan22VAEEncoder "
                          "(or Encoder3d + Decoder3d directly — keys mirror PyTorch).",
    }
    (output_dir / "vae_conversion_report.json").write_text(json.dumps(report, indent=2))
    print(f"  wrote vae_conversion_report.json")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Path to Wan2.2_VAE.pth")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output directory")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"ERROR: {args.src} not found", file=sys.stderr)
        return 1
    return convert(args.src, args.output, args.dtype)


if __name__ == "__main__":
    sys.exit(main())
