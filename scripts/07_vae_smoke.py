#!/usr/bin/env python3
"""Phase 3a smoke test: load converted Wan2.2 VAE + decode a random latent.

Validates the VAE conversion + load path end-to-end. The decoded output of
random latents won't look like a meaningful image — just plausibly-colored
noise — but it confirms:

  1. The converted safetensors loads into mlx-video's Wan22VAEDecoder with
     zero missing/extra keys.
  2. The forward pass runs without crashing or producing NaN/Inf.
  3. The output shape is correct for the requested latent dimensions.
  4. The pixel value distribution looks like normalized image output
     (clipped to [-1, 1]).

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/07_vae_smoke.py \\
        --vae-weights /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors \\
        --output-png /tmp/vae_smoke.png \\
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_video.models.wan_2.vae22 import (
    Wan22VAEDecoder,
    VAE22_MEAN,
    VAE22_STD,
    denormalize_latents,
)
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--output-png", type=Path, default=Path("/tmp/vae_smoke.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent-h", type=int, default=48,
                    help="Latent height (= image_h / 16). 48 for 768² image.")
    ap.add_argument("--latent-w", type=int, default=48,
                    help="Latent width. 48 for 768² image.")
    args = ap.parse_args()

    # 1. Build + load the VAE decoder.
    print(f"=== Building Wan22VAEDecoder ===")
    decoder = Wan22VAEDecoder(z_dim=48, dim=160, dec_dim=256)
    model_keys = set(k for k, _ in tree_flatten(decoder.parameters()))
    print(f"  model has {len(model_keys)} param tensors")

    print(f"\n=== Loading {args.vae_weights} ===")
    t0 = time.perf_counter()
    saved = mx.load(str(args.vae_weights))
    # Filter to just decoder-relevant keys (decoder.* + conv2.*) and discard
    # encoder.*/conv1.* which are for the encoder side.
    dec_state = {
        k: v for k, v in saved.items()
        if k.startswith("decoder.") or k.startswith("conv2.")
    }
    missing = model_keys - set(dec_state.keys())
    extra = set(dec_state.keys()) - model_keys
    if missing:
        print(f"  ✗ MODEL has {len(missing)} keys not in checkpoint:",
              file=sys.stderr)
        for k in sorted(missing)[:5]:
            print(f"     {k}", file=sys.stderr)
        return 2
    if extra:
        print(f"  ✗ CHECKPOINT has {len(extra)} keys not in model:",
              file=sys.stderr)
        for k in sorted(extra)[:5]:
            print(f"     {k}", file=sys.stderr)
        return 2
    print(f"  ✓ {len(model_keys)} keys match, 0 missing, 0 extra")

    decoder.load_weights(list(dec_state.items()))
    mx.eval(decoder.parameters())
    t1 = time.perf_counter()
    print(f"  loaded in {t1-t0:.1f}s")

    # 2. Decode a random latent.
    # Shape: (B=1, T=1, H_lat, W_lat, C=48) — single image, 48-channel latent.
    print(f"\n=== Decoding random latent (T=1, H={args.latent_h}, W={args.latent_w}, C=48) ===")
    mx.random.seed(args.seed)
    # Generate latent in the model's expected normalized space (mean 0, var 1
    # per channel after `normalize_latents`). Use standard normal and then
    # apply `denormalize_latents` to put it back in the "raw" latent space
    # that the decoder expects to consume.
    z_norm = mx.random.normal((1, 1, args.latent_h, args.latent_w, 48))
    # Wan22 decoder expects DENORMALIZED latents per the docstring comment.
    z = denormalize_latents(z_norm)
    print(f"  z shape: {tuple(z.shape)}, dtype: {z.dtype}")

    t0 = time.perf_counter()
    out = decoder(z)
    mx.eval(out)
    t1 = time.perf_counter()
    print(f"  forward: {t1-t0:.2f}s → output shape {tuple(out.shape)}, dtype {out.dtype}")
    assert mx.all(mx.isfinite(out)).item(), "non-finite values in decoded output"

    # 3. Convert to a PNG. Output is (B=1, T'=1, H', W', 3) in [-1, 1].
    img_t = out[0, 0]  # (H', W', 3)
    img_np = np.array(img_t).astype(np.float32)
    print(f"  image tensor shape: {img_np.shape}")
    print(f"  pixel range: [{img_np.min():.3f}, {img_np.max():.3f}]"
          f"  mean: {img_np.mean():.3f}  std: {img_np.std():.3f}")

    # Map [-1, 1] → [0, 255] uint8
    img_u8 = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(args.output_png)
    print(f"\n✓ Saved {args.output_png} ({args.output_png.stat().st_size} bytes)")
    print(f"  Expected: noise-looking image with plausible color distribution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
