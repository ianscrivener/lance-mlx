#!/usr/bin/env python3
"""Issue #1 v1-focused diagnostic.

The multi-variant script (39_issue1_nlat_diagnostic.py) died at V1, most
likely OOM-killed at 768²×17f (n_lat=11520). Stdout capture also broke
because we monkey-patched print.

This version: ONE variant at a time, plain verbose, normal stdout. Easy
to spot OOM messages, easy to see per-step stats trajectory.

Run with --frames N to pick t_lat. Defaults to t_lat=5 (17 frames) which
is just past the known cliff.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/40_issue1_v1_focused.py [--frames 17]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16"))
    ap.add_argument("--vae-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors"))
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", type=Path, default=Path("/tmp/lance_issue1_v1"))
    ap.add_argument("--prompt", type=str,
                    default="A medium-close shot shows a red panda wearing a gold-trimmed "
                            "cap and travel satchel on a bright seaside wave with a painted "
                            "surfboard, foam spray, and a glowing summer sky.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t_lat = (args.frames - 1) // 4 + 1
    h_lat = args.height // 16
    w_lat = args.width // 16
    n_lat = t_lat * h_lat * w_lat

    print(f"┏━━ Issue #1 single-variant diagnostic ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ scale  : {args.height}×{args.width}, {args.frames} frames")
    print(f"┃ t_lat  : {t_lat}  (frames = (t_lat-1)*4 + 1)")
    print(f"┃ n_lat  : {n_lat}  (= t_lat × h_lat × w_lat = {t_lat} × {h_lat} × {w_lat})")
    print(f"┃ steps  : {args.num_steps}, CFG={args.cfg}, seed={args.seed}")
    print(f"┃ reference points:")
    print(f"┃   n_lat=9216  (t_lat=4) → known PHOTOREAL (V0 baseline)")
    print(f"┃   n_lat=11520 (t_lat=5) → known DEGRADED (just past cliff)")
    print(f"┃   n_lat=29952 (t_lat=13)→ known PURE NOISE")
    print(f"┃ this run: n_lat={n_lat}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    import mlx.core as mx
    import numpy as np
    import imageio
    from PIL import Image
    from lance_mlx.pipeline.t2v import TextToVideoPipeline

    # Peak memory tracking
    initial_mem = mx.metal.get_active_memory() / (1024**3)
    print(f"  initial active memory: {initial_mem:.2f} GB")

    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    post_load_mem = mx.metal.get_active_memory() / (1024**3)
    peak_mem = mx.metal.get_peak_memory() / (1024**3)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  post-load active: {post_load_mem:.2f} GB  peak so far: {peak_mem:.2f} GB")

    print(f"\n=== Generating (verbose: per-step latent stats) ===")
    t0 = time.perf_counter()
    try:
        video = pipe.generate(
            args.prompt,
            num_frames=args.frames,
            height=args.height, width=args.width,
            num_steps=args.num_steps, cfg_scale=args.cfg,
            seed=args.seed, verbose=True,
            mape_anchor=None,
        )
    except Exception as e:
        print(f"\n⚠ GENERATION FAILED: {type(e).__name__}: {e}")
        peak_after = mx.metal.get_peak_memory() / (1024**3)
        print(f"  peak memory at failure: {peak_after:.2f} GB")
        return 1

    dt = time.perf_counter() - t0
    peak_after = mx.metal.get_peak_memory() / (1024**3)
    print(f"\n=== Done ===")
    print(f"  generated {video.shape[0]} frames in {dt:.1f}s")
    print(f"  peak memory: {peak_after:.2f} GB")

    label = f"frames{args.frames}_tlat{t_lat}_nlat{n_lat}"
    mp4 = args.out / f"{label}.mp4"
    with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
        for fr in video:
            w.append_data(np.asarray(fr))
    mid = int(video.shape[0] // 2)
    png = args.out / f"{label}_midframe.png"
    Image.fromarray(np.asarray(video[mid])).save(png)
    print(f"  → {mp4}")
    print(f"  → {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
