#!/usr/bin/env python3
"""Phase 4c — bisect t2v quality by frame count.

Generates at fixed 768×768 spatial size but varying frame counts. For
each generation, extracts the middle frame and runs x2t_image self-describe
to get a one-word quality verdict. Finds the largest T_frames that still
produces coherent content.

Math reminder (Wan2.2 VAE temporal compression):
  t_lat = (T_frames - 1) // 4 + 1
  n_lat = t_lat × h_lat × w_lat   (at 768×768: t_lat × 2304)

  T_frames=1  → t_lat=1, n_lat=2304   (same as t2i!)
  T_frames=5  → t_lat=2, n_lat=4608
  T_frames=9  → t_lat=3, n_lat=6912
  T_frames=13 → t_lat=4, n_lat=9216
  T_frames=17 → t_lat=5, n_lat=11520
  T_frames=25 → t_lat=7, n_lat=16128
  T_frames=49 → t_lat=13, n_lat=29952 (Lance default)

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/11_t2v_frame_bisect.py \\
        --frames 1,5,9,13 \\
        --lance-weights /Volumes/.../Lance-3B-Video-bf16 \\
        --vae-weights   /Volumes/.../Wan22-VAE-bf16/vae.safetensors \\
        --vit-weights   /Volumes/.../Lance-3B-Video-bf16/vit.safetensors
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A red panda wearing a gold-trimmed cap rides a bright seaside wave with foam spray.")
    ap.add_argument("--frames", default="1,5,9,13",
                    help="Comma-separated frame counts to test.")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lance-weights", type=Path, required=True)
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--vit-weights", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/lance_t2v_bisect"))
    args = ap.parse_args()

    frames_list = [int(x) for x in args.frames.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"┏━━ Phase 4c — t2v frame-count bisection ━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃  prompt: {args.prompt!r}")
    print(f"┃  spatial: {args.width}×{args.height}, steps={args.steps}, "
          f"cfg={args.cfg_scale}, seed={args.seed}")
    print(f"┃  frames to test: {frames_list}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # --- Stage 1: generate all sizes ---------------------------------------
    print(f"\n=== Loading t2v pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    t2v = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    timings: dict[int, float] = {}
    mid_frame_paths: dict[int, Path] = {}

    for n_frames in frames_list:
        t_lat = (n_frames - 1) // 4 + 1
        h_lat = args.height // 16
        w_lat = args.width // 16
        n_lat = t_lat * h_lat * w_lat
        print(f"\n=== Generating: {n_frames}f × {args.width}×{args.height}  "
              f"(t_lat={t_lat}, n_lat={n_lat}) ===")
        t0 = time.perf_counter()
        frames = t2v.generate(
            args.prompt,
            num_frames=n_frames, height=args.height, width=args.width,
            num_steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed,
        )
        elapsed = time.perf_counter() - t0
        timings[n_frames] = elapsed
        print(f"  generated {frames.shape[0]} decoded frames in {elapsed:.1f}s")

        # Save the middle frame as PNG for self-describe.
        mid_idx = frames.shape[0] // 2
        mid_path = args.out_dir / f"bisect_{n_frames:03d}f_mid.png"
        Image.fromarray(frames[mid_idx]).save(mid_path)
        mid_frame_paths[n_frames] = mid_path

        # Also save MP4 for inspection (if more than 1 frame).
        if frames.shape[0] > 1:
            import imageio
            mp4_path = args.out_dir / f"bisect_{n_frames:03d}f.mp4"
            with imageio.get_writer(mp4_path, fps=12, codec="libx264") as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"  saved {mp4_path}")

    # Free t2v before loading understanding pipeline.
    del t2v
    gc.collect()

    # --- Stage 2: self-describe each middle frame --------------------------
    print(f"\n=== Loading understanding pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    x2t = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vit_safetensors=args.vit_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    descriptions: dict[int, str] = {}
    for n_frames in frames_list:
        img = Image.open(mid_frame_paths[n_frames]).convert("RGB")
        desc = x2t.generate(
            img, "What is shown in this image?",
            max_new_tokens=64, prompt_style="lance",
        )
        descriptions[n_frames] = desc
        print(f"  {n_frames:3d}f mid → {desc!r}")

    # --- Stage 3: tabulate -------------------------------------------------
    print(f"\n┏━━ Frame-count bisection results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ {'frames':>6s} {'t_lat':>5s} {'n_lat':>6s} {'time_s':>8s}  description")
    for n_frames in frames_list:
        t_lat = (n_frames - 1) // 4 + 1
        n_lat = t_lat * (args.height // 16) * (args.width // 16)
        print(f"┃ {n_frames:6d} {t_lat:5d} {n_lat:6d} {timings[n_frames]:8.1f}  "
              f"{descriptions[n_frames]!r}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Save JSON summary too.
    import json
    summary = {
        "prompt": args.prompt,
        "spatial": [args.height, args.width],
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "results": [
            {
                "frames": n,
                "t_lat": (n - 1) // 4 + 1,
                "n_lat": ((n - 1) // 4 + 1) * (args.height // 16) * (args.width // 16),
                "time_s": round(timings[n], 1),
                "description": descriptions[n],
                "mid_frame": str(mid_frame_paths[n]),
            }
            for n in frames_list
        ],
    }
    (args.out_dir / "bisect_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Summary saved to {args.out_dir / 'bisect_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
