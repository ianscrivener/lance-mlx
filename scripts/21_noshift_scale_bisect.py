#!/usr/bin/env python3
"""Phase 5d — bisect the scale at which no-MaPE-shift t2v breaks.

At 256²×17f the no-shift t2v produces recognizable content (red panda
with hat on surfboard). At 768²×50f it collapses to gradient. Find the
threshold.

Probes (all with mape_anchor=None, seed=42, 30 steps, CFG=4.0):
  256² × 17f  (n_lat = 1280)   — known working
  480×704 × 17f (n_lat=6600)   — LTX comparison scale
  512² × 17f  (n_lat = 5120)
  768² × 13f  (n_lat = 9216)
  768² × 17f  (n_lat = 11520)
  768² × 25f  (n_lat = 16128)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ORACLE_PROMPT_FILE = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
    "t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630/prompt.json"
)


def main() -> int:
    LANCE = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16")
    VAE = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors")
    OUT = Path("/tmp/lance_noshift_bisect")
    OUT.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(ORACLE_PROMPT_FILE.read_text())
    prompt = prompts["000000.mp4"]

    probes = [
        # (label, h, w, frames)
        ("256_17", 256, 256, 17),
        ("480x704_17", 480, 704, 17),
        ("512_17", 512, 512, 17),
        ("768_13", 768, 768, 13),
        ("768_17", 768, 768, 17),
    ]

    print(f"┏━━ Phase 5d scale bisect — no MaPE shift ━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompt: {prompt[:80]}...")
    print(f"┃ all variants: mape_anchor=None, seed=42, 30 steps, CFG=4.0")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE,
        vae_safetensors=VAE,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    import imageio
    import numpy as np
    from PIL import Image

    summary = []
    for label, h, w, num_frames in probes:
        t_lat = (num_frames - 1) // 4 + 1
        n_lat = t_lat * (h // 16) * (w // 16)
        print(f"\n=== {label}: {num_frames}f × {w}×{h}, n_lat={n_lat} ===")
        out_dir = OUT / label
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            frames = pipe.generate(
                prompt,
                num_frames=num_frames, height=h, width=w,
                num_steps=30, cfg_scale=4.0,
                seed=42, mape_anchor=None,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append({"label": label, "h": h, "w": w, "frames": num_frames,
                            "n_lat": n_lat, "error": str(e)})
            continue
        elapsed = time.perf_counter() - t0

        mp4 = out_dir / "video.mp4"
        mid = out_dir / "mid.png"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as writer:
            for f in frames:
                writer.append_data(f)
        Image.fromarray(frames[frames.shape[0]//2]).save(mid)

        # Inter-frame MAD as a proxy for "is anything happening?"
        diffs = [float(np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32)).mean())
                 for i in range(1, len(frames))]
        mad = float(np.mean(diffs)) if diffs else 0.0
        # Variance across image as a proxy for "is it a gradient or scene?"
        # Gradient => very low spatial std. Scene => higher std (objects, edges).
        spatial_std = float(np.std(frames[frames.shape[0]//2].astype(np.float32)))

        print(f"  generated in {elapsed:.1f}s  inter-MAD={mad:.2f}  spatial_std={spatial_std:.1f}")
        summary.append({
            "label": label, "h": h, "w": w, "frames": num_frames, "n_lat": n_lat,
            "wall_clock_s": round(elapsed, 1),
            "inter_frame_mad": round(mad, 2),
            "spatial_std": round(spatial_std, 1),
            "mid_png": str(mid),
        })

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(f"{'label':12s} {'n_lat':>6s} {'time_s':>8s} {'inter_MAD':>9s} {'sp_std':>8s}")
    for s in summary:
        if "error" in s:
            print(f"  {s['label']:12s} {s['n_lat']:6d}  ERROR")
        else:
            print(f"  {s['label']:12s} {s['n_lat']:6d} {s['wall_clock_s']:8.1f} "
                  f"{s['inter_frame_mad']:9.2f} {s['spatial_std']:8.1f}")
    print(f"\nsummary at {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
