#!/usr/bin/env python3
"""Phase 5g — 4-variant gut-check for P0a (fp32 RoPE) × P0b (sms divisor).

Runs the red panda surfing prompt at 256²×17f (fast, ~30s each) with all 4
combinations of (rope_fp32, spatial_merge_size). Saves each variant's MP4
+ mid frame for visual comparison.

If any P0 candidate is the bug:
  - V0 (baseline)        : current legacy painterly/watercolor output
  - V1 (sms=2 only)      : if sharper → P0b is at least part of the fix
  - V2 (rope_fp32 only)  : if sharper → P0a is at least part of the fix
  - V3 (combined)        : if sharpest → both fixes additive
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
    LANCE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16")
    VAE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors")
    OUT_ROOT = Path("/tmp/lance_phase5g")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(ORACLE_PROMPT_FILE.read_text())
    prompt = prompts["000000.mp4"]

    # (label, rope_fp32, spatial_merge_size)
    variants = [
        ("V0_baseline",       False, 1),
        ("V1_sms2",           False, 2),
        ("V2_ropefp32",       True,  1),
        ("V3_sms2_ropefp32",  True,  2),
    ]

    print(f"┏━━ Phase 5g 4-variant gut-check at 256²×17f ━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompt: {prompt[:80]}...")
    print(f"┃ scale : 17f × 256×256, 30 steps, CFG=4.0, seed=42")
    print(f"┃ MaPE  : None (no-shift, Phase 5d default)")
    print(f"┃ variants:")
    for name, rfp32, sms in variants:
        print(f"┃   {name}: rope_fp32={rfp32}, spatial_merge_size={sms}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Loading pipeline (shared across variants) ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    import imageio
    from PIL import Image
    import numpy as np

    times = {}
    for name, rfp32, sms in variants:
        print(f"\n=== {name} (rope_fp32={rfp32}, sms={sms}) ===")
        t0 = time.perf_counter()
        frames = pipe.generate(
            prompt,
            num_frames=17, height=256, width=256,
            num_steps=30, cfg_scale=4.0,
            seed=42, verbose=False,
            mape_anchor=None,
            spatial_merge_size=sms,
            rope_fp32=rfp32,
        )
        dt = time.perf_counter() - t0
        times[name] = dt
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        # Write MP4.
        mp4 = OUT_ROOT / f"{name}.mp4"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))
        print(f"  → {mp4} ({mp4.stat().st_size/1e3:.0f} KB)")

        # Write mid-frame PNG for quick side-by-side.
        mid = int(frames.shape[0] // 2)
        png = OUT_ROOT / f"{name}_midframe.png"
        Image.fromarray(np.asarray(frames[mid])).save(png)
        print(f"  → {png}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for name in times:
        print(f"┃ {name:24s} {times[name]:6.1f}s")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\nNext: open {OUT_ROOT} in Finder and visually compare V0..V3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
