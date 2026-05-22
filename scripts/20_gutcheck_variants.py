#!/usr/bin/env python3
"""Phase 5d — 4-variant gut-check at small scale.

Runs the red panda surfing prompt through Lance t2v at 256²×17f (fast, ~30s
each) with all 4 combinations of (MaPE shift, cfg_interval). Saves each
variant's mid frame for direct comparison. Total wall-clock ~2-3 min for
all 4 variants.

This is the cheap triangulation BEFORE committing to a 2.25h oracle-scale
confirmation run.
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
    OUT_ROOT = Path("/tmp/lance_gutcheck")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Use the same red panda prompt as the oracle (slightly shorter for speed).
    prompts = json.loads(ORACLE_PROMPT_FILE.read_text())
    prompt = prompts["000000.mp4"]

    variants = [
        ("A_shift2000_cfgall", 2000, None),
        ("B_noshift_cfgall", None, None),
        ("C_shift2000_cfgint", 2000, (0.4, 1.0)),
        ("D_noshift_cfgint", None, (0.4, 1.0)),
    ]

    print(f"┏━━ Phase 5d 4-variant gut-check at 256²×17f ━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompt: {prompt[:80]}...")
    print(f"┃ scale : 17f × 256×256, 30 steps, CFG=4.0, seed=42")
    print(f"┃ variants:")
    for name, mape, cfgi in variants:
        print(f"┃   {name}: mape={mape}, cfg_interval={cfgi}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    import imageio
    import numpy as np
    from PIL import Image

    summary = []
    for name, mape, cfgi in variants:
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Variant {name}: mape={mape}, cfg_interval={cfgi} ===")
        t0 = time.perf_counter()
        frames = pipe.generate(
            prompt,
            num_frames=17, height=256, width=256,
            num_steps=30, cfg_scale=4.0,
            cfg_interval=cfgi,
            seed=42,
            mape_anchor=mape,
        )
        elapsed = time.perf_counter() - t0
        # Save MP4 + mid frame.
        mp4 = out_dir / "video.mp4"
        mid = out_dir / "mid.png"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as writer:
            for f in frames:
                writer.append_data(f)
        Image.fromarray(frames[frames.shape[0]//2]).save(mid)
        diffs = [float(np.abs(frames[i].astype(np.float32) - frames[i-1].astype(np.float32)).mean())
                 for i in range(1, len(frames))]
        mad = float(np.mean(diffs)) if diffs else 0.0
        print(f"  generated in {elapsed:.1f}s  inter-MAD={mad:.2f}")
        summary.append({"name": name, "mape": mape, "cfg_interval": cfgi,
                        "wall_clock_s": round(elapsed, 1), "inter_frame_mad": round(mad, 2),
                        "mid_png": str(mid)})

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n✓ Done. Mid frames at:")
    for s in summary:
        print(f"  {s['name']:25s} {s['mid_png']}")
    print(f"\nSummary: {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
