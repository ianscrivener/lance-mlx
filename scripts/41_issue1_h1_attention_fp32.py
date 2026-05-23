#!/usr/bin/env python3
"""Issue #1 H1 test — does fp32 attention recover quality at t_lat=5?

The previous diagnostic confirmed:
  - t_lat=4 (768²×13f): photoreal V0 baseline
  - t_lat=5 (768²×17f): regressed (smoother sky, less detail), but latent
    stats are NUMERICALLY BOUNDED (no NaN/inf, mean ~0, std 0.7-1.0).

This is silent semantic drift, not numerical blowup. Hypothesis #1: bf16
softmax / KV-aggregation accumulation in `scaled_dot_product_attention`
at n_lat ≥ 11,520 causes sub-eps additions to compound into observable
degradation.

Test: re-run t_lat=5 with `attention_fp32=True` — promotes Q/K/V to fp32
through RoPE + SDP, downcasts only before o_proj. If the sky/water/fur
detail returns to V0-tier, H1 is confirmed and we have a fix-or-quality
trade-off knob.

Side-by-side compare against:
  - V0 (t_lat=4 baseline, photoreal)
  - V1 (t_lat=5 default, regressed) — the existing midframe
  - V1_fp32attn (t_lat=5 + attention_fp32=True) — this run
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    LANCE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16")
    VAE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors")
    OUT = Path("/tmp/lance_issue1_h1")
    OUT.mkdir(parents=True, exist_ok=True)

    PROMPT = ("A medium-close shot shows a red panda wearing a gold-trimmed "
              "cap and travel satchel on a bright seaside wave with a painted "
              "surfboard, foam spray, and a glowing summer sky.")

    print(f"┏━━ Issue #1 H1: attention_fp32 at t_lat=5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ scale  : 768×768 × 17f  (t_lat=5, n_lat=11,520)")
    print(f"┃ config : 30 steps, CFG=4.0, seed=43, MaPE=None")
    print(f"┃ test   : attention_fp32=True")
    print(f"┃ vs     : /tmp/lance_issue1_diag/V0_tlat4_13f_BASELINE_midframe.png (V0 ref)")
    print(f"┃          /tmp/lance_issue1_v1/frames17_tlat5_nlat11520_midframe.png (V1 bf16)")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    import numpy as np
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    from lance_mlx.pipeline.t2v import TextToVideoPipeline

    print(f"\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s  "
          f"(peak mem so far: {mx.get_peak_memory()/(1024**3):.2f} GB)")

    print(f"\n=== Generating with attention_fp32=True ===")
    t0 = time.perf_counter()
    try:
        video = pipe.generate(
            PROMPT,
            num_frames=17, height=768, width=768,
            num_steps=30, cfg_scale=4.0,
            seed=43, verbose=True,
            mape_anchor=None,
            attention_fp32=True,   # ← the hypothesis under test
        )
    except Exception as e:
        print(f"\n⚠ GENERATION FAILED: {type(e).__name__}: {e}")
        peak = mx.get_peak_memory()/(1024**3)
        print(f"  peak memory at failure: {peak:.2f} GB")
        return 1
    dt = time.perf_counter() - t0
    peak = mx.get_peak_memory()/(1024**3)
    print(f"\n  generated {video.shape[0]} frames in {dt:.1f}s  (peak mem: {peak:.2f} GB)")

    mp4 = OUT / "frames17_tlat5_attention_fp32.mp4"
    with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
        for fr in video:
            w.append_data(np.asarray(fr))
    mid = int(video.shape[0] // 2)
    png = OUT / "frames17_tlat5_attention_fp32_midframe.png"
    Image.fromarray(np.asarray(video[mid])).save(png)
    print(f"  → {mp4}\n  → {png}")

    # Build 3-up compare: V0 (bf16 baseline t_lat=4) | V1 (bf16 t_lat=5) | V1+fp32attn (t_lat=5)
    v0_path = Path("/tmp/lance_issue1_diag/V0_tlat4_13f_BASELINE_midframe.png")
    v1_path = Path("/tmp/lance_issue1_v1/frames17_tlat5_nlat11520_midframe.png")
    if v0_path.exists() and v1_path.exists():
        v0 = Image.open(v0_path)
        v1 = Image.open(v1_path)
        v_new = Image.open(png)
        W, H = v0.size
        pad = 30
        margin = 12
        grid = Image.new("RGB", (3*W + 4*margin, H + pad + 2*margin), "black")
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
        except Exception:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(grid)
        for i, (label, img) in enumerate([
            ("V0  t_lat=4  bf16  (baseline)",       v0),
            ("V1  t_lat=5  bf16  (regressed)",      v1),
            ("V1+ t_lat=5  attention_fp32=True",    v_new),
        ]):
            x = margin + i * (W + margin)
            y = margin + pad
            grid.paste(img, (x, y))
            draw.text((x+4, y - pad + 5), label, fill="white", font=font)
        grid_path = OUT / "H1_compare_grid.png"
        grid.save(grid_path)
        print(f"  → {grid_path}")

    print(f"\n┏━━ Verdict gates ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ Inspect H1_compare_grid.png:")
    print(f"┃   - If V1+fp32attn sky/water detail matches V0 → H1 CONFIRMED (bf16 attn accum)")
    print(f"┃   - If V1+fp32attn looks like V1 → H1 REJECTED; try H2 (LPE) or H3 (CFG renorm)")
    print(f"┃   - If V1+fp32attn is BETWEEN V0 and V1 → H1 contributes but isn't sole cause")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
