#!/usr/bin/env python3
"""Phase 5c-prep — output-parity tester.

Runs t2i with two model variants on the SAME prompt + seed, computes
pixel-MAD between output images. Provides a parity metric that any
quantization experiment can use to measure how far it diverges from bf16.

Typical use:
    bf16 vs 8-bit Lance-3B  → measure quant degradation
    bf16 vs DWQ-future       → measure DWQ improvement

Default: compares bf16 against the broken 8-bit so we have a baseline
'how bad is current quant' number to beat with DWQ.

Reports:
  - Pixel MAD (mean absolute difference, 0-255 scale)
  - Pixel MAX-AD (max absolute difference)
  - Per-channel MAD (R, G, B)
  - SSIM-like quick proxy (correlation across image)

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/38_quant_parity_test.py \\
        --bf16-weights /Volumes/.../Lance-3B-bf16 \\
        --quant-weights /Volumes/.../Lance-3B-8bit \\
        [--prompt-id 000001.png] \\
        [--out /tmp/lance_calibration/parity_report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16"))
    ap.add_argument("--quant-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-8bit"))
    ap.add_argument("--vae-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors"))
    ap.add_argument("--prompt-id", type=str, default="000001.png")
    ap.add_argument("--prompt-file", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
                                 "t2i_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_084800/prompt.json"))
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/lance_calibration/parity_report.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_imgs_dir = args.out.parent / "parity_imgs"
    out_imgs_dir.mkdir(parents=True, exist_ok=True)

    PROMPT = json.loads(args.prompt_file.read_text())[args.prompt_id]

    print(f"┏━━ Phase 5c-prep — output-parity tester ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ bf16   : {args.bf16_weights.name}")
    print(f"┃ quant  : {args.quant_weights.name}")
    print(f"┃ prompt : {args.prompt_id}: {PROMPT[:60]}{'...' if len(PROMPT) > 60 else ''}")
    print(f"┃ config : {args.height}×{args.width}, {args.num_steps} steps, "
          f"CFG={args.cfg}, seed={args.seed}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from lance_mlx.pipeline.t2i import TextToImagePipeline

    def run_variant(label, weights_dir):
        print(f"\n=== {label} ({weights_dir.name}) ===")
        t0 = time.perf_counter()
        pipe = TextToImagePipeline.from_pretrained(
            lance_weights_dir=weights_dir,
            vae_safetensors=args.vae_weights,
        )
        img = pipe.generate(
            PROMPT,
            height=args.height, width=args.width,
            num_steps=args.num_steps, cfg_scale=args.cfg,
            seed=args.seed, verbose=False,
        )
        dt = time.perf_counter() - t0
        out_path = out_imgs_dir / f"{label}.png"
        img.save(out_path)
        print(f"  generated + saved in {dt:.1f}s → {out_path}")
        del pipe
        import gc
        gc.collect()
        mx.metal.clear_cache()
        return np.array(img, dtype=np.int32)   # int32 to allow signed subtraction

    bf16_arr = run_variant("bf16", args.bf16_weights)
    quant_arr = run_variant("quant", args.quant_weights)

    if bf16_arr.shape != quant_arr.shape:
        print(f"⚠ shape mismatch: bf16={bf16_arr.shape} vs quant={quant_arr.shape}")
        return 1

    print(f"\n=== Parity metrics ===")
    diff = np.abs(bf16_arr - quant_arr)
    overall_mad = float(diff.mean())
    overall_max = float(diff.max())
    print(f"  Pixel MAD (mean abs diff, 0-255): {overall_mad:.2f}")
    print(f"  Pixel MAX abs diff:               {overall_max}")

    # Per-channel breakdown
    for i, ch in enumerate("RGB"):
        ch_mad = float(diff[..., i].mean())
        ch_max = float(diff[..., i].max())
        print(f"  Channel {ch}: MAD={ch_mad:.2f}  MAX={ch_max}")

    # Quick correlation proxy: how often pixels are within 5 / 10 / 20 units
    within_5  = float((diff <= 5).mean())  * 100
    within_10 = float((diff <= 10).mean()) * 100
    within_20 = float((diff <= 20).mean()) * 100
    print(f"\n  % pixels within ±5  units:  {within_5:.1f}%")
    print(f"  % pixels within ±10 units:  {within_10:.1f}%")
    print(f"  % pixels within ±20 units:  {within_20:.1f}%")

    # Diff visualization (heatmap of |bf16 - quant| as grayscale, ×4 for visibility)
    diff_vis = np.clip(diff.mean(axis=-1) * 4, 0, 255).astype(np.uint8)
    diff_path = out_imgs_dir / "diff_heatmap.png"
    Image.fromarray(diff_vis, mode="L").save(diff_path)
    print(f"\n  → diff heatmap (×4 brightness): {diff_path}")

    # Side-by-side compare
    W, H = bf16_arr.shape[1], bf16_arr.shape[0]
    margin = 8
    pad = 30
    grid = Image.new("RGB", (3*W + 4*margin, H + pad + 2*margin), "black")
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i, (label, img) in enumerate([
        ("bf16 (reference)", Image.open(out_imgs_dir / "bf16.png")),
        ("quantized",        Image.open(out_imgs_dir / "quant.png")),
        ("|diff| ×4 (gray)", Image.open(out_imgs_dir / "diff_heatmap.png").convert("RGB")),
    ]):
        x = margin + i * (W + margin)
        y = margin + pad
        grid.paste(img, (x, y))
        draw.text((x + 4, y - pad + 5), label, fill="white", font=font)
    grid_path = out_imgs_dir / "parity_grid.png"
    grid.save(grid_path)
    print(f"  → 3-up grid: {grid_path}")

    # Write JSON report
    report = {
        "bf16_source": str(args.bf16_weights),
        "quant_source": str(args.quant_weights),
        "prompt_id": args.prompt_id,
        "prompt": PROMPT,
        "config": {
            "height": args.height, "width": args.width,
            "num_steps": args.num_steps, "cfg_scale": args.cfg,
            "seed": args.seed,
        },
        "metrics": {
            "pixel_mad": overall_mad,
            "pixel_max_ad": int(overall_max),
            "channel_R_mad": float(diff[..., 0].mean()),
            "channel_G_mad": float(diff[..., 1].mean()),
            "channel_B_mad": float(diff[..., 2].mean()),
            "within_5_pct":  within_5,
            "within_10_pct": within_10,
            "within_20_pct": within_20,
        },
        "interpretation": (
            "MAD < 2 → indistinguishable from bf16 (target for production quant). "
            "MAD 2-5 → minor degradation (acceptable for some uses). "
            "MAD > 10 → visibly different image (current 8-bit territory)."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n→ {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
