#!/usr/bin/env python3
"""L6 — t2i quality A/B across bf16, 8-bit, 4-bit-und, 4-bit-full.

Validate that 4-bit quantization preserves t2i quality. Per Reza2kn
evidence, full 4-bit on the GEN tower may degrade. The UND-only variant
keeps GEN at bf16 (modest size savings, full quality). The full variant
is much smaller (~28% of bf16) but may show GEN-degradation artifacts.

Test on the cat-STOP-sign oracle prompt 000001 at 768², seed=42, 30 steps,
CFG=4 — same prompt + config as L5-prep so we can compare directly to
existing bf16 baseline.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> int:
    MODELS_ROOT = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models")
    VAE_WEIGHTS = MODELS_ROOT / "Wan22-VAE-bf16/vae.safetensors"
    ORACLE_DIR = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
        "t2i_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_084800"
    )
    OUT = Path("/tmp/lance_L6_quant_compare")
    OUT.mkdir(parents=True, exist_ok=True)

    PROMPT_ID = "000001.png"
    PROMPT = json.loads((ORACLE_DIR / "prompt.json").read_text())[PROMPT_ID]

    variants = [
        ("V0_bf16",       "Lance-3B-bf16"),
        ("V1_8bit",       "Lance-3B-8bit"),
        ("V2_4bit_und",   "Lance-3B-4bit-und"),
        ("V3_4bit_full",  "Lance-3B-4bit-full"),
    ]

    print(f"┏━━ L6 t2i quant A/B at 768² ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompt: {PROMPT!r}")
    print(f"┃ config: 768×768, 30 steps, CFG=4.0, seed=42, latent_pos_base=None (legacy)")
    print(f"┃ variants:")
    for name, sub in variants:
        d = MODELS_ROOT / sub
        if d.exists():
            mb = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / (1024*1024)
            print(f"┃   {name:14s}  {sub:24s} ({mb/1024:.1f} GB)")
        else:
            print(f"┃   {name:14s}  {sub:24s} (MISSING)")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    from PIL import Image

    md5s = {}
    timings = {}
    for name, sub in variants:
        d = MODELS_ROOT / sub
        if not d.exists():
            print(f"\n!!! Skipping {name}: {d} not found")
            continue
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        from lance_mlx.pipeline.t2i import TextToImagePipeline
        pipe = TextToImagePipeline.from_pretrained(
            lance_weights_dir=d,
            vae_safetensors=VAE_WEIGHTS,
        )
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")
        t0 = time.perf_counter()
        img = pipe.generate(
            PROMPT,
            height=768, width=768,
            num_steps=30, cfg_scale=4.0,
            seed=42, verbose=False,
        )
        dt = time.perf_counter() - t0
        timings[name] = dt
        out_path = OUT / f"{name}.png"
        img.save(out_path)
        md5s[name] = hashlib.md5(out_path.read_bytes()).hexdigest()
        print(f"  → {out_path}  ({dt:.1f}s, md5={md5s[name][:16]})")
        # Free pipeline memory between variants
        del pipe
        import gc
        gc.collect()
        mx.metal.clear_cache()

    # Build comparison grid: 1 row × 5 cols (oracle | V0..V3)
    print(f"\n=== Building compare grid ===")
    cols = [("ORACLE (Phase 0 ref)", Image.open(ORACLE_DIR / PROMPT_ID))]
    for name, _ in variants:
        p = OUT / f"{name}.png"
        if p.exists():
            cols.append((name, Image.open(p)))
    W = cols[0][1].size[0]
    H = cols[0][1].size[1]
    pad = 30
    margin = 12
    grid_w = len(cols) * W + (len(cols) + 1) * margin
    grid_h = H + pad + 2 * margin
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i, (label, img) in enumerate(cols):
        x = margin + i * (W + margin)
        y = margin + pad
        img_resized = img.resize((W, H))
        grid.paste(img_resized, (x, y))
        draw.text((x + 4, y - pad + 5), label, fill='white', font=font)
    grid_path = OUT / "compare_grid.png"
    grid.save(grid_path)
    print(f"  → {grid_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for name, _ in variants:
        if name in md5s:
            print(f"┃ {name:14s}  {timings[name]:5.1f}s  md5={md5s[name]}")
    print(f"┃")
    print(f"┃ Inspect {grid_path} for quality comparison.")
    print(f"┃ Per Reza2kn: 4-bit full may show GEN-degradation; 4-bit-und should match bf16.")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
