#!/usr/bin/env python3
"""L5-prep: test t2i.py with latent_pos_base=0 vs legacy default.

t2i.py has the same single-trailing-latent-block structure as t2v.py
(no instruction text after the latent block, unlike image_edit which has
clean+instruction+noisy interleaved). The Phase 5j fix `latent_pos_base=0`
SHOULD apply cleanly here. Question: does it improve output quality?

Tests on the cat-STOP-sign oracle prompt (000001), Lance_3B, 768², seed=42.

Three variants:
  V0_legacy  (latent_pos_base=None) — production default since Phase 3e
  V1_pos0    (latent_pos_base=0)   — Phase 5j-style fix
  V2_baseline — load the published Phase 0 oracle for visual reference

Compare visually; MD5 V0 against existing test fixture for determinism.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> int:
    LANCE_WEIGHTS = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16"
    )
    VAE_WEIGHTS = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors"
    )
    ORACLE_DIR = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
        "t2i_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_084800"
    )
    OUT = Path("/tmp/lance_L5_t2i_pos_base")
    OUT.mkdir(parents=True, exist_ok=True)

    PROMPT_ID = "000001.png"        # cat-with-STOP-poster (simple subject)
    PROMPT = json.loads((ORACLE_DIR / "prompt.json").read_text())[PROMPT_ID]

    print(f"┏━━ L5-prep t2i pos_base A/B at 768² ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompt: {PROMPT!r}")
    print(f"┃ config: 768×768, 30 steps, CFG=4.0, seed=42")
    print(f"┃ V0: latent_pos_base=None (legacy, production default)")
    print(f"┃ V1: latent_pos_base=0    (Phase 5j-style fix)")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    from PIL import Image

    print(f"\n=== Loading t2i pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    md5s = {}
    for label, base in [("V0_legacy", None), ("V1_pos0", 0)]:
        print(f"\n=== {label} (latent_pos_base={base}) ===")
        t0 = time.perf_counter()
        img = pipe.generate(
            PROMPT,
            height=768, width=768,
            num_steps=30, cfg_scale=4.0,
            seed=42, verbose=False,
            latent_pos_base=base,
        )
        dt = time.perf_counter() - t0
        out_path = OUT / f"{label}.png"
        img.save(out_path)
        md5s[label] = hashlib.md5(out_path.read_bytes()).hexdigest()
        print(f"  → {out_path}  ({dt:.1f}s, md5={md5s[label][:16]})")

    # 2-row compare grid for visual inspection.
    print(f"\n=== Building compare grid ===")
    a = Image.open(OUT / "V0_legacy.png")
    b = Image.open(OUT / "V1_pos0.png")
    oracle = Image.open(ORACLE_DIR / PROMPT_ID)
    W, H = a.size
    pad = 30
    margin = 12
    # 1 row × 3 cols  (oracle | V0 | V1)  for direct side-by-side
    grid_w = 3 * W + 4 * margin
    grid_h = H + pad + 2 * margin
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i, (label, img) in enumerate([
        ("ORACLE (Phase 0 reference)", oracle.resize((W, H))),
        ("V0 LEGACY (production default)", a),
        ("V1 POS_BASE=0 (Phase 5j-style)", b),
    ]):
        x = margin + i * (W + margin)
        y = margin + pad
        grid.paste(img, (x, y))
        draw.text((x + 4, y - pad + 5), label, fill='white', font=font)
    grid_path = OUT / "compare_grid.png"
    grid.save(grid_path)
    print(f"  → {grid_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for k, v in md5s.items():
        print(f"┃ {k:14s} md5={v}")
    if len(set(md5s.values())) == 1:
        print(f"┃ → BYTE-IDENTICAL. latent_pos_base flag has no effect (unexpected).")
    else:
        print(f"┃ → DIFFERENT. Flag is wired; inspect compare_grid.png to judge.")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
