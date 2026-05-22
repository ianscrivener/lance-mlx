#!/usr/bin/env python3
"""Phase 5k — full oracle pass at 768²×13f after Phase 5j fix.

Generates all 8 Phase 0 oracle prompts through the Phase 5j-fixed pipeline
(latent_pos_base=0 default) at 768×768 × 13 frames (max resolution within
our production envelope, n_lat ≤ 9,216). Builds per-prompt side-by-side
midframe grids vs the PyTorch oracle, plus a meta-grid of all 8.

Oracle was generated at 768²×49f. We compare midframes from both:
  - oracle midframe: frame 24 of 49 (≈ 2s into the clip)
  - ours midframe:   frame 6 of 13  (≈ midpoint of our shorter clip)

Wall-clock: ~2.5 min/prompt × 8 prompts ≈ 20 min total.

Outputs:
  /tmp/lance_oracle_pass/
    {prompt_id}.mp4              — our generated video
    {prompt_id}_midframe.png      — our midframe
    {prompt_id}_oracle_mid.png    — oracle midframe (resized to match)
    {prompt_id}_compare.png       — 2-row per-prompt side-by-side
    META_GRID.png                 — all 8 prompts in one image
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


ORACLE_DIR = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
    "t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630"
)
LANCE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16")
VAE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors")
OUT_DIR = Path("/tmp/lance_oracle_pass")

# Generation config — production envelope (n_lat = 13 × 48 × 48 = 9,216)
GEN_HEIGHT = 768
GEN_WIDTH = 768
GEN_FRAMES = 13
NUM_STEPS = 30
CFG_SCALE = 4.0
SEED = 42

# Oracle has 49 frames; for comparison we extract one midframe
ORACLE_MID_IDX = 24       # frame 24 of 49
OURS_MID_IDX = 6          # frame 6 of 13


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = json.loads((ORACLE_DIR / "prompt.json").read_text())
    prompt_ids = sorted(prompts.keys())   # e.g. ['000000.mp4', '000001.mp4', ...]

    print(f"┏━━ Phase 5k — Full oracle pass at 768²×13f ━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompts : {len(prompt_ids)}")
    print(f"┃ config  : {GEN_HEIGHT}×{GEN_WIDTH} × {GEN_FRAMES}f, "
          f"{NUM_STEPS} steps, CFG={CFG_SCALE}, seed={SEED}")
    print(f"┃ fix     : latent_pos_base=0 (default since Phase 5j)")
    print(f"┃ out dir : {OUT_DIR}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    import imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = font_small = ImageFont.load_default()

    results = []  # (prompt_id, oracle_img, ours_img, dt_sec)

    total_t0 = time.perf_counter()
    for pid in prompt_ids:
        prompt = prompts[pid]
        print(f"\n=== {pid} ===")
        print(f"  prompt: {prompt[:80]}...")

        # 1. Generate
        t0 = time.perf_counter()
        try:
            frames = pipe.generate(
                prompt,
                num_frames=GEN_FRAMES,
                height=GEN_HEIGHT,
                width=GEN_WIDTH,
                num_steps=NUM_STEPS,
                cfg_scale=CFG_SCALE,
                seed=SEED,
                verbose=False,
                # All other args take Phase 5j-fixed defaults:
                #   mape_anchor=None, latent_pos_base=0
            )
        except Exception as e:
            print(f"  GENERATION FAILED: {e!r}")
            continue
        dt = time.perf_counter() - t0
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        # 2. Save our MP4
        mp4 = OUT_DIR / f"{pid}"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))
        print(f"  → {mp4} ({mp4.stat().st_size/1e3:.0f} KB)")

        # 3. Save our midframe
        ours_idx = min(OURS_MID_IDX, int(frames.shape[0]) - 1)
        ours_mid = Image.fromarray(np.asarray(frames[ours_idx]))
        ours_png = OUT_DIR / f"{pid.replace('.mp4', '')}_ours_mid.png"
        ours_mid.save(ours_png)

        # 4. Extract oracle midframe
        oracle_mp4 = ORACLE_DIR / pid
        if not oracle_mp4.exists():
            print(f"  oracle file not found: {oracle_mp4}")
            continue
        oracle_reader = imageio.get_reader(oracle_mp4)
        try:
            oracle_arr = oracle_reader.get_data(ORACLE_MID_IDX)
        except Exception:
            # Fall back to last frame if mid is out of range
            oracle_arr = oracle_reader.get_data(oracle_reader.count_frames() - 1)
        finally:
            oracle_reader.close()
        oracle_mid = Image.fromarray(oracle_arr)
        # Resize oracle to match our resolution if needed
        if oracle_mid.size != ours_mid.size:
            oracle_mid_resized = oracle_mid.resize(ours_mid.size, Image.LANCZOS)
        else:
            oracle_mid_resized = oracle_mid
        oracle_png = OUT_DIR / f"{pid.replace('.mp4', '')}_oracle_mid.png"
        oracle_mid_resized.save(oracle_png)

        # 5. Build 2-row per-prompt compare grid
        W, H = ours_mid.size
        label_h = 32
        margin = 8
        comp = Image.new("RGB", (W + 2*margin, 2*(H + label_h + margin) + margin), "black")
        draw = ImageDraw.Draw(comp)
        for i, (label, img) in enumerate([
            (f"ORACLE (PyTorch reference, frame {ORACLE_MID_IDX}/49)", oracle_mid_resized),
            (f"OURS   (MLX Phase 5j fix,  frame {ours_idx}/{frames.shape[0]})", ours_mid),
        ]):
            y = margin + i * (H + label_h + margin)
            draw.text((margin + 4, y), label, fill="white", font=font)
            comp.paste(img, (margin, y + label_h))
        comp_path = OUT_DIR / f"{pid.replace('.mp4', '')}_compare.png"
        comp.save(comp_path)
        print(f"  → {comp_path}")

        results.append((pid, oracle_mid_resized, ours_mid, dt))

    total_dt = time.perf_counter() - total_t0
    print(f"\n=== All {len(results)}/{len(prompt_ids)} prompts done in {total_dt/60:.1f} min ===")

    # Build meta-grid: 2 columns (oracle | ours), N rows.
    print(f"\n=== Building META_GRID.png ===")
    if not results:
        print("  no results, abort")
        return 1
    # Shrink each midframe for the meta-grid
    meta_H = 192
    meta_W = int(GEN_WIDTH * meta_H / GEN_HEIGHT)
    label_h = 22
    margin = 8
    col_label_h = 32
    grid_w = 2 * meta_W + 3 * margin
    grid_h = col_label_h + len(results) * (meta_H + label_h + margin) + margin
    grid = Image.new("RGB", (grid_w, grid_h), "black")
    draw = ImageDraw.Draw(grid)
    # column headers
    draw.text((margin + 4,                       4), "ORACLE (PyTorch ref)", fill="white", font=font)
    draw.text((margin + meta_W + 2 * margin + 4, 4), "OURS  (MLX Phase 5j)", fill="white", font=font)
    y = col_label_h
    for pid, oracle, ours, dt in results:
        # row label inside the row
        draw.text((margin + 4, y), f"{pid}  ({dt:.0f}s)", fill="white", font=font_small)
        oracle_s = oracle.resize((meta_W, meta_H), Image.LANCZOS)
        ours_s   = ours.resize((meta_W, meta_H), Image.LANCZOS)
        grid.paste(oracle_s, (margin, y + label_h))
        grid.paste(ours_s,   (margin + meta_W + 2 * margin, y + label_h))
        y += meta_H + label_h + margin
    meta_path = OUT_DIR / "META_GRID.png"
    grid.save(meta_path)
    print(f"  → {meta_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for pid, _, _, dt in results:
        print(f"┃ {pid:14s} {dt:6.1f}s")
    print(f"┃")
    print(f"┃ Wall-clock total: {total_dt/60:.1f} min")
    print(f"┃ Per-prompt grids: {OUT_DIR}/*_compare.png")
    print(f"┃ Meta grid       : {OUT_DIR}/META_GRID.png")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
