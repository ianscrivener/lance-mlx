#!/usr/bin/env python3
"""Phase 5f — RockTalk weights through OUR pipeline at 256²×17f.

Definitive triangulation experiment. Loads RockTalk's bf16 (well, F32-cast-
to-bf16) weights via our LanceModel and runs the exact same red-panda-
surfing config we ran for V0_baseline in scripts/23_gutcheck_phase5g.py.

The remap step (scripts/22_remap_rocktalk.py) is invoked as a subprocess
on first run; subsequent runs skip the remap if the output dir exists.

Verdicts:
  - RT pipeline output SHARP (panda clearly recognizable, water clean):
    → Bug is in OUR converter (scripts/02_convert.py loses precision).
      Compare key-by-key between RT's remapped weights and our converted
      Lance-3B-Video-bf16/model.safetensors to find which tensors differ.

  - RT pipeline output BLURRY (same painterly aesthetic as V0_baseline):
    → Bug is in OUR pipeline forward-pass code (t2v.py). Both ports use
      equivalent weights; the deviation is in our forward pass. Need to
      compare against P1 (VAE feat_cache), P2 (TimestepEmbedder t*1000),
      or chat-template logic.

  - RT pipeline FAILS to load (key mismatch):
    → Remap script bug. Fix the remap rules in scripts/22_remap_rocktalk.py.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


SRC_DIR = Path("/Volumes/DEV_VOL1/VideoResearch/rocktalk-weights")
OUT_DIR = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16-RT")
# IMPORTANT: RT ships their OWN Wan2.2 VAE port with different key layout
# (decoder.upsamples.X.upsamples.Y.residual.layer_Z) vs mlx-video's
# Wan22VAEDecoder (decoder.head/middle/conv1). Their VAE won't load through
# our pipeline without a separate port. For Phase 5f we only care about the
# LLM forward pass, so use OUR working VAE (same one V0_baseline used).
OUR_BASELINE_VAE = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors")
GUT_DIR = Path("/tmp/lance_phase5g")  # for V0_baseline comparison
OUT_PHASE5F = Path("/tmp/lance_phase5f")

ORACLE_PROMPT_FILE = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
    "t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630/prompt.json"
)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    OUT_PHASE5F.mkdir(parents=True, exist_ok=True)
    prompt = json.loads(ORACLE_PROMPT_FILE.read_text())["000000.mp4"]

    print(f"┏━━ Phase 5f — RockTalk weights × our pipeline ━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ scale : 17f × 256×256, 30 steps, CFG=4.0, seed=42")
    print(f"┃ MaPE  : None (Phase 5d default)")
    print(f"┃ sms   : 1 (Phase 5g default — sms=2 was refuted)")
    print(f"┃ comparison: /tmp/lance_phase5g/V0_baseline_midframe.png")
    print(f"┃    md5(V0) : {md5_file(GUT_DIR / 'V0_baseline_midframe.png') if (GUT_DIR / 'V0_baseline_midframe.png').exists() else 'NOT FOUND'}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # --- 1. Remap RockTalk weights (idempotent — skip if already done) -----
    remap_done_marker = OUT_DIR / "model.safetensors"
    if remap_done_marker.exists():
        print(f"\n=== Remap already done — {OUT_DIR} ===")
    else:
        print(f"\n=== Remapping RockTalk weights → our layout ===")
        t0 = time.perf_counter()
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "22_remap_rocktalk.py"),
            "--src-dir", str(SRC_DIR),
            "--out-dir", str(OUT_DIR),
            "--include-vae",
        ]
        result = subprocess.run(cmd, env={"HF_HUB_DISABLE_XET": "1", **dict(__import__('os').environ)})
        if result.returncode != 0:
            print(f"  remap failed (rc={result.returncode})")
            return 1
        print(f"  remap completed in {time.perf_counter()-t0:.1f}s")

    # --- 2. Run t2v with the remapped weights ------------------------------
    print(f"\n=== Loading TextToVideoPipeline with REMAPPED RockTalk weights ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    vae_path = OUR_BASELINE_VAE  # RT's VAE has incompatible keys; use ours
    print(f"  lance-weights: {OUT_DIR}")
    print(f"  vae-weights:   {vae_path}  (OUR mlx-video Wan22 VAE — RT's has different layout)")
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=OUT_DIR,
        vae_safetensors=vae_path,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== Generating with RT weights (same config as V0_baseline) ===")
    t0 = time.perf_counter()
    frames = pipe.generate(
        prompt,
        num_frames=17, height=256, width=256,
        num_steps=30, cfg_scale=4.0,
        seed=42, verbose=False,
        mape_anchor=None,        # Phase 5d default
        spatial_merge_size=1,    # Phase 5g verdict
        rope_fp32=False,         # Phase 5g verdict
    )
    dt = time.perf_counter() - t0
    print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

    # --- 3. Save MP4 + midframe ---------------------------------------------
    import imageio
    import numpy as np
    from PIL import Image

    mp4 = OUT_PHASE5F / "RT_through_our_pipeline.mp4"
    with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
        for fr in frames:
            w.append_data(np.asarray(fr))
    print(f"  → {mp4} ({mp4.stat().st_size/1e3:.0f} KB)")

    mid = int(frames.shape[0] // 2)
    png = OUT_PHASE5F / "RT_through_our_pipeline_midframe.png"
    Image.fromarray(np.asarray(frames[mid])).save(png)
    print(f"  → {png}")

    # --- 4. Compare against V0_baseline -------------------------------------
    v0_png = GUT_DIR / "V0_baseline_midframe.png"
    rt_md5 = md5_file(png)
    if v0_png.exists():
        v0_md5 = md5_file(v0_png)
        print(f"\n┏━━ Comparison vs V0_baseline ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"┃ V0 baseline (our weights, our pipeline): {v0_md5}")
        print(f"┃ RT remapped (RT weights, our pipeline) : {rt_md5}")
        if v0_md5 == rt_md5:
            print(f"┃ → BYTE-IDENTICAL. Conversion produces same numerical values.")
            print(f"┃   If V0 is blurry, then RT pipeline is too. Bug must be")
            print(f"┃   in our pipeline forward-pass, NOT our converter.")
        else:
            print(f"┃ → DIFFERENT. Weights diverge enough to change pixels.")
            print(f"┃   Visually inspect:")
            print(f"┃     V0: {v0_png}")
            print(f"┃     RT: {png}")
        print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # --- 5. Build side-by-side compare PNG ----------------------------------
    if v0_png.exists():
        v0_img = Image.open(v0_png)
        rt_img = Image.open(png)
        W, H = v0_img.size
        pad = 30
        margin = 12
        from PIL import ImageDraw, ImageFont
        grid = Image.new('RGB', (2*W + 3*margin, H + pad + 2*margin), 'black')
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
        except Exception:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(grid)
        for i, (label, img) in enumerate([
            ("V0: our weights × our pipeline",  v0_img),
            ("RT: RT weights × our pipeline",   rt_img),
        ]):
            x = margin + i * (W + margin)
            y = margin + pad
            grid.paste(img, (x, y))
            draw.text((x+5, y - pad + 5), label, fill='white', font=font)
        grid_path = OUT_PHASE5F / "compare_grid.png"
        grid.save(grid_path)
        print(f"\nSide-by-side: {grid_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
