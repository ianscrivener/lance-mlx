#!/usr/bin/env python3
"""Phase 5n / D1c — real-prompt t2i frame-index A/B.

D1b (noise-only) showed that for `T_latent=1` (the t2i regime) the Wan2.2 VAE
emits 3 output frames with materially different statistics:

    T_lat=1 frame 0:  mean +0.1100  std 0.2890  HF 9.20e+05   ← currently ships
    T_lat=1 frame 1:  mean +0.3761  std 0.3623  HF 1.14e+06
    T_lat=1 frame 2:  mean +0.4068  std 0.4349  HF 1.65e+06

`t2i.py:286` grabs `decoded[0, 0]`. Frame 2 has ~2× the high-frequency energy
of frame 0 in the noise-driven D1b test. This script does a real-prompt A/B:
runs the full t2i pipeline on 3 oracle prompts (low-res, deterministic),
captures the 3 decoded frames, and emits a comparison grid so a human can
judge which frame the t2i pipeline *should* be returning.

Cost: ~1–2 min per generation at 384², 30 steps. Total ~5 min.

Outputs land in notes/phase5n_diagnostics/d1c_real_prompt_frame_ab/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a standalone script: prepend src/ so `lance_mlx` resolves.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT.parent / "lance-mlx-models"
LANCE_WEIGHTS = MODELS_ROOT / "Lance-3B-bf16"
VAE_SAFETENSORS = MODELS_ROOT / "Wan22-VAE-bf16" / "vae.safetensors"
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d1c_real_prompt_frame_ab"


# Three oracle prompts, picked to cover photoreal, atmospheric/text, and
# stylized — the categories most sensitive to high-frequency detail and
# brightness/contrast bias (which is exactly what the 3 frames differ on).
PROMPTS = [
    (
        "p10_typography",
        "A vintage hand-lettered sign reading 'Open' hanging in a coffee shop "
        "window, warm interior lighting visible behind.",
    ),
    (
        "p01_fox_grass",
        "A red fox cautiously stepping through tall summer grass at golden "
        "hour, naturalistic, shallow depth of field.",
    ),
    (
        "p06_neon_alley",
        "Rain-soaked neon alley in a cyberpunk city at night, reflections "
        "in puddles, no people.",
    ),
]


class CapturingDecoder:
    """Transparent wrapper around `vae_decoder` that stashes the raw `decoded`
    tensor so we can pull frames 1 and 2 in addition to the frame-0 the
    pipeline returns."""

    def __init__(self, real):
        self._real = real
        self.last_decoded = None

    def __call__(self, z):
        out = self._real(z)
        self.last_decoded = out
        return out

    def __getattr__(self, name):
        return getattr(self._real, name)


def fft_hf_energy(img_hwc: np.ndarray) -> float:
    gray = img_hwc.mean(axis=-1)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.ones_like(mag)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    return float((mag * mask).sum())


def frame_to_pil(frame_hwc_fp32: np.ndarray) -> Image.Image:
    u8 = ((frame_hwc_fp32 + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(u8)


def label_image(img: Image.Image, text: str) -> Image.Image:
    """Stamp `text` in the top-left of `img` (returns a copy)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16
        )
    except OSError:
        font = ImageFont.load_default()
    # opaque-ish background strip for readability
    bbox = draw.textbbox((6, 4), text, font=font)
    draw.rectangle(
        [bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2],
        fill=(0, 0, 0, 180),
    )
    draw.text((6, 4), text, fill=(255, 255, 0), font=font)
    return out


def build_grid(rows: list[list[Image.Image]], pad: int = 6) -> Image.Image:
    """rows: list of equal-length lists of PIL images. Returns a single grid."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    cell_w, cell_h = rows[0][0].size
    w = n_cols * cell_w + (n_cols + 1) * pad
    h = n_rows * cell_h + (n_rows + 1) * pad
    out = Image.new("RGB", (w, h), (20, 20, 22))
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            out.paste(im, (x, y))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384,
                    help="Square output size; 256 or 384 recommended. Default 384.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5n / D1c — real-prompt t2i frame-index A/B ===")
    print(f"  size: {args.size}x{args.size}  steps: {args.steps}  "
          f"cfg: {args.cfg}  seed: {args.seed}")
    print(f"  out: {OUT_DIR}")

    print("\nLoading TextToImagePipeline ...")
    t_load = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter() - t_load:.1f}s")

    # Wrap the VAE decoder to capture the raw 3-frame output.
    capturing = CapturingDecoder(pipe.vae_decoder)
    pipe.vae_decoder = capturing

    grid_rows: list[list[Image.Image]] = []
    stats_lines: list[str] = []
    stats_lines.append(
        f"# D1c — real-prompt t2i frame-index A/B  ({args.size}², "
        f"{args.steps} steps, cfg={args.cfg}, seed={args.seed})\n"
    )

    for (pid, prompt) in PROMPTS:
        print(f"\n--- {pid} ---")
        print(f"  prompt: {prompt}")
        t0 = time.perf_counter()
        # `generate()` returns the frame-0 PIL image; we discard it because
        # we'll rebuild all three frames from the captured tensor.
        _ = pipe.generate(
            prompt,
            height=args.size, width=args.size,
            num_steps=args.steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        print(f"  generated in {dt:.1f}s")

        decoded = capturing.last_decoded            # (1, T', H, W, 3)
        assert decoded is not None
        decoded_np = np.array(decoded[0].astype(mx.float32))   # (T', H, W, 3)
        T_decoded = decoded_np.shape[0]
        print(f"  decoded shape: {decoded_np.shape}  (T'={T_decoded})")
        if T_decoded != 3:
            print(f"  WARNING: expected T'=3 from T_latent=1, got T'={T_decoded}")

        per_prompt = []
        stats_lines.append(f"\n## {pid}\n")
        stats_lines.append(f"prompt: {prompt}\n\n")
        stats_lines.append(
            f"| frame | mean   | std   | min    | max    | HF energy |\n"
            f"|-------|--------|-------|--------|--------|-----------|\n"
        )
        for fi in range(min(T_decoded, 3)):
            frame = decoded_np[fi]
            mean = float(frame.mean())
            std = float(frame.std())
            mn = float(frame.min())
            mx_ = float(frame.max())
            hf = fft_hf_energy(frame)
            stats_lines.append(
                f"| {fi}     | {mean:+.4f} | {std:.4f} | {mn:+.4f} | "
                f"{mx_:+.4f} | {hf:.2e} |\n"
            )
            print(f"  frame {fi}: mean={mean:+.4f}  std={std:.4f}  "
                  f"min={mn:+.4f}  max={mx_:+.4f}  HF={hf:.2e}")

            pil = frame_to_pil(frame)
            pil.save(OUT_DIR / f"{pid}_frame{fi}.png")
            labeled = label_image(pil, f"{pid}  frame {fi}")
            per_prompt.append(labeled)

        grid_rows.append(per_prompt)

    grid = build_grid(grid_rows)
    grid_path = OUT_DIR / "_grid_3prompts_x_3frames.png"
    grid.save(grid_path)
    print(f"\n✓ Grid saved: {grid_path}  size={grid.size}")

    stats_path = OUT_DIR / "stats.md"
    stats_path.write_text("".join(stats_lines))
    print(f"✓ Stats:      {stats_path}")

    print(
        "\nNext step: open the grid PNG and judge frame 0 vs 1 vs 2 visually.\n"
        "If frame 1 or 2 is clearly sharper / better composed, propose a\n"
        "one-line patch to t2i.py:286 and image_edit.py:302."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
