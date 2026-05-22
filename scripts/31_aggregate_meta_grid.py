#!/usr/bin/env python3
"""Aggregate a META_GRID.png from existing per-prompt midframe pairs.

Used when the oracle-pass script is invoked one-prompt-at-a-time — each
single-prompt invocation rebuilds the meta-grid with only its own results,
so we need a separate aggregator that walks the output dir and stitches
ALL existing {pid}_oracle_mid.png + {pid}_ours_mid.png pairs into one grid.

Usage:
    python scripts/31_aggregate_meta_grid.py /tmp/lance_oracle_pass_seed43
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: 31_aggregate_meta_grid.py <out_dir>")
        return 1
    out_dir = Path(sys.argv[1])
    if not out_dir.is_dir():
        print(f"Not a directory: {out_dir}")
        return 1

    from PIL import Image, ImageDraw, ImageFont

    # Find all {pid}_ours_mid.png with matching {pid}_oracle_mid.png
    pairs = []
    for ours_path in sorted(out_dir.glob("*_ours_mid.png")):
        pid_stem = ours_path.name.removesuffix("_ours_mid.png")
        oracle_path = out_dir / f"{pid_stem}_oracle_mid.png"
        if oracle_path.exists():
            pairs.append((pid_stem, oracle_path, ours_path))
    if not pairs:
        print(f"No paired midframes found in {out_dir}")
        return 1
    print(f"Found {len(pairs)} prompt pairs:")
    for pid, _, _ in pairs:
        print(f"  {pid}")

    # Load first to discover the size
    sample = Image.open(pairs[0][1])
    src_w, src_h = sample.size

    # Layout: 2 columns × N rows; thumbnail height ~256
    meta_h = 256
    meta_w = int(src_w * meta_h / src_h)
    label_h = 24
    margin = 8
    col_label_h = 36

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_row = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = font_row = ImageFont.load_default()

    grid_w = 2 * meta_w + 3 * margin
    grid_h = col_label_h + len(pairs) * (meta_h + label_h + margin) + margin
    grid = Image.new("RGB", (grid_w, grid_h), "black")
    draw = ImageDraw.Draw(grid)

    # Column headers
    draw.text((margin + 4, 6), "ORACLE (PyTorch ref)", fill="white", font=font)
    draw.text((margin + meta_w + 2 * margin + 4, 6), "OURS  (MLX Phase 5j)", fill="white", font=font)

    y = col_label_h
    for pid, oracle_path, ours_path in pairs:
        oracle = Image.open(oracle_path).resize((meta_w, meta_h), Image.LANCZOS)
        ours   = Image.open(ours_path).resize((meta_w, meta_h), Image.LANCZOS)
        draw.text((margin + 4, y), pid, fill="white", font=font_row)
        grid.paste(oracle, (margin, y + label_h))
        grid.paste(ours,   (margin + meta_w + 2 * margin, y + label_h))
        y += meta_h + label_h + margin

    out_path = out_dir / "META_GRID.png"
    grid.save(out_path)
    print(f"\n→ {out_path}  ({len(pairs)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
