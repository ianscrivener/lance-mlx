#!/usr/bin/env python3
"""Issue #1 H2 — LPE per-frame static analysis.

Hypothesis: Lance_3B_Video's 126,976-entry latent_pos_embed table covers
31 temporal positions × 64 × 64 spatial. If the training distribution
skewed toward shorter clips (frame index 0..3 more common than 12+),
the higher-frame entries would carry systematically weaker training
signal — visible as lower magnitudes / different statistics per frame.

This would manifest as gradual quality decay when generating at t_lat > 4
(uses LPE rows for frame indices 4+) — exactly the issue #1 pattern.

For each frame index 0..30, computes:
  - L2 norm per row, averaged across all 4096 (h*w) rows
  - Per-row std (signal strength)
  - Per-frame absolute mean (offset from zero — indicates bias)
  - Per-frame max-abs (peak signal)

Output: stats per frame + visual sparkline showing trend.

If stats are FLAT → H2 rejected (LPE is not the cause).
If stats DECAY monotonically with frame index → strong evidence for
  undertrained-higher-frame hypothesis.
If stats show DISCONTINUITY at a specific frame → reveals a training
  threshold.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/42_issue1_h2_lpe_stats.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/lance_issue1_h2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"┏━━ Issue #1 H2 — LPE per-frame static analysis ━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ source : {args.lance_weights}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    import numpy as np

    print(f"\n=== Loading LPE table ===")
    saved = mx.load(str(args.lance_weights / "model.safetensors"))
    lpe_key = "latent_pos_embed.pos_embed"
    if lpe_key not in saved:
        print(f"  ⚠ {lpe_key} not in model.safetensors")
        return 1
    lpe = saved[lpe_key]
    print(f"  shape: {tuple(lpe.shape)}  dtype: {lpe.dtype}")
    # Expected: (126976, 2048) for video — 31 × 64 × 64 entries × hidden_size=2048

    N, D = lpe.shape
    if N != 31 * 64 * 64:
        print(f"  ⚠ Unexpected shape — expected {31*64*64}×2048 for Lance_3B_Video")
        return 1
    n_frames, h_grid, w_grid = 31, 64, 64

    # Reshape to (31, 4096, 2048) — per-frame slices
    lpe_per_frame = np.array(lpe.astype(mx.float32)).reshape(n_frames, h_grid * w_grid, D)
    print(f"  reshaped to {lpe_per_frame.shape} (frame, h*w-positions, hidden)")

    print(f"\n=== Per-frame statistics ===")
    stats = []
    for f in range(n_frames):
        slab = lpe_per_frame[f]                          # (4096, 2048)
        l2_per_row = np.linalg.norm(slab, axis=1)        # (4096,) — L2 norm per (h,w) position
        per_row_std = np.std(slab, axis=1)               # (4096,)
        frame_stats = {
            "frame": f,
            "mean_abs":     float(np.abs(slab).mean()),
            "max_abs":      float(np.abs(slab).max()),
            "row_l2_mean":  float(l2_per_row.mean()),
            "row_l2_std":   float(l2_per_row.std()),
            "row_std_mean": float(per_row_std.mean()),
            "global_std":   float(slab.std()),
        }
        stats.append(frame_stats)

    # Print table + ASCII sparkline of row_l2_mean
    print(f"{'frame':>5}  {'mean_abs':>10}  {'max_abs':>10}  {'row_l2_mean':>12}  "
          f"{'row_l2_std':>10}  {'global_std':>10}  visualize→row_l2_mean")
    max_l2 = max(s["row_l2_mean"] for s in stats)
    for s in stats:
        bar_len = int(40 * s["row_l2_mean"] / max_l2) if max_l2 > 0 else 0
        bar = "█" * bar_len
        print(f"{s['frame']:>5}  {s['mean_abs']:>10.4f}  {s['max_abs']:>10.4f}  "
              f"{s['row_l2_mean']:>12.4f}  {s['row_l2_std']:>10.4f}  "
              f"{s['global_std']:>10.4f}  {bar}")

    # JSON dump
    out_json = args.out / "lpe_per_frame_stats.json"
    with open(out_json, "w") as f:
        json.dump({
            "source": str(args.lance_weights),
            "lpe_shape": list(lpe.shape),
            "n_frames": n_frames,
            "stats": stats,
        }, f, indent=2)
    print(f"\n→ {out_json}")

    # Verdict
    print(f"\n┏━━ Verdict ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    first_4_l2 = np.mean([stats[f]["row_l2_mean"] for f in range(4)])
    next_4_l2  = np.mean([stats[f]["row_l2_mean"] for f in range(4, 8)])
    later_l2   = np.mean([stats[f]["row_l2_mean"] for f in range(8, 31)])
    print(f"┃ Frames 0-3   row_l2_mean: {first_4_l2:.4f}")
    print(f"┃ Frames 4-7   row_l2_mean: {next_4_l2:.4f}  (ratio: {next_4_l2/first_4_l2:.3f}×)")
    print(f"┃ Frames 8-30  row_l2_mean: {later_l2:.4f}  (ratio: {later_l2/first_4_l2:.3f}×)")
    print(f"┃")
    if abs(next_4_l2 / first_4_l2 - 1.0) < 0.1 and abs(later_l2 / first_4_l2 - 1.0) < 0.1:
        print(f"┃ → FLAT distribution. H2 REJECTED. LPE is not the n_lat-ceiling cause.")
    elif next_4_l2 < first_4_l2 * 0.7:
        print(f"┃ → DECAYING. H2 SUPPORTED — higher-frame LPE entries are weaker.")
        print(f"┃   Possible mitigations: scale LPE indices, skip LPE additive, or")
        print(f"┃   normalize LPE per-frame before lookup.")
    else:
        print(f"┃ → MIXED. Worth visual inspection of the table; not a smoking gun")
        print(f"┃   either way.")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
