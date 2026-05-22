#!/usr/bin/env python3
"""Phase 5c-prep — static weight-stats profiler for Lance_3B.

For each Linear in Lance_3B bf16, computes:
  - Per-output-channel max-abs (which channels carry most signal)
  - Per-group max-abs at group_size=64 (the affine-quant grouping unit)
  - Per-group outlier ratio (max-abs / median-of-channel-max-abs)

The outlier ratio is the key signal: groups with high ratio contain a few
large weights that dominate the quantization grid, leaving the smaller
weights under-resolved. This is the mechanism by which naive affine
quantization degrades quality (per Reza2kn/lance-quant's findings on
o_proj/down_proj specifically).

Output: /tmp/lance_calibration/lance_3b_weight_stats.json

The JSON is consumed by future DWQ work to decide per-Linear:
  - group_size (smaller for outlier-heavy layers)
  - bits (higher for sensitive layers)
  - skip (keep at bf16) if outliers are too pathological

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/37_calibration_weight_stats.py \\
        [--lance-weights /Volumes/.../Lance-3B-bf16] \\
        [--out /tmp/lance_calibration/lance_3b_weight_stats.json] \\
        [--group-size 64] \\
        [--top-k 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16"),
                    help="bf16 Lance directory.")
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/lance_calibration/lance_3b_weight_stats.json"))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"┏━━ Phase 5c-prep — weight-stats profiler ━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ source     : {args.lance_weights}")
    print(f"┃ group_size : {args.group_size}")
    print(f"┃ out        : {args.out}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    import numpy as np

    print(f"\n=== Loading bf16 weights ===")
    t0 = time.perf_counter()
    saved = mx.load(str(args.lance_weights / "model.safetensors"))
    print(f"  loaded {len(saved)} tensors in {time.perf_counter()-t0:.1f}s")

    # Walk all Linear-like keys (anything ending in .weight that's 2D)
    print(f"\n=== Profiling Linears ===")
    layers = {}
    t0 = time.perf_counter()
    for key, val in saved.items():
        if not key.endswith(".weight"):
            continue
        if val.ndim != 2:
            continue
        # nn.Linear stores weight as (out_features, in_features)
        out_features, in_features = val.shape
        if in_features % args.group_size != 0:
            continue
        path = key[: -len(".weight")]

        # Compute per-output-channel max-abs (axis=1 is in_features)
        abs_val = mx.abs(val.astype(mx.float32))
        per_out_max = mx.max(abs_val, axis=1)               # (out_features,)
        per_out_median_max = float(mx.median(per_out_max))  # scalar
        per_out_max_max = float(mx.max(per_out_max))         # scalar
        per_out_min_max = float(mx.min(per_out_max))         # scalar

        # Compute per-group max-abs at group_size=64 along in_features axis
        # Reshape weight to (out_features, num_groups, group_size)
        num_groups = in_features // args.group_size
        grouped = val.astype(mx.float32).reshape(out_features, num_groups, args.group_size)
        per_group_max = mx.max(mx.abs(grouped), axis=2)      # (out_features, num_groups)
        # Per-group outlier ratio: max-abs vs median-of-group-max-abs
        group_median = float(mx.median(per_group_max))
        group_max = float(mx.max(per_group_max))
        outlier_ratio = group_max / max(group_median, 1e-8)

        layers[path] = {
            "shape": [int(out_features), int(in_features)],
            "num_groups": int(num_groups),
            "per_out_max_max": per_out_max_max,
            "per_out_max_min": per_out_min_max,
            "per_out_max_median": per_out_median_max,
            "group_max": group_max,
            "group_median": group_median,
            "outlier_ratio": outlier_ratio,
        }

    dt = time.perf_counter() - t0
    print(f"  profiled {len(layers)} Linears in {dt:.1f}s")

    # Rank by outlier ratio — these are the layers most damaged by naive
    # affine quantization (a few big weights swamp the per-group scale).
    ranked = sorted(layers.items(), key=lambda kv: kv[1]["outlier_ratio"], reverse=True)

    print(f"\n=== Top {args.top_k} most outlier-heavy Linears (highest ratio = most quant-sensitive) ===")
    print(f"{'#':>3}  {'path':<60s}  {'shape':<14s}  {'outlier_ratio':>14s}  {'group_max':>10s}")
    for i, (path, stats) in enumerate(ranked[: args.top_k]):
        shape = f"{stats['shape'][0]}x{stats['shape'][1]}"
        print(f"{i+1:>3}  {path:<60s}  {shape:<14s}  {stats['outlier_ratio']:>14.2f}  {stats['group_max']:>10.3f}")

    # Tower / module-type aggregation
    print(f"\n=== Outlier ratio by module category ===")
    categories = {}
    for path, stats in layers.items():
        # Determine category from path
        if "_moe_gen" in path:
            tower = "GEN"
        elif "lm_head" in path:
            tower = "LM_HEAD"
        elif "vae_in_proj" in path or "llm2vae" in path or "time_embedder" in path:
            tower = "ADAPTER"
        else:
            tower = "UND"

        if "q_proj" in path:
            mtype = "q_proj"
        elif "k_proj" in path:
            mtype = "k_proj"
        elif "v_proj" in path:
            mtype = "v_proj"
        elif "o_proj" in path:
            mtype = "o_proj"
        elif "gate_proj" in path:
            mtype = "gate_proj"
        elif "down_proj" in path:
            mtype = "down_proj"
        elif "up_proj" in path:
            mtype = "up_proj"
        elif "lm_head" in path:
            mtype = "lm_head"
        else:
            mtype = "other"

        key = f"{tower}.{mtype}"
        categories.setdefault(key, [])
        categories[key].append(stats["outlier_ratio"])

    for cat in sorted(categories.keys()):
        vals = categories[cat]
        if not vals:
            continue
        print(f"  {cat:<22s}  n={len(vals):>3d}  "
              f"min={min(vals):>6.2f}  median={float(np.median(vals)):>6.2f}  max={max(vals):>6.2f}")

    # Save JSON for downstream tools
    output = {
        "source": str(args.lance_weights),
        "group_size": args.group_size,
        "n_linears": len(layers),
        "layers": layers,
        "ranked_top_k": [
            {"path": path, **stats} for path, stats in ranked[: args.top_k]
        ],
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n→ {args.out}  ({args.out.stat().st_size/1024:.0f} KB)")

    print(f"\n┏━━ Verdict ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ {len(layers)} Linears profiled.")
    print(f"┃ Highest outlier ratio: {ranked[0][1]['outlier_ratio']:.1f}× at {ranked[0][0]}")
    print(f"┃ Median ratio across all: {float(np.median([s['outlier_ratio'] for s in layers.values()])):.2f}")
    print(f"┃")
    print(f"┃ Layers with outlier_ratio > 100 are strong candidates for DWQ-only")
    print(f"┃ quantization (or bf16 skip). Per Reza2kn evidence, o_proj + down_proj")
    print(f"┃ are the dominant sensitivities in Lance.")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
