#!/usr/bin/env python3
"""Issue #1 — n_lat ceiling diagnostic.

At 768²×13f (t_lat=4, n_lat=9216) t2v produces photoreal output.
At 768²×17f (t_lat=5, n_lat=11520) t2v degrades.
At 768²×49f (t_lat=13, n_lat=29952) t2v is pure noise.

Goal: identify where in the Euler loop the failing runs diverge from
the working run. Three hypotheses:
  H1 — velocity prediction becomes NaN/inf at some step
  H2 — latent magnitudes blow up (exploding) across steps
  H3 — output is bounded but somehow wrong (silent drift)

This script runs 4 variants at 768², logs per-step latent stats (min,
max, mean, std, |max-finite|), and saves both video + a per-step trace
JSON so we can compare trajectories.

Variants:
  V0 — t_lat=4  (13 frames)  — known photoreal baseline
  V1 — t_lat=5  (17 frames)  — known degraded (just past cliff)
  V2 — t_lat=6  (21 frames)  — degraded further into the bad zone
  V3 — t_lat=8  (29 frames)  — well into the bad zone

If V0 has stable stats but V1/V2/V3 blow up at the same step, that step's
forward path is the bug surface. If they all blow up at DIFFERENT steps,
the n_lat ceiling is a magnitude-dependent issue (some accumulation that
saturates faster at higher n_lat).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16"))
    ap.add_argument("--vae-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/lance_issue1_diag"))
    ap.add_argument("--prompt", type=str,
                    default="A medium-close shot shows a red panda wearing a gold-trimmed "
                            "cap and travel satchel on a bright seaside wave with a painted "
                            "surfboard, foam spray, and a glowing summer sky.")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # (label, num_frames). frames = (t_lat-1)*4 + 1
    variants = [
        ("V0_tlat4_13f_BASELINE",    13),    # known photoreal
        ("V1_tlat5_17f_DEGRADED",    17),    # just past cliff
        ("V2_tlat6_21f_FURTHER",     21),
        ("V3_tlat8_29f_DEEP",        29),
    ]

    print(f"┏━━ Issue #1 — n_lat ceiling diagnostic ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ scale : {args.height}×{args.width}, {args.num_steps} steps, "
          f"CFG={args.cfg}, seed={args.seed}")
    print(f"┃ variants (frames → t_lat → n_lat):")
    for label, frames in variants:
        t_lat = (frames - 1) // 4 + 1
        n_lat = t_lat * (args.height // 16) * (args.width // 16)
        print(f"┃   {label:30s} frames={frames}  t_lat={t_lat}  n_lat={n_lat}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    import numpy as np
    import imageio
    from PIL import Image

    # Monkey-patch the verbose path to ALSO capture per-step min/max + has-nan/inf
    from lance_mlx.pipeline import t2v as t2v_mod

    captured_traces: dict[str, list[dict[str, Any]]] = {}
    current_label: list[str] = [""]   # mutable holder

    # Wrap the generate method on TextToVideoPipeline to inject our per-step capture
    # by patching the inner _step_velocity call site is complex; simpler: monkey-patch
    # mx.eval through a hook isn't trivial either. We'll do it the explicit way:
    # patch the verbose-print block by overriding `print` builtin DURING generate.

    # Simplest: re-run the loop manually here. Need access to pipe internals.
    print(f"\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # Monkey-patch verbose printing to capture stats too
    import builtins
    real_print = builtins.print

    def trace_print(*pa, **kw):
        msg = " ".join(str(x) for x in pa)
        real_print(*pa, **kw)
        # Parse lines like "step N/M t=... mean=... std=..."
        if "step " in msg and "/" in msg and "t=" in msg and "mean=" in msg:
            try:
                # Crude parse
                parts = msg.strip().split()
                # parts: ['step', '1/30', 't=0.9902', 'dt=0.0102', 'mean=0.001', 'std=1.013']
                step_n = int(parts[1].split("/")[0])
                t_val = float(parts[2].split("=")[1])
                mean_val = float(parts[4].split("=")[1])
                std_val = float(parts[5].split("=")[1])
                if current_label[0]:
                    captured_traces.setdefault(current_label[0], []).append({
                        "step": step_n, "t": t_val,
                        "mean": mean_val, "std": std_val,
                    })
            except (IndexError, ValueError):
                pass

    builtins.print = trace_print

    try:
        for label, frames in variants:
            current_label[0] = label
            print(f"\n=== {label} (frames={frames}) ===")
            t0 = time.perf_counter()
            try:
                video = pipe.generate(
                    args.prompt,
                    num_frames=frames,
                    height=args.height, width=args.width,
                    num_steps=args.num_steps, cfg_scale=args.cfg,
                    seed=args.seed, verbose=True,
                    mape_anchor=None,
                )
            except Exception as e:
                print(f"  ⚠ GENERATION FAILED: {e!r}")
                continue
            dt = time.perf_counter() - t0
            print(f"  generated {video.shape[0]} frames in {dt:.1f}s")

            mp4 = args.out / f"{label}.mp4"
            with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
                for fr in video:
                    w.append_data(np.asarray(fr))
            mid = int(video.shape[0] // 2)
            png = args.out / f"{label}_midframe.png"
            Image.fromarray(np.asarray(video[mid])).save(png)
            print(f"  → {mp4}")

    finally:
        builtins.print = real_print

    # Write trace JSON
    trace_path = args.out / "trace.json"
    with open(trace_path, "w") as f:
        json.dump(captured_traces, f, indent=2)
    print(f"\n→ {trace_path}  ({len(captured_traces)} variants traced)")

    # Print step-trajectory summary
    print(f"\n=== Per-variant trajectory summary ===")
    print(f"{'variant':<32s}  {'step1_mean':>12s}  {'step15_mean':>12s}  "
          f"{'step30_mean':>12s}  {'max_std':>10s}")
    for label, _ in variants:
        trace = captured_traces.get(label, [])
        if not trace:
            print(f"  {label:<32s}  (no trace)")
            continue
        first = trace[0]
        mid_idx = len(trace) // 2
        mid = trace[mid_idx]
        last = trace[-1]
        max_std = max((t["std"] for t in trace), default=0.0)
        print(f"  {label:<32s}  {first['mean']:>12.4f}  {mid['mean']:>12.4f}  "
              f"{last['mean']:>12.4f}  {max_std:>10.2f}")

    print(f"\n┏━━ Inspect ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ Videos+midframes: {args.out}/")
    print(f"┃ Per-step trace : {trace_path}")
    print(f"┃ Look for:")
    print(f"┃   - V1+ traces showing |mean| or std diverging vs V0")
    print(f"┃   - Specific step where divergence starts (the bug surface)")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
