#!/usr/bin/env python3
"""Phase 2 — run x2t_image and x2t_video against Phase 0 fixtures.

Validation gate: ≥95% token agreement vs PyTorch reference at greedy decode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lance_mlx import Timer, log_run, peak_memory_gb

PHASE = "phase2"


def run_x2t(case: dict, weights_dir: Path, out_root: Path) -> dict:
    """Run one understanding case end-to-end.

    TODO(claude-code): wire to lance_mlx.pipeline.understanding once it lands.
        from lance_mlx.pipeline.understanding import UnderstandingPipeline
        pipe = UnderstandingPipeline.from_pretrained(weights_dir)
        result = pipe(prompt=case["prompt"], image=case.get("image"),
                      video=case.get("video"), do_sample=False)
    """
    run_id = f"{PHASE}_{case['task']}_{case['id']}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    with Timer("load") as t:
        # pipe = UnderstandingPipeline.from_pretrained(weights_dir)
        pipe = None
    timings["load"] = t.elapsed

    with Timer("vit_encode") as t:
        # vit_tokens = pipe.vit(case["image"]) if case.get("image") else None
        pass
    timings["vit_encode"] = t.elapsed

    with Timer("ar_decode") as t:
        # response = pipe.generate(prompt=case["prompt"], ...)
        response = None
    timings["ar_decode"] = t.elapsed

    peak = peak_memory_gb()
    log_run(
        run_id=run_id,
        model=str(weights_dir),
        task=case["task"],
        prompt=case["prompt"],
        seed=0,
        timings=timings,
        peak_rss_gb=peak,
        extra={"phase": PHASE, "case_id": case["id"]},
    )
    return {"response": response, "timings": timings, "peak_rss_gb": peak}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    # Load Phase 0 fixtures
    cases = []
    for task in ["x2t_image", "x2t_video"]:
        task_dir = args.fixtures_dir / task
        if not task_dir.exists():
            print(f"WARN: {task_dir} not found — run Phase 0 first", file=sys.stderr)
            continue
        for case_dir in sorted(task_dir.iterdir()):
            cfg_path = case_dir / "config.json"
            if cfg_path.exists():
                cases.append(json.loads(cfg_path.read_text()))

    if not cases:
        print("ERROR: no Phase 0 fixtures found", file=sys.stderr)
        return 1

    out_root = args.out / PHASE
    out_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        print(f"\n=== {case['task']}/{case['id']} ===")
        run_x2t(case, args.weights, out_root)

    print(f"\nDone. Compare outputs against tests/fixtures/ via tests/test_parity_understanding.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
