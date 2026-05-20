#!/usr/bin/env python3
"""Phase 2 — x2t_image VQA demo.

Loads the converted Lance_3B-bf16 LLM + the bundled Lance-3B-Video ViT,
runs greedy-decoded VQA on a Phase 0 oracle image, prints the decoded
answer alongside the oracle's expected answer.

Usage:
    HF_HUB_DISABLE_XET=1 \\
    uv run python scripts/04_x2t_image_demo.py \\
        --case 02 \\
        --lance-weights /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16 \\
        --vit-weights   /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16/vit.safetensors

The oracle's prompt + expected answer are read from
`tests/fixtures/results/x2t_image_sample_*/result.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="02",
                    help="Oracle case to run (01..06). Default: 02 (shortest answer).")
    ap.add_argument("--lance-weights", type=Path, required=True)
    ap.add_argument("--vit-weights", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path,
                    default=Path("tests/fixtures/images"))
    ap.add_argument("--results-glob", default="tests/fixtures/results/x2t_image_sample_*")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # --- Find the oracle entry for the requested case --------------------
    result_dirs = sorted(Path(".").glob(args.results_glob))
    if not result_dirs:
        print(f"ERROR: no result dir matched {args.results_glob}", file=sys.stderr)
        return 1
    result_json = result_dirs[0] / "result.json"
    oracle = json.loads(result_json.read_text())
    case_filename = f"image-understanding-case-{args.case}.png"
    case_entry = next(
        (e for e in oracle if e["image"].endswith(case_filename)), None,
    )
    if case_entry is None:
        print(f"ERROR: case {args.case} not found in {result_json}", file=sys.stderr)
        return 1

    image_path = args.images_dir / case_filename
    if not image_path.exists():
        print(f"ERROR: {image_path} not found", file=sys.stderr)
        return 1

    print(f"=== Oracle case {args.case} ===")
    print(f"  image:    {image_path}")
    print(f"  question: {case_entry['question']!r}")
    print(f"  expected: {case_entry['answer']!r}")

    # --- Load pipeline (heavy) -------------------------------------------
    print(f"\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vit_safetensors=args.vit_weights,
    )
    t1 = time.perf_counter()
    print(f"  loaded in {t1-t0:.1f}s")

    # --- Generate ----------------------------------------------------------
    print(f"\n=== Generating (max {args.max_new_tokens} tokens, greedy, no KV cache) ===")
    image = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()
    answer = pipe.generate(
        image, case_entry["question"],
        max_new_tokens=args.max_new_tokens, verbose=args.verbose,
    )
    t1 = time.perf_counter()
    print(f"  generated in {t1-t0:.1f}s")

    print(f"\n=== Result ===")
    print(f"  Lance MLX:  {answer!r}")
    print(f"  Oracle:     {case_entry['answer']!r}")
    same = answer.strip() == case_entry["answer"].strip()
    print(f"  Match:      {'✓ EXACT' if same else '✗ different'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
