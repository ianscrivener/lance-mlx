#!/usr/bin/env python3
"""x2t_video oracle sweep — run all locally-available cases vs Phase 0.

Phase 2-extension validated only the cooking-video caption case. This
revisits the broader oracle set: runs x2t_video on every video in
tests/fixtures/video_understanding that has a matching oracle answer
in tests/fixtures/results/x2t_video_sample_*/prompt.json, prints both
side-by-side, and reports content-correctness.

The Phase 0 oracle expects strict "Answer: ..." token-format outputs for
VQA cases and longer paragraph captions for caption cases. Content match
is approximate — we report the full strings for human judgment.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ORACLE_DIR = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
    "x2t_video_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_085514"
)
VIDEOS_DIR = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/video_understanding"
)
LANCE_WEIGHTS = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16"
)
VIT_WEIGHTS = LANCE_WEIGHTS / "vit.safetensors"


# Mapping between (oracle key in prompt.json) and (local filename).
# Phase 0 used the "video-understanding-{kind}-{NN}.mp4" naming; we mirrored
# only some files locally under shorter names.
LOCAL_NAME_MAP = {
    "video-understanding-caption-short-01.mp4": "caption-short-01.mp4",
    "video-understanding-vqa-03.mp4":            "vqa-03.mp4",
}


def main() -> int:
    # Load oracle prompts + results
    expected = json.loads((ORACLE_DIR / "prompt.json").read_text())
    questions = {q["video"].rsplit("/", 1)[-1]: q["question"]
                 for q in json.loads((ORACLE_DIR / "result.json").read_text())}

    cases = []
    for oracle_key, local_name in LOCAL_NAME_MAP.items():
        vp = VIDEOS_DIR / local_name
        if not vp.exists():
            print(f"⚠ skip {oracle_key}: local file {vp} not found")
            continue
        if oracle_key not in expected:
            print(f"⚠ skip {oracle_key}: no oracle answer")
            continue
        q = questions.get(oracle_key)
        if q is None:
            # Fallback for caption-style cases that may not have explicit question
            q = "Describe what happens in this video."
        cases.append((oracle_key, vp, q, expected[oracle_key]))

    if not cases:
        print("No runnable cases.")
        return 1

    print(f"┏━━ x2t_video oracle sweep ({len(cases)} locally-available cases) ━━━━━━")
    for k, v, _, _ in cases:
        print(f"┃ {k}  ←  {v.name}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Loading UnderstandingPipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vit_safetensors=VIT_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    results = []
    for key, video, question, oracle_answer in cases:
        print(f"\n=== {key} ===")
        print(f"  question : {question[:80]}{'...' if len(question) > 80 else ''}")
        t0 = time.perf_counter()
        answer = pipe.generate_video(
            video, question,
            num_sample_frames=16, target_h=224, target_w=224,
            max_new_tokens=256,
            prompt_style="lance",
        )
        dt = time.perf_counter() - t0
        print(f"  generated in {dt:.1f}s")
        print(f"  ORACLE: {oracle_answer[:200]}{'...' if len(oracle_answer) > 200 else ''}")
        print(f"  OURS:   {answer[:200]}{'...' if len(answer) > 200 else ''}")
        results.append((key, oracle_answer, answer, dt))

    # Write a side-by-side text report
    report_path = Path("/tmp/lance_x2t_video_sweep_report.txt")
    with open(report_path, "w") as f:
        f.write("x2t_video oracle sweep report\n")
        f.write("=" * 78 + "\n\n")
        for key, oracle, ours, dt in results:
            f.write(f"## {key}  ({dt:.1f}s)\n\n")
            f.write(f"ORACLE: {oracle}\n\n")
            f.write(f"OURS:   {ours}\n\n")
            f.write("-" * 78 + "\n\n")
    print(f"\n→ Full report: {report_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ {len(results)}/{len(cases)} cases generated")
    print(f"┃ Locally missing: video-understanding-{{vqa-01, vqa-02, vqa-04, caption-long-01}}.mp4")
    print(f"┃ → Total Phase 0 oracle set: 6 cases; we have 2 local. Content judgment is human.")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
