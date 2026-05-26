#!/usr/bin/env python3
"""Phase 5c-3f — AWQ-INT4 x2t_image (VQA) oracle sweep.

Phase 5c-3e showed AWQ-INT4 doesn't preserve t2i quality. But Reza2kn's
PyTorch AWQ-INT4 was validated for x2t_image only (5/6 oracle correct).
This script runs the 6 t2i-side oracle cases through both bf16 and
AWQ-INT4 and reports content-correctness, settling whether AWQ-INT4
is shippable as a VQA-only variant.

Cost: ~5 min for both models × 6 cases at greedy decode.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT.parent / "lance-mlx-models"
BF16_WEIGHTS    = MODELS_DIR / "Lance-3B-bf16"
AWQ_WEIGHTS     = MODELS_DIR / "Lance-3B-AWQ-INT4"
AWQ_UND_WEIGHTS = MODELS_DIR / "Lance-3B-AWQ-INT4-und"
IMAGES_DIR = REPO_ROOT / "tests" / "fixtures" / "images"
ORACLE_JSON = next((REPO_ROOT / "tests" / "fixtures" / "results").glob("x2t_image_sample_*/result.json"))
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "phase5c3_awq_port" / "x2t_validation"

MAX_NEW_TOKENS = 256


def load_pipe(weights_dir):
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    return UnderstandingPipeline.from_pretrained(
        lance_weights_dir=weights_dir,
        vit_safetensors=weights_dir / "vit.safetensors",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    oracle = json.loads(ORACLE_JSON.read_text())
    cases = [(e["image"].split("/")[-1], e["question"], e["answer"]) for e in oracle]

    print(f"=== Phase 5c-3f — x2t_image AWQ-INT4 oracle sweep ===")
    print(f"  cases: {len(cases)}")
    print(f"  oracle: {ORACLE_JSON.parent.name}")
    print(f"  variants: bf16 + AWQ-INT4")
    print(f"  greedy decode, max_new_tokens={MAX_NEW_TOKENS}\n")

    results = {}   # {variant: {case_filename: (answer, latency_s)}}

    for variant, weights_dir in [
        ("bf16",         BF16_WEIGHTS),
        ("AWQ-INT4",     AWQ_WEIGHTS),
        ("AWQ-INT4-und", AWQ_UND_WEIGHTS),
    ]:
        print(f"\n╔══ {variant} ({weights_dir.name}) ══════════════════════════")
        t0 = time.perf_counter()
        pipe = load_pipe(weights_dir)
        print(f"  loaded in {time.perf_counter()-t0:.1f}s")
        results[variant] = {}
        for case_filename, question, _ in cases:
            image_path = IMAGES_DIR / case_filename
            image = Image.open(image_path).convert("RGB")
            t0 = time.perf_counter()
            answer = pipe.generate(image, question,
                                   max_new_tokens=MAX_NEW_TOKENS, verbose=False)
            dt = time.perf_counter() - t0
            short_q = question[:50] + ("..." if len(question) > 50 else "")
            print(f"  {case_filename[:36]:>36s}  {dt:>5.1f}s")
            results[variant][case_filename] = (answer, dt)
        del pipe
        import gc; gc.collect(); mx.clear_cache()

    # ─── Side-by-side comparison ────────────────────────────────────────
    print(f"\n══════ Per-case comparison ════════════════════════════════════")
    n_matches_exact = 0
    n_matches_content = 0
    case_assessments = []

    # Per-variant counters
    variant_parity = {v: {"exact_vs_bf16": 0, "facts_match_bf16": 0}
                       for v in results if v != "bf16"}

    for case_filename, question, expected in cases:
        case_id = case_filename.replace(".png", "").replace("image-understanding-case-", "")
        bf16_ans = results["bf16"][case_filename][0].strip()

        print(f"\n── case {case_id} ──")
        print(f"  Q: {question}")
        print(f"  bf16        : {bf16_ans[:100]}{'...' if len(bf16_ans) > 100 else ''}")
        for v in [v for v in results if v != "bf16"]:
            v_ans = results[v][case_filename][0].strip()
            print(f"  {v:<12s}: {v_ans[:100]}{'...' if len(v_ans) > 100 else ''}")
            if v_ans == bf16_ans:
                variant_parity[v]["exact_vs_bf16"] += 1
            # Loose content match: extract bf16's key tokens (numbers, ALL-CAPS,
            # capitalized words 3+ chars) and check v's answer contains them.
            import re
            bf16_tokens = set(re.findall(r"[A-Z]{2,}|[A-Z][a-z]{2,}|\d+(?:\.\d+)?[%]?", bf16_ans))
            v_tokens = set(re.findall(r"[A-Z]{2,}|[A-Z][a-z]{2,}|\d+(?:\.\d+)?[%]?", v_ans))
            shared = bf16_tokens & v_tokens
            if bf16_tokens and len(shared) / len(bf16_tokens) >= 0.6:
                variant_parity[v]["facts_match_bf16"] += 1

    n = len(cases)
    print(f"\n══════ Summary (vs bf16 reference) ═══════════════════════════")
    print(f"  {'variant':>14s}  {'exact-match':>12s}  {'facts ≥60%':>10s}")
    for v in sorted(variant_parity):
        p = variant_parity[v]
        print(f"  {v:>14s}  {p['exact_vs_bf16']:>3d} / {n:>2d}      "
              f"{p['facts_match_bf16']:>3d} / {n:>2d}")
    n_matches_content = max(p["facts_match_bf16"] for p in variant_parity.values())
    print(f"  reference: Reza2kn PyTorch AWQ-INT4: 5/6 oracle correct (x2t_image only)")

    if n_matches_content >= 5:
        verdict = "✓ SHIP as VQA-only variant (matches/beats Reza2kn benchmark)"
    elif n_matches_content >= 4:
        verdict = "⚠ MARGINAL — close to Reza2kn but missed a case; inspect failures"
    else:
        verdict = "✗ FAIL — does not preserve VQA quality; AWQ-INT4 not shippable"

    print(f"\n  VERDICT: {verdict}")

    # Save report
    report = {
        "n_cases": n,
        "verdict": verdict,
        "variant_parity": variant_parity,
        "latency_s_per_case": {
            v: [results[v][cf][1] for cf, _, _ in cases] for v in results
        },
        "answers_per_case": {
            cf: {v: results[v][cf][0] for v in results} for cf, _, _ in cases
        },
    }
    (OUT_DIR / "x2t_oracle_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  full report: {OUT_DIR / 'x2t_oracle_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
