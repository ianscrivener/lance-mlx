# Phase 5c-3g — AWQ on UND-only (GEN bf16): refuted

**Date:** 2026-05-25
**Hypothesis:** Phase 5c-2 naive UND-only failed because uncalibrated
8-bit QKV projections corrupted shared attention. With AWQ rebalancing,
UND-only-INT4 might preserve t2i quality where naive UND-only didn't.
**Result:** REFUTED on every axis. AWQ-INT4-und has zero advantage
over AWQ-INT4 (full quant) while being 2.2× larger and slower.

## t2i quality (4-prompt sweep, 384²)

```
prompt        AWQ-INT4 Δ vs bf16   AWQ-INT4-und Δ vs bf16   delta
P1 cat_stop   -79.3%               -76.7%                   +2.6 pp
P2 dragon     -85.3%               -86.9%                   -1.6 pp
P3 cat_skate  -78.2%               -76.9%                   +1.3 pp
P4 cat_dog    -78.9%               -89.5%                  -10.6 pp
─────────────────────────────────────────────────────────────────
average       -80.4%               -82.5%                   -2.1 pp
```

AWQ-INT4-und is **marginally worse on average**. Two prompts slightly
better, one similar, one notably worse. The -80% HF floor is robust;
keeping GEN tower at bf16 doesn't help t2i quality.

Visual confirmation matches the numbers — both AWQ variants produce
similar outputs: signs partially rendered, subjects mostly lost.

## x2t_image quality (6-case oracle sweep)

```
case  bf16            AWQ-INT4     AWQ-INT4-und
 1    "Yes"           "Yes"        "Yes"               ← IDENTICAL
 2    "43"            "43"         "43"                ← IDENTICAL
 3    "Bx62bfy"       "Byfky"      "Byfky"             ← IDENTICAL (same garbling)
 4    "1.8 million"   "198%"       "198%"              ← IDENTICAL (same wrong answer)
 5    Colosseum desc  Colosseum    Colosseum           ← IDENTICAL
 6    Eclipse desc    Eclipse      Eclipse             ← IDENTICAL
```

**AWQ-INT4 and AWQ-INT4-und produce literally identical VQA answers
across all 6 cases.** Why: VQA is text-only forward through the UND
tower; the GEN tower is never exercised. Both variants share the SAME
UND quantization, so they produce SAME UND outputs → SAME VQA answers.

The GEN tower bf16 precision is irrelevant to VQA.

## Cost / size comparison

```
                  AWQ-INT4 (full)   AWQ-INT4-und
size              3.31 GB           7.38 GB        (2.2× larger)
avg bits/wt       4.28              9.55
loaded RAM        ~3.5 GB           ~7.5 GB        (2.1× more RAM)
case 5 latency    1.4 s             5.4 s          (3.9× slower)
case 6 latency    1.2 s             4.6 s          (3.8× slower)
```

The bf16 GEN tower costs memory traffic per inference step even when
GEN isn't computationally exercised. Long-form VQA decodes pay this
cost most visibly.

## Why this is consistent with prior findings

Phase 5c-2 found UND quant corrupts image gen via shared attention:
"text tokens cross-attend with latent tokens → corrupted text-side
projections poison the SDP". The hope was AWQ calibration would
reduce that corruption enough to recover image quality.

Empirically: AWQ does reduce the UND-quant corruption (per Phase 5c-3e
where AWQ-INT4 beats naive 8-bit by 3-15 pp per prompt), but only
marginally. The remaining ~80% HF loss is structural — and it's
present even when only UND is quantized.

This means the shared-attention corruption hypothesis isn't the
dominant failure mode. There's some other systematic source of error
at the 4-bit precision floor that AWQ can dent but not close. The
Phase 5c-3h research thread (why does AWQ-INT8 ≈ naive-INT8?) likely
points at the same root cause.

## Shipping decision unchanged

**Ship `Lance-3B-AWQ-INT4` (full quant) as the VQA variant.**
- 3.31 GB on disk (27% of bf16)
- 6-9× faster on long decodes
- ~4/6 VQA parity with bf16 (matches AWQ-INT4-und exactly)
- Caveats: precision-required outputs degrade; not for t2i

**Don't ship `Lance-3B-AWQ-INT4-und`.** Strictly worse value: same VQA
quality, 2.2× larger, 3-4× slower long decodes, no t2i benefit.

## What 5c-3g closes

- Hypothesis "AWQ + UND-only quant fixes the Phase 5c-2 shared-attention
  corruption": REFUTED.
- Question "does AWQ-INT4-und make a better VQA variant than AWQ-INT4
  full quant": NO (identical VQA, worse on every other axis).
- Three of three quant-strategy variants now characterized:
  - Naive UND-only (5c-2): broken
  - AWQ full both towers (5c-3d/e/f): shippable for VQA only
  - AWQ UND-only (5c-3g): strictly worse than AWQ full quant

Quant work for Lance has reached a clean stopping point until either
(a) someone investigates the 8-bit floor (5c-3h research thread), or
(b) MLX's affine quantization scheme gets a precision upgrade upstream.

## Artifacts

- Reused `scripts/quant/apply_awq_quantize.py` with new `--skip-gen-tower` flag
- `notes/phase5n_diagnostics/phase5c3_awq_port/validation_g/`
  - `_run_t2i.log` (t2i 4-prompt sweep)
  - `_run_x2t.log` (x2t_image 6-case sweep)
- `lance-mlx-models/Lance-3B-AWQ-INT4-und/` (7.38 GB, not for shipping)
