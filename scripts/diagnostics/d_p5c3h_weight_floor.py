#!/usr/bin/env python3
"""Phase 5c-3h — investigate the 8-bit precision floor mystery.

Background: AWQ-INT8 produces ~80% HF detail loss on t2i, identical to
naive 8-bit. At 4-bit, AWQ improves 3-15 pp over naive. The mystery:
why doesn't AWQ's per-channel scale rebalancing help at 8-bit?

Hypothesis: at 8-bit, mx.fast.quantized_matmul's per-group scales
ALREADY capture enough of Lance's activation distribution that the
AWQ scale fusion has no remaining work to do — i.e. the per-group
min/max approximation at 8-bit precision is so close to the true
distribution that pre-scaling has no improvement to offer.

Method: weight-level introspection. For several Linears in the
fusion-group set:

  1. Get bf16 reference weight w_ref
  2. Dequantize naive 8-bit weight: w_naive
  3. Dequantize AWQ-INT8 weight: w_awq_scaled (= w_ref * s_awq * quant_noise)
  4. Recover s_awq from the norm modification:
        s_awq = w_bf16_norm / w_awq_norm   (elementwise)
  5. Compute "effective" AWQ weight: w_awq_eff = w_awq_scaled / s_awq[None, :]
  6. Compare MSE(w_naive vs w_ref) vs MSE(w_awq_eff vs w_ref)

Repeat for several layers + Linears to see the pattern. If AWQ MSE ≈
naive MSE at 8-bit, the AWQ scale fusion is being absorbed by the
per-group quant scheme — confirms the floor is structural.

For context, also run the same at 4-bit (where AWQ visibly helps).

Cost: ~30 s. Pure tensor analysis, no forward passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

MODELS = Path("/Users/dustinnielson/DEV_INT/lance-mlx-research/lance-mlx-models")
ACT_STATS = Path(__file__).resolve().parents[2] / "notes" / "phase5n_diagnostics" / "phase5c3_awq_port" / "act_stats" / "act_stats.safetensors"


# Fusion-group targets: norm → consumers (these get AWQ scale fusion)
TARGETS = [
    # (norm path, consumer path, group_size_naive, group_size_awq)
    ("layers.0.input_layernorm",                  "layers.0.self_attn.q_proj"),
    ("layers.18.input_layernorm",                 "layers.18.self_attn.q_proj"),
    ("layers.35.input_layernorm",                 "layers.35.self_attn.q_proj"),
    ("layers.0.post_attention_layernorm",         "layers.0.mlp.up_proj"),
    ("layers.18.post_attention_layernorm",        "layers.18.mlp.up_proj"),
    ("layers.35.post_attention_layernorm",        "layers.35.mlp.up_proj"),
]


def dequant_linear(saved: dict, base: str, bits: int, group_size: int) -> mx.array:
    """Dequantize an MLX QuantizedLinear weight back to fp."""
    qw = saved[f"{base}.weight"]
    sc = saved[f"{base}.scales"]
    bi = saved[f"{base}.biases"]
    return mx.dequantize(qw, sc, bi, bits=bits, group_size=group_size)


def compare_one(name: str, w_ref: mx.array, w_dq: mx.array,
                act_mean: mx.array | None = None) -> dict:
    """Compute weight-level MSE and (more importantly) output-level MSE.

    Weight-level MSE alone is misleading for AWQ — AWQ's whole design is
    to MOVE error from outlier channels to low-act channels, so
    weight MSE can go UP while output MSE goes DOWN. The output-level
    synthetic-input MSE is what AWQ's own search optimizes against and
    is what matters for forward-pass quality.
    """
    w_ref_f = w_ref.astype(mx.float32)
    w_dq_f = w_dq.astype(mx.float32)
    err = w_dq_f - w_ref_f                          # (out, in)
    weight_mse = float(mx.mean(err * err))

    stats = {
        "name": name,
        "weight_mse": weight_mse,
        "max_abs_err": float(mx.abs(err).max()),
    }
    if act_mean is not None:
        # Synthetic-input output MSE (what AWQ's loss optimizes):
        #   x ~ randn(B, in) * act_mean
        #   y_ref = x @ w_ref.T
        #   y_dq  = x @ w_dq.T
        #   output_mse = mean((y_ref - y_dq)²)
        in_features = w_ref.shape[1]
        act_f = act_mean.astype(mx.float32)
        # Seeded random for reproducibility across rows
        mx.random.seed(0xC0DE)
        x = mx.random.normal((512, in_features)) * act_f
        y_ref = x @ w_ref_f.T
        y_dq = x @ w_dq_f.T
        output_mse = float(mx.mean((y_ref - y_dq) ** 2))
        stats["output_mse"] = output_mse

        # Per-input-channel weight error squared, weighted by act_mean²
        # (analytic approximation to output_mse modulo cross-channel terms):
        per_chan_w_mse = mx.mean(err * err, axis=0)   # (in,)
        analytic_output_mse = float(mx.sum(per_chan_w_mse * act_f * act_f))
        stats["analytic_output_mse"] = analytic_output_mse
    return stats


def main() -> int:
    print(f"=== Phase 5c-3h — 8-bit precision floor introspection ===\n")
    print(f"Loading model variants ...")

    bf16 = mx.load(str(MODELS / "Lance-3B-bf16" / "model.safetensors"))
    naive_8 = mx.load(str(MODELS / "Lance-3B-8bit" / "model.safetensors"))
    awq_8 = mx.load(str(MODELS / "Lance-3B-AWQ-INT8" / "model.safetensors"))
    naive_4 = mx.load(str(MODELS / "Lance-3B-4bit-full" / "model.safetensors"))
    awq_4 = mx.load(str(MODELS / "Lance-3B-AWQ-INT4" / "model.safetensors"))
    print(f"  loaded 5 variants\n")

    print(f"Loading activation stats (for act-weighted error)")
    act_stats_raw = mx.load(str(ACT_STATS))
    # Compute act_mean (sum_abs / n_tokens) — n_tokens is in meta JSON but
    # constant across all entries (~152790 from the calibration run). Use
    # sum_abs directly since the constant scales out for comparison.
    act_means = {
        k.replace(".sum_abs", ""): act_stats_raw[k]
        for k in act_stats_raw if k.endswith(".sum_abs")
    }
    print(f"  {len(act_means)} act_mean entries\n")

    # For 8-bit: gs=64; for 4-bit AWQ: gs=128; for 4-bit naive: gs=64
    rows = []
    for norm_path, consumer_path in TARGETS:
        # ─── Recover s from the AWQ norm diff (8-bit and 4-bit versions) ───
        w_norm_bf16 = bf16[f"{norm_path}.weight"]
        w_norm_awq8 = awq_8[f"{norm_path}.weight"]
        w_norm_awq4 = awq_4[f"{norm_path}.weight"]
        s_awq8 = (w_norm_bf16.astype(mx.float32) / w_norm_awq8.astype(mx.float32))
        s_awq4 = (w_norm_bf16.astype(mx.float32) / w_norm_awq4.astype(mx.float32))

        # ─── Reference + naive at 8-bit ────────────────────────────────────
        w_ref = bf16[f"{consumer_path}.weight"]
        w_naive_8 = dequant_linear(naive_8, consumer_path, bits=8, group_size=64)
        w_awq_8_scaled = dequant_linear(awq_8, consumer_path, bits=8, group_size=64)
        # Unscale AWQ to bring back to bf16-comparable space
        w_awq_8_eff = w_awq_8_scaled / s_awq8.reshape(1, -1)

        # ─── 4-bit ──────────────────────────────────────────────────────────
        w_naive_4 = dequant_linear(naive_4, consumer_path, bits=4, group_size=64)
        w_awq_4_scaled = dequant_linear(awq_4, consumer_path, bits=4, group_size=128)
        w_awq_4_eff = w_awq_4_scaled / s_awq4.reshape(1, -1)

        act_mean = act_means.get(consumer_path)

        # ─── Compare ──────────────────────────────────────────────────────
        for label, w_dq in [
            ("naive_8bit",    w_naive_8),
            ("AWQ_INT8_eff",  w_awq_8_eff),
            ("naive_4bit",    w_naive_4),
            ("AWQ_INT4_eff",  w_awq_4_eff),
        ]:
            stats = compare_one(f"{consumer_path}/{label}", w_ref, w_dq, act_mean)
            stats["consumer_path"] = consumer_path
            stats["variant"] = label
            rows.append(stats)

    # ─── Render table (weight-MSE + output-MSE) ──────────────────────────────
    print(f"{'consumer_path':<28s}  {'variant':>14s}  {'weight_MSE':>11s}  "
          f"{'output_MSE':>11s}  {'analytic_oMSE':>14s}")
    print(f"{'─' * 28}  {'─' * 14}  {'─' * 11}  {'─' * 11}  {'─' * 14}")
    for r in rows:
        oMSE = f"{r.get('output_mse', 0):.4e}"
        aoMSE = f"{r.get('analytic_output_mse', 0):.4e}"
        print(f"{r['consumer_path']:<28s}  {r['variant']:>14s}  "
              f"{r['weight_mse']:>11.4e}  {oMSE:>11s}  {aoMSE:>14s}")
        if r['variant'] == 'AWQ_INT4_eff':
            print(f"  {'─' * 88}")

    # ─── Headline analysis on OUTPUT MSE (what matters) ──────────────────────
    print(f"\n{'=' * 90}")
    print(f"AWQ vs naive — OUTPUT-MSE delta (what AWQ actually optimizes):")
    print(f"{'=' * 90}")
    print(f"{'consumer_path':<28s}  {'8bit AWQ vs naive':>20s}  {'4bit AWQ vs naive':>20s}")
    for cp in sorted(set(r['consumer_path'] for r in rows)):
        n8 = next(r['output_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'naive_8bit')
        a8 = next(r['output_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'AWQ_INT8_eff')
        n4 = next(r['output_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'naive_4bit')
        a4 = next(r['output_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'AWQ_INT4_eff')
        d8 = (a8 - n8) / n8 * 100
        d4 = (a4 - n4) / n4 * 100
        print(f"{cp:<28s}  {d8:>+19.1f}%  {d4:>+19.1f}%")

    print(f"\n{'=' * 90}")
    print(f"AWQ vs naive — WEIGHT-MSE delta (deliberately worse for AWQ):")
    print(f"{'=' * 90}")
    print(f"{'consumer_path':<28s}  {'8bit AWQ vs naive':>20s}  {'4bit AWQ vs naive':>20s}")
    for cp in sorted(set(r['consumer_path'] for r in rows)):
        n8 = next(r['weight_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'naive_8bit')
        a8 = next(r['weight_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'AWQ_INT8_eff')
        n4 = next(r['weight_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'naive_4bit')
        a4 = next(r['weight_mse'] for r in rows if r['consumer_path'] == cp and r['variant'] == 'AWQ_INT4_eff')
        d8 = (a8 - n8) / n8 * 100
        d4 = (a4 - n4) / n4 * 100
        print(f"{cp:<28s}  {d8:>+19.1f}%  {d4:>+19.1f}%")

    print(f"\n{'=' * 90}")
    print(f"Interpretation")
    print(f"{'=' * 90}")
    print(f"  AWQ DELIBERATELY trades weight-MSE for OUTPUT-MSE — it moves quant")
    print(f"  error from outlier (high-act) channels to low-act channels because")
    print(f"  outliers dominate output. So weight-MSE going UP with AWQ is EXPECTED")
    print(f"  and not a bug.")
    print(f"")
    print(f"  The right question for 5c-3h: does AWQ reduce OUTPUT-MSE at 8-bit?")
    print(f"  If YES at 4-bit and ~zero at 8-bit, the AWQ improvement saturates")
    print(f"  out of useful range at 8-bit precision — supports the structural-")
    print(f"  precision-floor hypothesis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
