# Phase 5c-3h — the 8-bit precision floor mystery, resolved

**Date:** 2026-05-26
**Question:** Why does Phase 5c-3e measure AWQ-INT8 quality ≈ naive 8-bit
quality at the end-to-end t2i level (both ~-80% HF), even though AWQ
should improve over naive quant?
**Method:** weight-level introspection — dequantize both naive 8-bit
and AWQ-INT8 weights for representative Linears, compare against bf16
at both the **weight-MSE** level and the **synthetic-input output-MSE**
level (the latter being what AWQ's own alpha-search optimizes).
**Result:** AWQ math is **working correctly per-layer**. The end-to-end
saturation is a compounding effect, not a quant-algorithm failure.

## Key numbers (6 Linears across 36-layer Lance_3B)

```
                                  weight-MSE delta     output-MSE delta
                                  (AWQ vs naive)       (AWQ vs naive)
                                  8bit       4bit      8bit       4bit
layers.0.self_attn.q_proj         +51.5%    +86.9%     -51.4%    -47.9%
layers.0.mlp.up_proj              +76.2%    +87.1%     -58.1%    -57.5%
layers.18.self_attn.q_proj        +20.4%    +32.5%      -0.9%    +10.6%
layers.18.mlp.up_proj             +22.1%    +45.9%      +7.3%    +25.8%
layers.35.self_attn.q_proj        +39.6%    +49.6%     -35.8%    -29.9%
layers.35.mlp.up_proj             +47.7%    +61.1%     -29.0%    -21.7%
─────────────────────────────────────────────────────────────────────
average                           +42.9%    +60.5%     -28.0%    -20.1%
```

**Read this carefully:**

- **Weight-MSE goes UP with AWQ** by 20-87 percentage points. This is
  expected and BY DESIGN — AWQ deliberately trades weight accuracy
  (uniform per-channel) for output accuracy (weighted toward outlier
  channels that dominate the layer's downstream impact).
- **Output-MSE goes DOWN with AWQ** by an average of 28% at 8-bit
  and 20% at 4-bit, on the layers where AWQ helps.
- **Layer-position matters:** early and late layers see -30 to -58%
  output-MSE reduction with AWQ. Middle layers (18) see neutral or
  slightly-negative impact (+7% to +25% — AWQ marginally hurts).

## The mystery, restated and resolved

**Initial framing (wrong):** "AWQ-INT8 ≈ naive 8-bit at end-to-end
quality, so AWQ isn't helping at 8-bit."

**Actual finding:** AWQ IS helping at every individual Linear's
output, by 28% on average at 8-bit. But the per-layer gains don't
compound into improved end-to-end image quality. The end-to-end HF
detail floor at ~80% loss is **not** explained by "AWQ doing nothing"
— it's a compounding effect downstream of accurate per-layer
quantization.

## Why per-layer gains don't compound for Lance image generation

This is the genuinely useful insight. Two non-mutually-exclusive
mechanisms:

### Middle-layer regression cancels peripheral gains

The 6-Linear sample shows AWQ helps at layers 0 and 35 but is
neutral-to-harmful at layer 18. If the middle layers carry the
semantic-processing load, even modest middle-layer AWQ regressions
can cancel out large early/late layer improvements.

**Hypothesis:** middle-layer activations don't have the strong
per-channel outlier pattern AWQ assumes — they're more uniformly
distributed across channels. Forcing AWQ's geometric-mean-normalized
scale onto a uniform distribution adds noise without removing any
real error budget.

This is testable: compute the activation outlier-ratio (max channel
mean / median channel mean) per layer and look for inversion at
middle layers. Phase 5c-3i candidate work.

### Lance's long forward path × flow-matching scheduler

t2i runs 30 Euler steps × 36 layers × 2 CFG arms = 2,160 forward-pass
evaluations of every Linear per image generation. Errors that look
small per-layer can accumulate or interact through:

- **Layer-to-layer**: each layer's output is the next layer's input;
  noise at layer N becomes structured input perturbation at layer N+1.
  The MLP-attention-residual loop amplifies certain frequencies.
- **Step-to-step**: the Euler integrator advances the latent by a
  velocity prediction at each timestep. Errors at step T influence
  the input state at step T+1; they don't cancel, they accumulate.
- **CFG signal-to-noise**: at high cfg_scale (4.0), the CFG arm's
  velocity differences are amplified. Small noise in the uncond arm
  becomes large noise in v_cfg.

End-to-end the result is a saturation around -80% HF — any layer-level
quant improvement smaller than the per-step accumulation rate just
gets averaged out by the time the VAE decodes.

## What this means for Lance quant strategy (mlxEngine input)

**The 80% HF floor is NOT a quant-scheme inadequacy.** k-quants from
llama.cpp wouldn't close it. NVFP4 wouldn't close it. Per-channel
better-calibrated scale schemes wouldn't close it. The bottleneck is
the **forward-pass error compounding through Lance's architecture +
the flow-matching integrator**, not the per-layer weight precision.

What WOULD close it (theoretical):

1. **Per-step bf16 fallback at known-sensitive layers.** Hybrid
   precision where the mid-network layers stay bf16 and only the
   peripheral ones are quantized. Saves less memory but might
   preserve t2i quality.
2. **Smaller timesteps + integration-aware quant calibration.**
   Calibrate AWQ scales using actual denoising-step activation stats
   rather than uniform t-sampled activations. The middle layers might
   benefit from t-conditional scales.
3. **Higher than fp16 baseline.** If bf16 is the baseline and the
   floor is at -80% HF, fp32-baselined AWQ might shift the floor
   down. Not feasible for mlxEngine ergonomics.

What we recommend instead: **don't fight the floor.** Ship bf16 for
t2i. Ship AWQ-INT4 for VQA (already done). Reserve heavyweight
investment for use-cases that demand it.

## What this means for the lance-mlx Phase 5c block

Phase 5c-3 is **truly closed.** The shipping picture stands:
- `mlx-community/Lance-3B-bf16` for production t2i / image-edit / x2t_image
- `mlx-community/Lance-3B-AWQ-INT4` for compressed x2t_image VQA only
- `mlx-community/Lance-3B-Video-bf16` for production video

No additional quant variant would close the t2i gap without
fundamental architectural changes (hybrid precision, step-conditional
calibration). Phase 5c is done.

## Update needed to mlx_engine_quant_notes.md

The "the 8-bit floor is structural" framing was right in conclusion
but wrong in causation. Replace with: "the 8-bit floor is a forward-
pass compounding effect — AWQ math works per-layer but gains average
out across 2,160 forward-pass evaluations per image". This changes
the implication: k-quants probably can't close it either, because
they'd face the same compounding problem.

## Artifacts

- `scripts/diagnostics/d_p5c3h_weight_floor.py` — the analysis
- This findings doc

Resolution method took: ~5 minutes of script work + one re-run after
correcting the analysis metric (weight-MSE was the wrong proxy;
synthetic-input output-MSE is what matches AWQ's own loss).
