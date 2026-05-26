# mlxEngine — quantization considerations

**Date:** 2026-05-26
**Purpose:** Forward-pointer note from the lance-mlx Phase 5c-3 work to
a future mlxEngine inference pipeline decision. Not actionable in this
repo — feed into the mlxEngine planning track.

## The GGUF / k-quants question

Two unvalidated GGUF Lance repos exist on HF
([samuelchristlie/Lance-GGUF](https://huggingface.co/samuelchristlie/Lance-GGUF),
[Abiray/Lance_3B_Video-GGUF](https://huggingface.co/Abiray/Lance_3B_Video-GGUF)).
Neither demonstrates working inference; both essentially ship "we ran
the conversion, find your own runtime" model cards. User-reported
loading failures are consistent with the fact that **llama.cpp has no
native Lance architecture support**.

But — *llama.cpp's k-quants* (Q4_K_M, Q5_K_M, Q6_K with importance-matrix
calibration) are a genuinely different and often superior quantization
scheme to MLX's affine. They use per-group sub-block scales, mixed
precision per group, and have first-class imatrix calibration that
isn't equivalent to AWQ. **In principle, k-quants on Lance image gen
*might* close our documented ~80% HF detail floor where MLX-affine
(naive or AWQ) cannot.**

## Why this is not actionable in lance-mlx itself

Three independent blockers, each substantial:

1. **k-quants to MLX is a multi-week port, on `mx.fast.quantized_matmul`'s
   upstream roadmap, not ours.** Sub-block scale fan-out + the specific
   K_M / K_S variants would need to land in MLX core's quant kernel
   family. We can't ship a k-quant Lance variant without that, and
   wrapping llama.cpp from MLX-Swift apps isn't a sane path.

2. **The k-quants approach requires writing custom Metal kernels** —
   much heavier than our AWQ port (~200 LOC of pure math + scale
   fusion that hands off to mlx-lm's existing `nn.quantize`). The
   payoff would have to be substantial to justify the engineering cost.

3. **Even confirming "k-quants help Lance image gen" would require
   running llama.cpp with custom Lance architecture (which doesn't
   exist)** — Lance has flow-matching velocity prediction + Wan2.2 VAE
   + MoE-gen routing, none of which llama.cpp ships. Someone would
   have to fork llama.cpp first, then test, then port the result.
   Three multi-week steps in sequence.

## Lance specifics that bear on this for mlxEngine

These are findings from our Phase 5c-2 / 5c-3 sweep that any mlxEngine
quant strategy will need to account for:

### The empirical precision wall

| Variant | t2i HF Δ vs bf16 | VQA parity vs bf16 |
|---------|------------------|--------------------|
| bf16 baseline | 0% | 6/6 |
| Naive 8-bit full | -78.5% | (broken) |
| Naive 8-bit UND-only | -80.2% | (broken) |
| AWQ-INT4 full | -80.4% | 4/6 |
| AWQ-INT8 full | -80.4% | identical to AWQ-INT4 |
| AWQ-INT4 UND-only | -82.5% | identical to AWQ-INT4 |

**Pattern:** every MLX-affine quant we've tested lands at the same
~-80% HF wall on t2i, regardless of bit-width (4, 8) or calibration
(naive, AWQ) or skip-policy (full, UND-only). VQA survives the wall
because it's text-only forward through UND (which AWQ helps); image
generation does not.

### The 5c-3h finding — investigated and resolved

The "AWQ-INT8 ≈ naive 8-bit at end-to-end quality" observation looked
like AWQ wasn't helping at 8-bit, but **weight-level introspection
(Phase 5c-3h, 2026-05-26) shows AWQ IS working per-Linear**, reducing
output MSE by an average of 28% at 8-bit and 20% at 4-bit. Weight
MSE goes UP by 20-87% (deliberately, by AWQ's design — it trades
uniform weight error for outlier-channel output error). Output MSE
goes DOWN where it matters.

**The catch:** middle layers (sampled at layer 18) see AWQ as
neutral-to-slightly-harmful (+7% to +25% output MSE). Early and late
layers (0, 35) see -30% to -58%. Net per-layer average is -28%
improvement, but per-layer gains **don't compound** into end-to-end
image quality.

**Why no compounding:** Lance t2i runs 36 layers × 30 Euler steps ×
2 CFG arms = 2,160 forward-pass evaluations of every Linear per image
generation. Errors at each step feed into the next step's input via
the Euler integrator. Per-step quant improvements average out over
this long path, and the middle-layer AWQ regression actively cancels
peripheral-layer gains.

**The 80% HF floor is NOT a quant-scheme inadequacy.** It's a
compounding effect of forward-pass error through Lance's specific
architecture + the flow-matching integrator. K-quants would face the
same compounding problem. NVFP4 would face it. Custom Metal kernels
would face it. **No quant scheme tested or hypothesized would close
this floor for Lance image generation without changing the
architecture itself.**

Full writeup: `phase5c3_awq_port/PHASE_5C3H_FINDINGS.md`.

### Lance's non-LLM components

A k-quants approach (or any framework-level quant strategy) only
addresses the LLM bulk (the 1021 tensors of Lance_3B). It does **not**
quantize:

- **Wan2.2 3D causal VAE** (1.34 GB, 48-channel latent, used for
  every t2i / t2v decode step). Highly sensitive to precision per
  Phase 5b empirical evidence.
- **Qwen2.5-VL ViT** (1.28 GB, used for x2t image understanding and
  for image_edit conditioning).
- **`time_embedder`, `llm2vae`** (small Linears in the flow head).
  Numerically sensitive at the velocity-prediction boundary; must
  stay bf16 per our empirical skip list.

Total non-LLM bf16 footprint: ~2.6 GB. Any quant strategy hits a
floor at this size regardless of LLM compression ratio. Our shipped
AWQ-INT4 variant is 5.65 GB total (3.31 GB LLM + 2.6 GB others). A
hypothetical k-quants Q4_K_S Lance might shave the LLM further but
won't significantly shift the total repo size.

### The MoE-gen routing constraint

Lance has dual-tower per-block weights (`.q_proj` + `.q_proj_moe_gen`,
14 such pairs per layer × 36 layers = 504 quant-target Linears in our
QUANT_SUFFIXES list). Routing per token: `mx.where(gen_mask, gen_path(x),
und_path(x))` — both Linears compute on the full sequence, then
per-token selection.

This is a non-standard pattern. Any custom inference engine that
wants to be smart about quant scheme per-tower (e.g., aggressive
quant on UND, conservative on GEN) must understand the routing,
which is bespoke. Our `src/lance_mlx/quant/` infrastructure already
encodes this: `FUSION_GROUPS` enumerates the four groups per layer
(input_layernorm, input_layernorm_moe_gen, post_attention_layernorm,
post_attention_layernorm_moe_gen), and `NO_FUSE_LINEARS` lists the
four output projections that get plain quant.

For mlxEngine: this fan-out is the unit of any per-tower quant
decision. Don't expect Reza2kn's PyTorch AWQ pipeline to ship the
qk-norms intact (theirs drops them in an UND-only extraction step);
ours does.

## Decision rule for mlxEngine (revised post-5c-3h)

**Phase 5c-3h closed the k-quants question by showing it doesn't matter.**
The 80% HF floor isn't caused by any quant scheme's inadequacy at
the per-Linear level. AWQ already reduces per-layer output MSE by
28% on average at 8-bit. K-quants would do similar work. None of
them close the end-to-end gap because **the gap isn't where the
quant work is happening.**

Three paths, in order of decreasing expected value:

**Path A (default) — ship what we have, don't fight the floor.**
- bf16 ships t2i / image-edit / x2t_image / t2v at production quality
- AWQ-INT4 ships compressed VQA (3.31 GB LLM, 6-9× faster decode)
- Don't invest in new quant schemes; they won't close the t2i gap

**Path B — hybrid precision investigation.** If mlxEngine wants
compressed t2i specifically, the candidate strategy is layer-position
hybrid: bf16 at semantic-processing middle layers (12-24 maybe),
AWQ-INT4 at peripheral layers. Phase 5c-3h showed AWQ helps at
layers 0 and 35 but hurts at layer 18. Empirical layer-by-layer
analysis would map exactly where to cut bf16 boundaries. Memory
savings would be partial (~50-60% of bf16 instead of 27%) but
might preserve t2i quality.

**Path C — step-conditional AWQ calibration.** Re-calibrate AWQ
scales separately for high-noise vs low-noise denoising timesteps.
Lance's activation distribution may shift across the 30 Euler steps,
and a single set of scales averaging across all timesteps may
under-serve the high-noise early steps where most semantic
information is laid down. Multi-week effort; speculative value.

**k-quants port is off the table** — Phase 5c-3h shows it wouldn't
help even if successfully ported. The compounding bottleneck is
architectural, not algorithmic.

## See also

- `notes/phase5n_diagnostics/phase5c3_awq_port/PHASE_5C3_COMPLETE.md`
- `notes/phase5n_diagnostics/phase5c3_awq_port/PHASE_5C3G_FINDINGS.md`
- `notes/phase5n_diagnostics/phase5c2_validation/FINDINGS.md`
- `src/lance_mlx/quant/{awq,calibrate}.py` (production code)
- HF shipped artifact: <https://huggingface.co/mlx-community/Lance-3B-AWQ-INT4>
