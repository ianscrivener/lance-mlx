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

### The 5c-3h mystery

**AWQ-INT8 ≈ naive 8-bit quality**. At 8-bit precision, the AWQ scale
rebalancing provides essentially zero benefit, while at 4-bit AWQ
improves over naive by 3-15 percentage points HF per prompt.

This is the *opposite* of what precision intuition predicts. The
most likely explanation: at 8-bit, the dominant error budget isn't
weight quantization — it's either kernel-side activation noise from
`mx.fast.quantized_matmul`, or Lance's specific activation
distribution has structure that per-group affine just cannot
capture regardless of scale.

If true, **k-quants might or might not help depending on which root
cause is real**. K_M sub-block scales would help if it's a
distribution-structure problem; they wouldn't help if it's a kernel
issue. The Phase 5c-3h research thread (per-layer activation
diff bf16 vs quantized) would settle which, and is the cheapest
investment before deciding on k-quants.

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

## Decision rule for mlxEngine

Two paths if quant matters more in mlxEngine than it does in
lance-mlx:

**Path A — wait for upstream MLX k-quants.** Track
`mx.fast.quantized_matmul` for Q4_K_M / IQ4 / k-quant landing.
When they land, our existing `src/lance_mlx/quant/awq.py` machinery
generalizes: replace `mx.quantize` with whichever new MLX kernel,
keep the alpha-search + scale fusion. ~1 day of work post-landing.

**Path B — invest in a custom Metal quant kernel for Lance
specifically.** Multi-week effort. Only worth it if Phase 5c-3h
research conclusively shows the 80% floor is distribution-structure
that k-quants would close, AND we have a downstream user that needs
4-bit Lance image gen at production quality. Currently neither
condition is met.

Default: **Path A.** AWQ-INT4 ships VQA today; bf16 ships t2i today;
shipping decisions don't depend on closing the t2i quant gap.

## See also

- `notes/phase5n_diagnostics/phase5c3_awq_port/PHASE_5C3_COMPLETE.md`
- `notes/phase5n_diagnostics/phase5c3_awq_port/PHASE_5C3G_FINDINGS.md`
- `notes/phase5n_diagnostics/phase5c2_validation/FINDINGS.md`
- `src/lance_mlx/quant/{awq,calibrate}.py` (production code)
- HF shipped artifact: <https://huggingface.co/mlx-community/Lance-3B-AWQ-INT4>
