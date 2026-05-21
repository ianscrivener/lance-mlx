# Phase 3 — t2i milestone + quality investigation

## ✓ Pipeline works end-to-end

The GEN expert path runs cleanly. 30 Euler steps + Wan2.2 VAE decode produce a 768×768 image in:
- ~32s without CFG (single conditional forward per step)
- ~66s with CFG (two forwards per step: cond + uncond)

Latent stats look correct in flow loop:
- Start: mean=0.001, std=0.99 (Gaussian noise per `mx.random.normal`)
- End:   mean=0.017, std=0.603 (some denoising happens)

VAE decode succeeds. PIL.Image produced. Saved to disk.

## ⚠ Output is structurally an image but not prompt-aligned

Tested prompt: `"A cat holds a poster with rainbow text 'STOP'"` at seed=42, 30 steps.

| CFG | Output content |
|---|---|
| 1.0 (no CFG) | Nighttime urban/crowd scene |
| 4.0 (Lance default) | Same nighttime urban scene |
| 15.0 (extreme) | Same nighttime urban scene, slightly more contrast |

The PyTorch oracle at the same prompt+seed+CFG=4.0 produces a clear cat + colorful poster. Our output is consistently off-prompt across all CFG values.

## Diagnosis: NOT a CFG plumbing bug

CFG=15 produced almost identical output to CFG=1, meaning either:
1. Conditional and unconditional velocities are essentially identical → text conditioning isn't being absorbed by the model.
2. The model has converged to a fixed point that's prompt-independent (the "mean image" of its training distribution).

Both diagnoses point to the same root: **text → velocity conditioning is too weak in our forward pass.**

## Candidate root causes (each carries a **Benefit:** line per project convention)

### Numerical-precision drift (Phase 2.1b carried-over)

The bf16-weights + F32-norm-scales setup promotes intermediate activations to fp32. PyTorch oracle stays in bf16 throughout. Small per-step accumulation differences could compound over 30 steps and end-stage the latents at a "generic" location in latent space.

**Cost:** rebuild the converter without KEEP_F32 (~10 LOC change), reconvert Lance_3B (~10 min for a cached download).
**Benefit:** rules in/out the most-suspected source of numerical drift. Cleanest single experiment.

### Unconditional-prompt format

Our v1 uses an empty user message (`<|im_start|>user\n<|im_end|>`). Lance's `cfg_uncond_token_id` parameter (visible in `inference_lance.py`) suggests a specific token-ID-based uncond, not just an empty string.

**Cost:** ~30 min — find `cfg_uncond_token_id` in upstream config, swap our uncond builder.
**Benefit:** if the model was trained with a specific null embedding, our empty-string approach gives WEAK uncond → CFG amplification has nothing to bite. Fixing this could unlock proper CFG response.

### Position IDs structure

Our 3D position layout for latent tokens uses `(t=0, h=row, w=col)` starting from `text_len` then re-anchored to 1000. This matches mlx-vlm's `get_rope_index` convention for image tokens. Lance may use a different convention for noisy latent positions specifically.

**Cost:** ~1 hr — read more of upstream's `process_text_template` to see exact position-ID construction for NOISY_VAE tokens.
**Benefit:** if positions are wrong, the model can't attend correctly across modalities. This could be the biggest single fix.

### Time-embedding scale

Our scaffold passes `t ∈ [0, 1]` to TimestepEmbedder. The HANDOFF/scaffold docstring mentions "Lance" vs "[0, 1000]" conventions. If Lance trained with t scaled, we'd be feeding wrong magnitudes.

**Cost:** ~15 min — try `t * 1000` and re-run.
**Benefit:** cheap experiment. Wrong time-embed magnitudes would make the model unable to know "where it is" in the flow trajectory.

### Latent normalization

We init in N(0, 1) per `mx.random.normal` (which IS the normalized-latent space). At decode, we call `denormalize_latents` then `Wan22VAEDecoder`. The model expects latents in normalized space.

**Cost:** ~30 min — check whether VAE decoder is supposed to receive normalized OR denormalized; check whether init noise should be in some other space.
**Benefit:** if normalization is wrong, the velocity head sees out-of-distribution inputs and predicts unreliably.

## Recommended Phase 3d sequence

1. **Time-embedding scale ablation** — fastest test. Try `t * 1000`. If output dramatically changes, that's our smoking gun.
2. **Position-IDs deep dive** — if (1) doesn't unlock it, look at upstream `process_text_template` carefully to see exact position-ID layout for NOISY_VAE tokens.
3. **`cfg_uncond_token_id` fix** — once CFG can meaningfully bite, the prompt should pull through.
4. **bf16-norm ablation** — last resort; expensive (reconvert) but definitively rules in/out precision drift.

## What landed this session (Phase 3a + 3b + 3c)

- ✓ Wan2.2 VAE: convert .pth → MLX safetensors, load into `Wan22VAEDecoder` (0 missing keys, smoke decode works)
- ✓ `TextToImagePipeline.from_pretrained()` composes LanceModel + VAE + tokenizer
- ✓ `pipeline.generate(prompt, ...)` runs 30 Euler steps with proper position IDs, MaPE re-anchoring, latent_pos_embed, time_embedder, vae_in_proj, llm2vae
- ✓ Classifier-free guidance: two-forward CFG blend (cond + uncond, weighted)
- ✗ Output quality: structurally an image, not prompt-aligned. Phase 3d work.
