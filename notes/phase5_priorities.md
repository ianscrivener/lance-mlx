# Phase 5 — performance & deployment priorities

Phase 5 per `HANDOFF.md` is "Quantization + packaging + publish". Based on
performance characteristics observed through Phase 4, expanding the scope
to also cover **inference performance** since current speeds on M5 Max
range from interactive (x2t, t2i) to long-running-batch (t2v at production
size). Each priority carries a **Cost / Benefit** line per project convention.

## Measured baselines (M5 Max 128 GB, bf16, KEEP_F32 norms)

| Task | Resolution / Frames | Latent tokens | Time | Tier |
|---|---|---:|---:|---|
| x2t_image (case 03) | n/a | ~600 (prompt) | 1.3s | interactive |
| x2t_image (case 05) | n/a | ~640 (prompt) | 12.5s | interactive |
| t2i | 768×768 | 2304 | ~66s (cfg=4) | interactive |
| t2v MVP | 256×256×16f | 1280 | ~33s (cfg=4) | interactive |
| t2v intermediate | 512×512×16f | 5120 | ~155s (cfg=4) | "few-minute wait" |
| **t2v production** | **768×768×50f** | **29952** | **75.9 min (measured)** | **batch only — AND OUTPUT IS BROKEN, see Phase 4c below** |

Time scaling pattern: attention SDP is `O(T²)` and dominates at large T;
MLP is `O(T)` and dominates at small T. Crossover is around T ≈ 5000 on
M5 Max. The t2v production size sits ~6× past crossover, so attention
is the bottleneck.

Quality observations (Phase 4 user-confirmed):
- **t2i at 768×768 is crystal-clear**, production quality.
- t2v at 256×256×16f: recognizable subject + motion but painterly.
- t2v at 512×512×16f: recognizable subject (gold cap visible) but painterly.
  Same symptom as pre-fix t2i.
- **t2v at 768×768×50f (Lance defaults): completely broken — output is
  pure noise texture across all 49 frames.** No coherent content at all.

**Important reversal of expected pattern:** quality DEGRADES as we scale up
toward Lance's trained resolution, when we expected the opposite (sharper
at trained size). This rules out simple resolution-sensitivity and points
to a t2v-specific bug that compounds at large T. See Phase 4c below.

## Priorities

### P1 — Gather/scatter routing (deferred from Phase 1c)

Current LanceMoT routing uses `mx.where` to compute BOTH UND and GEN expert
paths on every token, then merge. For T = 29952, this doubles MLP FLOPs vs
the upstream gather/scatter approach that only computes each expert on its
own tokens.

**Cost:** ~4-6 hours. Rewrite `LanceMoTLayer.__call__` to use gather/scatter
patterns. MLX has `mx.take` for gather; scatter is reconstructable via
`mx.concatenate`. Adds complexity around index tensor management.

**Benefit:** ~30-50% MLP speedup at production t2v sizes. Smaller benefit
at t2i sizes (where attention is fraction). Probably 20% off the ~60 min
t2v production runtime → ~45 min. Significant but doesn't change the tier.

**Trigger:** Phase 4 is closed, before quantization (which is itself a
~30% optimization compounding with this).

### P2 — bf16 norm scales (validated in Phase 3e)

Already verified safe in Phase 3e: pure-bf16 conversion (no F32 norms)
gives ~36% speedup on t2i (65s → 42s) with NO quality loss. Same speedup
would apply to t2v.

**Cost:** 5 minutes — rebuild converter without KEEP_F32 patterns,
reconvert checkpoints, update default paths.

**Benefit:** ~36% across-the-board speedup. Production t2v from ~60 min →
~38 min. Combines multiplicatively with P1 — ~30 min combined.

**Trigger:** Immediately after Phase 4 closes — this is the cheapest win
with zero downside.

### P3 — 8-bit quantization

Per HANDOFF Phase 5a: `mlx-community/Lance-3B-8bit`. MLX has `mlx.nn.quantize`
that quantizes linear weights in-place. The MoT layer has many `nn.Linear`
modules (q/k/v/o × 2 experts × 36 layers + MLP × 2 × 36) — clean quantization
target.

**Cost:** ~1-2 days. Quantize each expert separately (per-expert bit width
is a Phase 5 knob). Validate Phase 3 t2i quality holds at 8-bit — re-run
the consistency suite. Tune any KEEP_F32 patterns that quantization
should respect.

**Benefit:** ~50% memory reduction (12 GB → 6 GB for Lance_3B). Matmul
speedup depends on MLX's quantized kernels — likely 1.5-2× on M5 Max for
large matmuls. Combined with P1+P2: ~15 min production t2v.

**Trigger:** After P1 + P2, before HF publish.

### P4 — 4-bit quantization

`mlx-community/Lance-3B-4bit`. Same approach as P3 but tighter precision.

**Cost:** ~1 day (mostly validation — the conversion is similar to P3).
Re-run the consistency suite at 4-bit; relax thresholds per HANDOFF Phase
5a notes (FID < 0.10, LPIPS < 0.04 vs the < 0.05 / < 0.02 at higher bits).

**Benefit:** ~75% memory reduction (12 GB → 3 GB). Makes Lance runnable
on 16 GB Macs. Matmul speedup ~2-3×. Quality cost: noticeable on t2i;
unknown on t2v.

**Trigger:** After P3, optional based on whether the quality holds.

### P5 — KV cache for the flow loop

Currently the t2i / t2v flow loop runs the full prompt forward every step
because each step has different timestep values. The prompt embeddings
(text + initial latent values) change per step; we can't naive-cache.

But — there might be a sub-cache opportunity: for fixed text positions,
the K, V values at those positions don't depend on timestep (text
embeddings are fixed). We could cache K/V for text positions across steps
and recompute only for VAE positions.

**Cost:** ~1 day. Need to carefully manage which positions get cached vs
recomputed. Mixed cache state across LanceMoT layers.

**Benefit:** Significant speedup since text positions are ~50 of 30000
total tokens. The attention math saves recomputing K, V at those 50
positions each of 30 steps. But the attention SDP must still attend
across all 30000 tokens — so the saving is just on the projection step,
not the SDP itself. Maybe 5-10% speedup.

**Trigger:** Lower priority than P1-P4 due to small relative win.

### P6 — Faster image-token / latent_pos_embed lookup

Current Phase 4 has 3D-nested-loop construction of `lpe_indices`:
```python
[f * 4096 + r * 64 + c for f in ... for r in ... for c in ...]
```

Python loops at startup. Negligible for one-shot use but should use
vectorized MLX construction for any batch / repeated-call scenarios.

**Cost:** 30 minutes. `mx.arange` + reshape + broadcast.

**Benefit:** Removes ~few-second startup overhead. Negligible at runtime.

**Trigger:** Polish pass before HF publish.

### P7 — Resolution / step / CFG quality-vs-speed knobs

User-facing: expose well-documented knobs for trading off quality vs speed
on a per-generation basis. Current defaults match Lance's `inference_lance.sh`
production settings.

**Cost:** 1-2 hours documentation + smoke testing.

**Benefit:** Users can pick interactive (fewer steps, smaller resolution)
vs publication-quality (full Lance defaults). Currently all-or-nothing.

**Trigger:** Before HF publish.

## Suggested execution order

1. **P2 bf16 norms** (5 min, 36% speedup, zero downside) — immediate
2. **Validate t2v at 768×768×50f produces quality output** (currently running)
3. **P1 gather/scatter routing** (4-6 hrs, additional ~30% on top of P2)
4. **P3 8-bit quantization** (1-2 days, plus ~1.5-2× matmul)
5. **P4 4-bit quantization** (1 day, optional)
6. **P7 quality knobs** (1-2 hrs, polish)
7. **P5 partial KV cache** (1 day, small win) — possibly skip
8. **P6 lpe_indices vectorize** (30 min, polish)
9. **HF publish**: `mlx-community/Lance-3B-bf16` / `-8bit` / `-4bit` /
   `-Video-bf16` / `-Video-8bit` / `-Video-4bit`

After P1 + P2 + P3 the production t2v target is **~10-15 min on M5 Max
128 GB** — moves from "batch only" to "lunch break" tier.

## Phase 4c findings (2026-05-21) — MOST OF "painterly t2v" is not a bug

Investigation via the same Phase 3 self-describe + bisection pattern:

### Frame-count bisection at 768×768 with Lance_3B_Video

| Frames | t_lat | n_lat | Time | Self-describe verdict | Visual |
|---:|---:|---:|---:|---|---|
| 1 | 1 | 2304 | 69s | "abstract image... face on object" | painterly red panda + confetti |
| 5 | 2 | 4608 | 255s | "abstract scene... no clear objects" | **best result** — clear red panda with cap on wave |
| 9 | 3 | 6912 | 480s | "chaotic mix... no recognizable features" | painterly with subject |
| 13 | 4 | 9216 | 723s | "dog wearing a hat... floating" | red panda + cap visible |
| 49 | 13 | 29952 | 4552s | n/a | **pure noise** across all frames |

Pattern: 1f-13f all produce painterly-but-recognizable output. 49f breaks
completely. Sweet spot for Lance_3B_Video painterly aesthetic appears to
be ~5 frames at 768×768.

### KEY DIAGNOSIS: Lance_3B vs Lance_3B_Video are different fine-tunes

Compared every shared tensor key (1021 LLM keys, all in common). Most
differ by significant mean-abs values:

  layers.26.self_attn.q_norm_moe_gen.weight   diff = 0.849
  layers.26.self_attn.k_norm_moe_gen.weight   diff = 0.826
  layers.32.self_attn.k_norm_moe_gen.weight   diff = 0.770
  layers.10.self_attn.{q,k}_norm_moe_gen      diff ≈ 0.70
  layers.16.self_attn.q_norm_moe_gen          diff = 0.647
  layers.{15,35}.self_attn.{q,k}_norm_moe_gen diff ≈ 0.59
  ...

`lm_head.weight` and `embed_tokens.weight` are byte-identical (diff =
0.000). The two checkpoints share base weights but Lance_3B_Video has
**different `_moe_gen` (GEN expert) QK-norms** — it's been further
fine-tuned on video data, which gives it a painterly/stylized output
aesthetic even on single-frame tasks.

### Cross-check: t2i pipeline with Lance_3B_Video weights

Ran our working t2i pipeline (which produces crystal-clear output with
Lance_3B) but pointed it at Lance_3B_Video weights. Result at the same
prompt/seed: recognizable red panda with cap, but painterly confetti
background. NOT crystal-clear. **Confirms the quality difference is the
weights, not our pipeline code.**

### Implications

- **Phase 3 / t2i quality is PROVEN with Lance_3B** (photorealistic).
- **Phase 4a smaller-scale t2v outputs at 256×256, 512×512×16f, 768×768
  at 1f/5f/9f/13f are all WORKING AS DESIGNED for Lance_3B_Video** — the
  painterly aesthetic is the Video model's intrinsic style.
- **Phase 0 oracle t2i outputs (cat with STOP, fox, etc.) were almost
  certainly generated with Lance_3B**, NOT the Lance_3B_Video that
  inference_lance.sh defaults to. Otherwise upstream's t2i output would
  look like our painterly version.
- The pipeline / scaffolds / conversion / mask / position-ID code is
  validated correct.

### Remaining real bug: noise output at 768×768×50f

This is a separate, narrower bug. At t_lat=13 / n_lat=29952, output
collapses to pure noise (not even painterly). The bisection's working
sizes (t_lat 1-4, n_lat ≤ 9216) all produce coherent painterly output.
Failure mode emerges somewhere between n_lat=9216 (works painterly) and
n_lat=29952 (pure noise).

Candidate causes (now narrowed since we know weights/pipeline are fine):
- bf16 SDP softmax overflow at very large T
- The Lance bidirectional mask (T,T) being ~1.8 GB at 30k tokens
- VAE decode failure at t_lat=13 specifically
- CFG amplification interaction with large T despite our renorm

### Recommended Phase 4c continuation

1. **Run 25f or 33f at 768×768** to find the exact T_lat where things
   break (bisect 13 → 49).
2. **Try cfg=1 at 49f** to isolate CFG-amplification.
3. **Try VAE-decode-only test** at t_lat=13 with random latents to rule
   in/out the VAE.

After 4c: the smaller-scale t2v already-working at painterly quality
counts as Phase 4 success. The 50f×768 fix is a polish / production-tier
follow-up.

## Phase 4d — explore using Lance_3B for SINGLE-frame "video" tasks?

Since Lance_3B produces crystal-clear t2i and Lance_3B_Video has the
larger latent_pos_embed for t_lat > 1, a hybrid approach could use
Lance_3B for image-only paths (already done in t2i pipeline) and
Lance_3B_Video only for genuine multi-frame video.

Probably not worth pursuing — Lance_3B_Video at 5f already works at
"recognizable subject" quality which IS valid video generation. The
"painterly style" is the model's character, not a defect.

## Phase 4c — t2v "noise-at-scale" investigation (PREREQUISITE before Phase 5)

The full Lance-defaults t2v generation (768×768×50f, T=29952 latent tokens)
ran end-to-end without crashing but produced pure noise output. The
smaller-resolution outputs were degraded-but-coherent. This degradation
pattern as T grows points to one of:

### Candidate A: bf16 attention numerical instability at large T
SDP attention computes `softmax(QK / √d)` and at T=29952 with bf16
intermediates, the logits range can overflow or vanish before softmax,
especially when CFG amplifies the velocity. The conditional and
unconditional forwards run independently — both can be borderline-stable
on their own but cfg×4 amplification of the delta blows up.

**Diagnostic:** re-run at cfg=1 (no CFG, single forward per step). If
output becomes coherent, CFG-amplification-at-large-T is the cause.

**Fix (if confirmed):** cast attention QK softmax to fp32 inside the SDP
call (mlx-vlm's `scaled_dot_product_attention` may already do this; verify
by inspecting the math).

### Candidate B: latent_pos_embed at high temporal indices is noisy
The trained `latent_pos_embed` table has 31 temporal slots (Lance_3B_Video).
Production-size video uses temporal indices 0..12. If the model was
predominantly trained at smaller temporal extents (e.g., only used indices
0..3 frequently), the high-index entries could be poorly learned.

**Diagnostic:** plot the L2 norm per temporal slot in
`latent_pos_embed.pos_embed[t*4096:(t+1)*4096]` — if higher t indices
have anomalously small or large norms vs lower indices, they're under-trained.

### Candidate C: mask construction breaks at large T
Our `_build_block_mask` creates a (T, T) float mask. For T=29952+text ≈ 30000,
that's 900M entries × 2 bytes = 1.8 GB. May exceed MLX's mask-handling
capacity (memory, indexing) and silently produce wrong values.

**Diagnostic:** check mask shape/dtype/sample values at the failing scale.
Maybe reduce to a 1D causal+segment mask if MLX has support.

### Candidate D: position_ids saturation
At t_lat=13, the latent t-axis (post-MaPE) is `2000..2012`. The h/w axes
use small numbers (`text_len + 0..47` ≈ `50..100`). mRoPE handles
positions up to `max_position_embeddings=128000`. No saturation expected.
Probably NOT the cause but worth a quick print-and-verify.

### Candidate E: VAE decode at large temporal extent
Wan22VAEDecoder uses internal chunked decoding (first_chunk=True). Maybe
at t_lat=13 it crashes silently / produces noise. Test: feed a known-good
random latent at (1, 13, 48, 48, 48) and see what the VAE produces. If
noise, it's the VAE; if reasonable noise-image, it's the flow loop.

**Recommended Phase 4c sequence:**

1. **cheapest:** run cfg=1 at 768×768×50f (Candidate A — saves ~38 min
   per run, single forward per step). If output works, CFG renorm at
   large T is the fix.
2. Run 768×768×16f (small temporal × big spatial) — isolates spatial vs
   temporal scaling. If THIS works, temporal axis is the issue.
3. Run 384×384×50f (small spatial × big temporal) — converse isolation.
4. Probe latent_pos_embed norms and mask values directly.

Until Phase 4c is resolved, t2v "production quality" claim cannot be made.
Smaller-resolution t2v works at painterly quality, which is itself useful
but doesn't match Lance's `inference_lance.sh` default outputs.

## Out of scope for Phase 5

- Speculative decoding / consistency-style few-step variants (Phase 6+
  research direction per HANDOFF)
- Mixed-precision per-expert quantization (UND 4-bit, GEN 8-bit) — worth
  experimenting but not a Phase-5 must-have
- Distillation / Lance-Lightning fork — HANDOFF non-goal
