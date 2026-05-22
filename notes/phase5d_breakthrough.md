# Phase 5d — t2v port-bug FIX FOUND (2026-05-21)

## Headline

**Removing our MaPE temporal-anchor shift (`mape_anchor=None`) fixes the t2v port bug at every practical generation scale.** Lance MLX now produces photorealistic, prompt-aligned video output — matching the Phase 0 PyTorch oracle aesthetic — at 256² through 768²×13f.

t2v's default `mape_anchor` is now `None`. The shift was a port-side deviation: upstream's `data/common.py::shift_position_ids` is gated by
`attn_mode in ["full_noise", "full"]` which never fires for pure t2v
(which has only `"noise"` segments). We were adding a 2000-anchor shift
to the latent t-axis that upstream doesn't apply.

## Scale-coherence map (no-shift)

| Scale | n_lat | spatial_std | Quality |
|---|---|---|---|
| 256² × 17f | 1,280 | 56.1 | ✅ recognizable red panda + hat |
| 512² × 17f | 5,120 | 64.1 | ✅ **photoreal panda, cinematic** |
| 480×704 × 17f | 6,600 | 61.7 | ✅ **best — photoreal + cap + satchel + board** |
| 768² × 13f | 9,216 | 54.9 | ✅ recognizable panda + satchel + ocean |
| 768² × 17f | 11,520 | 39.5 | ⚠️ partial degradation (subject fragmenting) |
| 768² × 50f | 29,952 | — | ❌ gradient collapse (oracle scale; separate bug) |

The transition is clean and monotone: above ~n_lat=9,216 coherence starts
to slip; by n_lat ≈ 30k it collapses entirely. For all production-scale
video use (256–768² × ≤13 frames; up to ~5s clips at 12 fps), the no-shift
fix is a strict improvement.

## Comparison: pre-fix vs post-fix at 480×704×17f

Same prompt ("A medium-close shot shows a red panda wearing a gold-trimmed
cap and travel satchel on a bright seaside wave..."), same seed (42), same
sampler config.

- **Pre-fix (mape_anchor=2000)**: painterly impressionistic scene, subject barely visible
- **Post-fix (mape_anchor=None)**: photorealistic 3D-cinematic — clearly visible red panda with gold-trimmed cap, travel satchel, painted surfboard, foam spray, glowing summer sky

Fixture: `tests/fixtures/lance_vs_ltx_pre_fix/p00_oracle_panda_480x704_noshift_BREAKTHROUGH.png`

## What's still open

**768² × 50f (oracle reference scale) collapses with no-shift.** This is
a *different* second bug that our MaPE shift was partially papering over
at large n_lat. Candidates:
- bf16 RoPE precision loss at high position values (Candidate 1 in
  [phase5d_candidates.md](./phase5d_candidates.md))
- Numerical accumulation in attention softmax at very long sequences
- Memory-pressure-driven Metal compilation/eval differences (29k tokens
  is near practical limits)

For production-quality video output today, **stay at or below n_lat ≈ 9k**
(e.g. 480×704×17f, 768²×13f, 256–512²×17–25f). All cover the meaningful
prompt envelope.

## Cleanup of earlier framing

Three earlier conclusions to formally retract:

1. **Phase 4c**: "Lance_3B_Video's painterly aesthetic is by design."
   ❌ Refuted by Phase 5d. The painterly look was the port bug.

2. **Phase 4e (Candidate 0 in issue #1)**: Closed issue #1 with
   "prompt-content + painterly-aesthetic misinterpretation." The
   misinterpretation was real but only ABOUT THE NOISE FRAMING; the
   underlying port bug was genuine. Issue #2 was the right framing.

3. **Phase 4e per-tensor diff conclusion**: We claimed the `_moe_gen`
   QK-norm differences between Lance_3B and Lance_3B_Video CAUSED a
   painterly aesthetic. They don't. Those differences are real but
   subtle; both checkpoints can produce sharp output when driven
   correctly.

## What ships

- `src/lance_mlx/pipeline/t2v.py`: default `mape_anchor=None` (this commit)
- 47/47 pytest still passes (no regressions)
- Re-run of LTX comparison + HF model card update + github README cleanup
  follow in subsequent commits

## Code surface

The kwarg is still there:

```python
pipe.generate(
    "your prompt here",
    num_frames=17, height=480, width=704,
    num_steps=30, cfg_scale=4.0, seed=42,
    # mape_anchor=None is the default; pass 2000 only to reproduce the
    # legacy buggy behavior for A/B comparison.
)
```

## Scripts

- `scripts/19_oracle_replay.py` — replay Phase 0 oracle config (with --no-mape-shift flag)
- `scripts/20_gutcheck_variants.py` — 4-variant MaPE × cfg_interval matrix at small scale
- `scripts/21_noshift_scale_bisect.py` — 5-probe scale bisect (this finding's source)
