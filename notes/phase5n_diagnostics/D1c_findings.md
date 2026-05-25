# Phase 5n / D1c — real-prompt t2i frame-index A/B

**Date:** 2026-05-24
**Goal:** Decide whether `t2i.py:286` and `image_edit.py:302` should
keep picking `decoded[0, 0]` from the 3-frame T_latent=1 VAE output,
or switch to frame 1 / frame 2.
**Cost:** ~75 s of generation (3 prompts × ~25 s @ 384²).

## TL;DR

**Frame 0 is correct. Keep `decoded[0, 0]`. No patch needed.**

The D1b follow-up question is closed: across three real, denoised
oracle prompts (typography, photoreal fox, atmospheric cyberpunk),
**frame 0 wins on every measurable axis and looks sharpest by eye**.
Frame 2, which D1b's noise-only test suggested was the "high-detail"
frame, is in fact the softest of the three on real latents.

## Why D1b's hint was misleading

D1b measured HF energy on a **pure Gaussian latent** (the VAE
operating on out-of-distribution input). With structured, denoised
latents the ranking reverses:

| Test           | Frame 0 HF | Frame 1 HF | Frame 2 HF | Winner |
|----------------|-----------:|-----------:|-----------:|:------:|
| D1b noise-only |   9.20e+05 |   1.14e+06 |   1.65e+06 | f2     |
| D1c real prompt p10 (typography)   | **3.02e+06** | 2.73e+06 | 2.00e+06 | **f0** |
| D1c real prompt p01 (fox_grass)    | **1.71e+06** | 1.60e+06 | 1.23e+06 | **f0** |
| D1c real prompt p06 (neon_alley)   | **2.96e+06** | 2.69e+06 | 2.14e+06 | **f0** |

Frame 0 has the highest HF energy *and* the widest dynamic range on
real latents — later frames compress contrast (e.g. p10 f2 hits only
[-0.985, +0.923] vs f0's full [-1.0, +1.0]) and progressively soften.

## Visual judgment

Grid: `notes/phase5n_diagnostics/d1c_real_prompt_frame_ab/_grid_3prompts_x_3frames.png`

- **p10 (sign "Open")** — f0 has the crispest lettering and the
  most detail in the suspended cords; f1 is close; f2 is visibly
  muddied.
- **p01 (fox)** — f0 shows the sharpest fur and grass-blade
  detail with the brightest highlights; f2's fur looks waxier and
  grass softer.
- **p06 (neon alley)** — f0 has the deepest blacks, crispest
  neon, sharpest building geometry, and the most droplet detail
  in the wet pavement; f2 looks slightly fogged.

The trend matches the HF and std numbers exactly: a monotonic
soft-and-compress walk from f0 → f1 → f2.

## Interpretation

The Wan2.2 VAE in its `T_latent=1` regime emits three "frames" that
are best understood as a **single image with progressive
low-pass and contrast compression**. With structured latents from
the trained Lance flow head, the first emission carries the full
detail signal; later emissions look like internal causal-padding
relaxations that monotonically smooth toward a neutral DC.

Lance was almost certainly trained against frame-0 as the t2i
target (consistent with the comment at `t2i.py:285`), so the
calibration of LLM-side latents and the VAE's frame-0 decode are
locked together. Switching the index would silently downgrade the
ship.

## Action items

**Closes:**
- "Is `decoded[0, 0]` the right index for t2i?" — yes, confirmed
  on real latents across three categories. No patch to
  `t2i.py:286` or `image_edit.py:302`.

**Open (still worth a follow-up at some point):**
- The "synthetic T_latent=2 → take frame 0 of the higher-detail
  regime" idea from D1's action items remains untested. It would
  require duplicating or noise-padding the latent into a second
  temporal slot before decode, which may or may not move the
  output back into the VAE's "good" T≥2 regime. Lower priority
  than the video-side investigation; the current ship works.

## Files

- Script: `scripts/diagnostics/d1c_real_prompt_frame_ab.py`
- Grid:   `notes/phase5n_diagnostics/d1c_real_prompt_frame_ab/_grid_3prompts_x_3frames.png`
- Stats:  `notes/phase5n_diagnostics/d1c_real_prompt_frame_ab/stats.md`
- Per-frame PNGs: `notes/phase5n_diagnostics/d1c_real_prompt_frame_ab/{pid}_frame{0,1,2}.png`
