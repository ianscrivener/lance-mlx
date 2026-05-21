# Phase 4e — video debug loop plan

**Status:** plan only, executed iff Phase 4c reference-scale (480×848×121f)
discriminator confirms the bug is NOT just resolution-OOD.

## Discriminator branch logic

The 480×848×121f reference-scale run (n_lat = 30 × 53 × 31 = 49,290) is in
flight. When it completes:

### Branch A — coherent output at reference scale
- **Diagnosis:** 768²×17f failure WAS off-distribution. Lance was trained at
  480p; 768² aspect ratio puts grid coords outside the trained envelope.
- **Action:** close [issue #1](https://github.com/xocialize/lance-mlx/issues/1)
  as "OOD config, fixed by using reference scale." Mark `t2v at 480p` as
  ✅ Production in the model card. Move to Phase 4d video_edit validation
  (already scaffolded).

### Branch B — still noise at reference scale
- **Diagnosis:** real architectural bug — bug is in our forward pass, not
  scale-related.
- **Action:** execute the debug loop below. Self-describe pattern proven in
  Phase 3d/3e for image works equally well for video — each candidate gets
  validated against a "what's in the middle frame?" self-describe via x2t_image.

### Branch C — partial degradation (e.g., recognizable subject + temporal artifacts)
- **Diagnosis:** scale-related but not catastrophic. Likely a numerical
  precision drift at large n_lat (e.g., bf16 RoPE precision loss).
- **Action:** run Candidate 1 (fp32 RoPE) first; if that helps, ship; if
  not, run Candidate 2/3.

## Debug-loop recipe (Branch B/C)

### Setup
- Branch: `phase4e-video-debug`
- Test scale: **fixed 768×768 × 17 frames** (n_lat = 11,520; just past the
  known failure threshold; small enough to iterate ~30 s per attempt with
  CFG=4.0 + 30 steps, ~5 min/attempt).
- Prompt: `"Five balls on a wooden table — two are blue, three are green."`
  Strong-content. Easy ground truth. Asking x2t_image "How many balls
  total, and what colors?" gives a clean correctness signal.
- Self-score: pass `scripts/11_t2v_frame_bisect.py` with `--frames 17`
  modified to use this prompt and to ask the counting question.
- One change per iteration. Document each candidate's diff + result.

### Candidates (in priority order)

1. **fp32 RoPE freqs + position grid**  
   *Hypothesis:* bf16 has 7 mantissa bits; at n_lat ≥ ~11,520 adjacent
   position indices can quantize to the same value, collapsing mRoPE.
   Documented as a real failure mode in `Blaizzy/mlx-video` LTX.  
   *Change:* in `src/lance_mlx/model/mape.py` and wherever rotary freqs
   are constructed (likely inherited from mlx-vlm), force `mx.float32` for
   the freq table + the position grid. Cast back to bf16 only at output.  
   *Cost:* maybe 5% slower step, no extra memory.  
   *Strong indicator:* if 768²×17f goes from noise → recognizable, this
   was the bug and likely fixes the upper scales too.

2. **Per-frame timestep_embed scoping**  
   *Hypothesis:* analog to Phase 3d fix. We currently broadcast a single
   `time_embedder(t)` across all `t_lat` frames. Upstream Lance assigns
   one timestep value per noise position, all set to the same t — this is
   what we do. But if there's some per-frame timestep variation we're
   missing (e.g., temporal-MaPE-aware schedule), the broadcast collapses
   it.  
   *Change:* compute `time_embedder(t)` per-frame, e.g. with a tiny
   temporal offset matching the MaPE temporal anchor. Or verify upstream
   really does broadcast (re-check `lance_lance.py:644-660`).  
   *Cost:* trivial.  
   *Strong indicator:* output regains temporal structure across frames.

3. **Cross-frame attention mask shape**  
   *Hypothesis:* our `_build_block_mask` makes the ENTIRE noisy-latent
   block (all t_lat × h_lat × w_lat tokens) bidirectional. Upstream might
   only bidir WITHIN each frame's slab and causal across frames. If true,
   at t_lat = 5 (n_lat=11,520) frame 0 starts to see frame 4's noise
   acausally and confusion compounds.  
   *Change:* split the bidir region in `_build_block_mask` to per-frame:
   `bidir = (in_frame_0 & in_frame_0) | (in_frame_1 & in_frame_1) | ...`.
   Compare to upstream `create_sparse_mask` behavior on a multi-frame
   sample.  
   *Cost:* slightly more compact mask.  
   *Strong indicator:* frame-0 content stays coherent across all frame
   counts.

4. **latent_pos_embed indexing order**  
   *Hypothesis:* our flatten is `f * 4096 + r * 64 + c` (frame-major,
   row-major within frame). Upstream computes via
   `get_flattened_position_ids_extrapolate_video(t, h, w, max_latent_size=64)`
   — need to verify that function's flatten convention.  
   *Change:* if upstream is, e.g., row-major across all frames combined
   (`r * 64 + c` independent of f, no temporal stride), the LPE lookup
   today is sampling the wrong cells at high t_lat.  
   *Cost:* trivial.  
   *Strong indicator:* the higher-frame-index frames suddenly become
   non-noise.

### Stopping criteria
- **Success:** middle frame's self-describe returns "balls" + count present.
  Close branch B, ship.
- **Failure after 4 candidates:** document each attempt in `phase4e_findings.md`
  with the actual self-describe output and a one-line hypothesis-rejection
  rationale. Branch B requires upstream-source archaeology beyond our
  current envelope; defer for a research session.

### Cross-validation
After any successful candidate, re-run the original bisection grid
(`scripts/11_t2v_frame_bisect.py`) to confirm:
- 1f (n_lat=2,304): still coherent (no regression on t2i scale)
- 5f, 9f, 13f: still coherent (no regression on previously-working scales)
- 17f, 25f, 49f: now coherent (the fix actually scales)

If the candidate fixes 17f but breaks 5f, it's not the right fix; document
and revert.

## Tools available
- `scripts/11_t2v_frame_bisect.py` — already runs the bisection grid +
  self-describes middle frame of each. Modify the prompt + question pair
  for our debug session.
- `scripts/04_x2t_image_demo.py` — for manual frame inspection if needed.
- `notes/phase3_findings.md` + `notes/phase3e_*.md` — image-side
  precedent for how this same self-describe pattern unblocked t2i.
