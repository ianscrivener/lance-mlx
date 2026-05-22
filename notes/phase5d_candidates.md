# Phase 5d — t2v port-bug debug ladder

Pre-staged candidates for the [issue #2](https://github.com/xocialize/lance-mlx/issues/2)
t2v port-quality investigation. Each candidate addresses a specific deviation
from upstream Lance that we've identified. Candidates are intended to be
*additive* — apply the most upstream-faithful set first, fall back to layering
if one candidate alone doesn't land the fix.

## Candidate priority (descending prior probability)

### Candidate 0 — Remove the MaPE 2000-anchor we add (RUNNING)

**Deviation:** Our `t2v.py::_build_position_ids` re-anchors latent t-axis to
`MAPE_ANCHOR_VIDEO_GEN=2000`. Upstream's `data/common.py::shift_position_ids`
is gated by `attn_mode in ["full_noise", "full"]`. Pure t2v has only `"noise"`
attn_mode for the target latents — the gate never fires. **Upstream does not
re-anchor for pure t2v.**

**Patch:** Pass `mape_anchor=None` to `TextToVideoPipeline.generate()`.

**Implementation:** Already in t2v.py (commit `5765569`). `mape_anchor: int | None`
kwarg defaults to 2000 (legacy); None skips the shift.

**Test:** `scripts/19_oracle_replay.py --no-mape-shift` against oracle 000000.

---

### Candidate 1b — Add `cfg_interval=[0.4, 1.0]` (NEW, high prior)

**Deviation:** Upstream `config_factory.py` defaults `cfg_interval = [0.4, 1.0]`,
applying CFG=4.0 only when `t > 0.4 and t <= 1.0`. For t <= 0.4 (the late-stage
denoising steps), CFG scale collapses to 1.0 (no CFG). Our MLX port applies
CFG=4.0 at *every* step. The late-stage extra CFG likely over-smooths and
produces the painterly softness we see.

For a 30-step run with `timestep_shift=3.5`, the schedule starts at t=1.0 and
falls to t=0.0. CFG would apply for the first ~14-17 steps (while t > 0.4)
then collapse for the last ~13-16 steps. Lance is specifically trained
expecting CFG to vanish in the final stages.

**Patch:** Pass `cfg_interval=(0.4, 1.0)` to `TextToVideoPipeline.generate()`.

**Implementation:** Already in t2v.py (this commit). `cfg_interval: tuple[float, float] | None`
kwarg. None (default) = legacy every-step CFG; pass tuple = upstream behavior.

**Test:** `scripts/19_oracle_replay.py --cfg-interval 0.4,1.0` against oracle.

---

### Candidate 1 — fp32 RoPE rotation throughout

**Deviation:** mlx-vlm's `Qwen2RotaryEmbedding.__call__` computes cos/sin in
fp32 but casts back to `x.dtype` (bf16) before they're applied via
`apply_multimodal_rotary_pos_emb`. The rotation `(q * cos) + (rotate_half(q) * sin)`
runs in bf16. At our larger position values, the bf16 rotation may lose
detail PyTorch's fp32 preserves.

**Patch:** Wrap the rotary application to upcast q, k, cos, sin to fp32 for
the multiply-add, then downcast q_embed, k_embed back to bf16.

**Implementation:** Pre-staged location: would land in
`src/lance_mlx/model/lance_llm.py::LanceMoTAttention.__call__` right after
`self.rotary_emb(values, position_ids)` is called.

**Test:** Requires t2v.py to plumb a `fp32_rope: bool` flag through to
LanceMoTAttention — slightly more invasive, but mechanical.

---

### Candidate 2 — Per-frame attention-mask scope

**Deviation candidate:** Maybe video should treat each frame as a separate
"noise segment" with bidir-within-frame + causal-across-frame.

**Status of investigation:** Read of upstream `val_ds.py::process_text_template`
shows ONE `"noise"` attn_mode appended per VAE block, regardless of frame
count. So upstream uses one bidir block matching our impl. **Likely not the bug.**

**Patch:** N/A (not really a deviation).

---

## Execution order

1. **Now (running):** Candidate 0 alone. Output → `/tmp/lance_oracle_replay_cand0/`.
2. **If Cand 0 doesn't land it:** Candidate 1b alone. Output → `/tmp/lance_oracle_replay_cand1b/`.
3. **If neither alone:** Candidate 0 + 1b combined. Output → `/tmp/lance_oracle_replay_cand0_1b/`.
4. **If still off:** Candidate 1 (fp32 RoPE), additive to whatever helped.

Each cycle is ~2.25 hours wall-clock at oracle scale (768×768×50f). For faster
iteration we can test at 256² × 17f first (~30 s/run) — same architecture, no
oracle-direct comparison but rapid signal on whether the patch changes
aesthetic at all.

## Comparison oracle

`tests/fixtures/results/t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630/000000.mp4`
— the red panda surfing, photorealistic 3D-cinematic. This is the ground
truth we're trying to match (or get close to with bf16-rounding-expected drift).
