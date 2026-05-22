# Phase 5g — P0a (fp32 RoPE) + P0b (sms divisor) implementations

**Date:** 2026-05-21
**Status:** **BOTH CANDIDATES REFUTED by 4-variant gutcheck at 256²×17f.** Patches
remain in tree as off-by-default kwargs for future re-test, but neither
closes the residual fine-detail gap. Continuing with Phase 5f (RockTalk
weights through our pipeline) as the next-best triangulation.
**Source:** `/Volumes/DEV_VOL1/VideoResearch/research_video_notes2.md` (deep-research response to issue #2)

## Empirical verdict (2026-05-21, 256²×17f red-panda-surfing, seed=42)

| Variant | MD5 (midframe.png) | vs V0 | Wall-clock |
|---|---|---|---|
| V0 baseline (sms=1, rope_fp32=False)  | `2ca49d9…` | — | 41.4s |
| V1 sms=2 only                         | `e612884…` | ≠ V0; **worse** (subject lost) | 45.7s |
| V2 rope_fp32 only                     | `2ca49d9…` | **= V0 byte-identical** | 49.7s |
| V3 sms=2 + rope_fp32                  | `e612884…` | **= V1 byte-identical** | 52.4s |

- **P0a (rope_fp32) is a no-op.** V0 ≡ V2 and V1 ≡ V3 at the bit level. The
  fp32-upcast path takes ~10s longer (rotation is genuinely happening in
  fp32) but produces identical bf16-quantized output. Most likely MLX's
  `mx.fast.scaled_dot_product_attention` and the element-wise multiplies in
  `apply_multimodal_rotary_pos_emb` already use fp32 internal accumulators
  on Apple silicon Metal — the bf16 cast at `qwen2_5_vl/language.py:73`
  doesn't actually destroy precision because rounding to bf16 happens at
  store-back regardless of whether the multiply was bf16×bf16 or fp32×fp32.
  **Refutes the research-brief's strongest claim.**

- **P0b (sms=2) actively HURTS output.** V1/V3 lose the red-panda subject
  and degrade to abstract blur — same failure pattern as Candidate 3
  (no_text uncond) from Phase 5d. The Explore agent's initial reading was
  correct: our VAE latent grid is *independent* of ViT's `spatial_merge_size`
  hyperparameter. Upstream Lance's `shift_position_ids` divisor applies to
  ViT-patch grids (UND tower path), not to VAE-latent grids (GEN tower
  path). Lance's GEN tower was trained against position-IDs at the raw VAE
  resolution; dividing by 2 produces under-spread positions and confuses
  the velocity field. **Refutes the research-brief's second claim.**

## Implications

Two of the research's four ranked candidates are now empirically off the
table. The remaining candidates from research_video_notes2.md:

- **P1 (VAE feat_cache audit)** — the research's loudest comparative
  signal: RockTalk shipped a *separate* `Wan2.2-VAE-MLX` port instead of
  reusing `mlx-video`'s decoder. Worth investigating, but our mlx-video
  decoder produced clean t2i output at 768², so any feat_cache bug would
  have to be temporal-chunk-boundary specific.
- **P2 (TimestepEmbedder t*1000 scaling)** — one-line test. Cheap.
- **P3 (o_proj/down_proj precision)** — Reza2kn quant evidence; deferred.

But the cleaner triangulation is **Phase 5f (RockTalk weights through our
pipeline)** — once the 26 GB download finishes, we run their bf16 weights
(remapped via `scripts/22_remap_rocktalk.py`) through OUR pipeline. If
sharp → bug is in our converter (`scripts/02_convert.py`); if still
watercolor → bug is in our pipeline forward-pass code (which is now MUCH
narrower since P0a/P0b are eliminated).

## TL;DR

Research engagement returned two highest-leverage candidates for the residual

## TL;DR

Research engagement returned two highest-leverage candidates for the residual
fine-detail gap on t2v (water/paws/surfboard softness in the panda surfing
prompt). Both are gated experiments that ship in this commit:

1. **P0a — fp32 RoPE rotation** (`--rope-fp32` on `10_t2v_demo.py`,
   `rope_fp32=True` kwarg on `TextToVideoPipeline.generate`). Recovers
   high-frequency precision in the flow-matching velocity field by skipping
   mlx-vlm's bf16 downcast of `cos`/`sin` at
   `qwen2_5_vl/language.py:73`.

2. **P0b — divide latent h/w by spatial_merge_size=2** in `_build_position_ids`
   (`--spatial-merge-size 2` on the demo, `spatial_merge_size=2` kwarg).
   Matches upstream `data/common.py::shift_position_ids` and RockTalk's
   parallel MLX port (their HF card explicitly states `h_patches/sms ×
   w_patches/sms`).

Both default OFF (legacy behavior) so existing t2i/t2v paths are unchanged.
The 4-variant gutcheck (`scripts/23_gutcheck_phase5g.py`) runs all
combinations at 256²×17f on the red-panda-surfing oracle prompt.

## Research's reasoning recap

### P0a — bf16 downcast of cos/sin

`mlx_vlm/models/qwen2_5_vl/language.py:73`:
```python
return cos.astype(x.dtype), sin.astype(x.dtype)
```
Where `x` is the `values` tensor (bf16 in our run). cos/sin are computed in
fp32 then cast down. The actual rotation
`q*cos + rotate_half(q)*sin` (line 100-101) therefore runs in bf16.

Lance's GEN tower is a continuous flow-matching denoiser — bf16 rotation
error perturbs the velocity field across the whole spatial grid, with
**largest error in the high-frequency channels**. That precisely matches the
observed water/paws/surfboard softness while composition (low-freq) stays
correct.

The fix passes `values.astype(fp32)` to `self.rotary_emb()` so its line-73
downcast becomes a no-op (fp32→fp32). q/k are upcast for the rotation, then
downcast back so downstream q@k^T proceeds in bf16 (cheap), with `mx.fast.
scaled_dot_product_attention` doing fp32 softmax internally per MLX spec.

### P0b — spatial_merge_size divisor

Upstream Qwen2.5-VL convention: visual tokens get mrope position-IDs at the
**post-merge** resolution, i.e. `h_patches // spatial_merge_size,
w_patches // spatial_merge_size`. ByteDance's `data/common.py::
shift_position_ids` follows this for Lance's visual tokens. RockTalk's HF
card confirms verbatim: *"Image positions inside `<|vision_start|>..
<|vision_end|>` use 3D mrope grid coords (h_patches/sms × w_patches/sms)"*.

Our current `_build_position_ids` uses raw `h_lat`, `w_lat`, so every visual
token has **twice** the positional spread the model was trained against.
The MoE-gen tower learned against the merged grid → each denoising step's
velocity vector is misaligned in the high-frequency direction. Symptom-
consistent: correct composition, soft texture.

Note: upstream's spatial_merge_size is `2` for Qwen2.5-VL-3B. Whether we
divide t-axis is an open question — currently P0b divides h/w only, matching
the "spatial" in "spatial_merge_size".

## Code changes

### `src/lance_mlx/pipeline/t2v.py`

- `_build_position_ids` gains `spatial_merge_size: int = 1` kwarg (default 1
  = no-op). When `sms > 1`, the loop computes `pos[1, 0, token_pos] = base +
  (r // sms)` and `pos[2, 0, token_pos] = base + (c // sms)`; tail tokens
  start from `max(t_lat - 1, (h_lat - 1) // sms, (w_lat - 1) // sms) + 1`.
- `_prepare_state` gains `spatial_merge_size` kwarg, threaded through.
- `generate` gains `spatial_merge_size: int = 1` and `rope_fp32: bool =
  False` kwargs. At the top, calls
  `self.lance_model.set_rope_fp32(bool(rope_fp32))` to toggle the flag on
  all 36 attention layers.

### `src/lance_mlx/model/lance_llm.py`

- `LanceMoTAttention.__init__` sets `self._rope_fp32 = False`.
- `LanceMoTAttention.__call__` gains an `if self._rope_fp32:` branch around
  the rotary-embed call that upcasts `values`, q, k to fp32; downcasts q,k
  back after rotation.
- `LanceModel.set_rope_fp32(enabled: bool)` walks `self.layers` and sets
  `layer.self_attn._rope_fp32 = bool(enabled)`.

### `scripts/10_t2v_demo.py`

- Three new CLI flags: `--spatial-merge-size <int>`, `--rope-fp32`,
  `--mape-anchor <int>` (the last was an existing kwarg, just newly
  surfaced on the CLI).

### `scripts/23_gutcheck_phase5g.py`

New gutcheck script — 4 variants at 256²×17f, red-panda-surfing prompt
seed=42, 30 steps, CFG=4.0, MaPE=None (no shift, per Phase 5d default):
- `V0_baseline`        : sms=1, rope_fp32=False  (legacy)
- `V1_sms2`            : sms=2, rope_fp32=False  (P0b only)
- `V2_ropefp32`        : sms=1, rope_fp32=True   (P0a only)
- `V3_sms2_ropefp32`   : sms=2, rope_fp32=True   (both)

Saves MP4 + mid-frame PNG per variant to `/tmp/lance_phase5g/` for direct
side-by-side comparison. Wall-clock ~3-4 min total.

## Decision tree

After the gutcheck:

- **V0 == V3 (no visible improvement)** → P0a + P0b are not the bug.
  Fall back to P1 (VAE feat_cache audit) per research brief.
- **V1 sharper but V2 same as V0** → P0b is part of the fix; P0a's
  surgical-upcast may not have caught the right precision regime.
- **V2 sharper but V1 same as V0** → P0a is part of the fix; sms=1 was
  already correct for our slab layout.
- **V3 sharpest, V1/V2 partial** → both additive; ship combined.
- **V3 worse than V0** → one or both candidates broke something
  (e.g. position-id collisions when sms=2 produces too-low max grid coord).

We will A/B at 256² first (fast), then if any variant wins, lift to
480×704×17f (the user-reference scale) for final confirmation.

## Why these came from research, not me

Phase 5e brief listed P0a as one of my own working hypotheses but I had not
implemented it (was deferred behind "is RockTalk-comparison the better
signal?"). P0b I had **not** considered — I had been computing h/w
position-IDs at the raw VAE-compressed grid since Phase 4a and that bug
survived every previous bisect (it doesn't shift composition, only fine
detail).

RockTalk's HF card noting `h_patches/sms × w_patches/sms` was the
critical disambiguation. Their port works and our port has the same MoE-gen
weights — so the only thing in the GEN path that could produce localized
high-freq loss is exactly P0a/P0b.

## Next steps

1. ✅ DONE: Ran `scripts/23_gutcheck_phase5g.py`. Visual grid at
   `/tmp/lance_phase5g/grid_compare.png` shows V1/V3 lose the panda
   subject; V0/V2 retain it. Output identical at the bit level for V0/V2.
2. ⏭ SKIPPED: lifting to 480×704×17f — no candidate survived 256² gut-check.
3. ▶ NEXT: Proceed with the RockTalk-weights-on-our-pipeline test
   (Phase 5f, `scripts/22_remap_rocktalk.py`) once download completes.
   This is now the highest-leverage remaining experiment — definitively
   disambiguates "conversion bug" (our `02_convert.py` loses precision)
   from "pipeline bug" (our `t2v.py` forward-pass has a residual error).
4. After Phase 5f verdict, update `notes/phase5e_research_brief.md` with
   the outcome and update GitHub issue #2 with the empirical evidence
   that P0a and P0b from the research brief are not the bug.
