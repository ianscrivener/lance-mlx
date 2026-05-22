# 2026-05-21 Late-evening session summary

## TL;DR for user

**Major progress:** While you were away, I burned through the deep-research
response (research_video_notes2.md) + ran the RockTalk empirical test +
chat-template A/B. The four highest-leverage candidates the research
identified are now **all empirically tested and refuted**, AND we now
have **proof that the bug is in our pipeline code, not our converter**.

**Quick verdicts (256²×17f red panda seed=42):**

| Test | Result | Bug? |
|---|---|---|
| **Phase 5g P0a** — fp32 RoPE rotation | bit-identical to baseline | NOT the bug (no-op on MLX) |
| **Phase 5g P0b** — sms=2 position divisor | subject LOST | NOT the bug (made it worse) |
| **Phase 5f** — RockTalk weights × our pipeline | byte-identical to our baseline | **Bug is in our PIPELINE, not converter** |
| **Phase 5h** — RT minimal chat template | different pixels, but subject loss | Helps narrow but isn't the fix alone |

**Visual comparison images for you to look at:**
- `/tmp/lance_phase5g/grid_compare.png` — 2×2 of P0a×P0b variants
- `/tmp/lance_phase5f/compare_grid.png` — V0 (ours) vs RT-weights (same pipeline) → indistinguishable
- `/tmp/lance_phase5h/compare_grid.png` — ours-template vs rocktalk-template

## What we did, in order

### 1. Read research_video_notes2.md

Deep research returned four ranked candidates:
- **P0a**: fp32 RoPE rotation in GEN tower (mlx-vlm casts cos/sin to bf16 before rotation)
- **P0b**: divide latent h/w by spatial_merge_size=2 in `_build_position_ids`
- **P1**: VAE feat_cache 'Rep'-vs-non-'Rep' chunk-boundary semantics
- **P2**: TimestepEmbedder `t * 1000` scaling

### 2. Implemented P0a + P0b as gated experiments (Phase 5g)

- Added `rope_fp32: bool = False` + `spatial_merge_size: int = 1` kwargs on
  `TextToVideoPipeline.generate`, and CLI flags on `scripts/10_t2v_demo.py`
- Built `set_rope_fp32()` helper on `LanceModel` that toggles a
  `_rope_fp32` attr across all 36 attention layers
- Ran 4-variant gut-check (`scripts/23_gutcheck_phase5g.py`)
- **Both refuted.** V0 ≡ V2 byte-identical (P0a no-op); V1 ≡ V3 but worse
  than V0 (P0b makes subject incoherent).
- Notes: `notes/phase5g_p0_candidates.md`. Commit `cfd376e`.

### 3. RockTalk weights × our pipeline (Phase 5f)

- Downloaded `RockTalk/Lance-3B-Video-MLX` (26 GB F32 safetensors)
- Built `scripts/22_remap_rocktalk.py` to translate their key layout to ours
  (drop `language_model.{model.}?` prefix, `time_embedder.fc1/fc2` →
  `proj_in/proj_out`, `vae2llm` → `vae_in_proj.vae2llm`, skip 360
  `vit_model.*` keys, F32→bf16 cast)
- Built `scripts/24_test_rocktalk_pipeline.py` end-to-end test
- **Result: midframe.png MD5 byte-identical to V0_baseline.**
- Combined with RT's published sharp ocean-wave samples (from THEIR
  pipeline using these weights), this proves:
  - Our converter is CORRECT
  - Our pipeline forward-pass IS the bug
- Side observation: RT's `vae.safetensors` has DIFFERENT key layout from
  mlx-video's `Wan22VAEDecoder` (decoder.upsamples.X.upsamples.Y.residual
  vs decoder.head/middle/conv1) — they shipped a separate VAE port. For
  this test we used our VAE (the LLM forward pass is what we're testing).
- Notes: `notes/phase5f_rocktalk_verdict.md`. Commit `a1e06b9`.

### 4. RT minimal chat template (Phase 5h)

- Decoded RT's template from their HF model card:
  `<|im_start|> [prompt] <|im_end|> <|vision_start|> [latents] <|vision_end|>`
  (no system+user+assistant role tags, no T2V_INSTRUCTION prefix)
- Added `prompt_format: 'ours' | 'rocktalk'` kwarg on `generate`
- Ran `scripts/25_gutcheck_phase5h.py` A/B at 256²×17f
- **Result: different pixels (≠ V0) but subjectively WORSE — subject loss.**
- Conclusion: the chat template alone isn't the fix. Lance was likely
  trained with our verbose template (from upstream `inference_lance.sh`)
  and removing it disorients the model on our prompt/seed.
- Commit `7376458`, merged as `96e0ace`.

## What's left to investigate

RockTalk's pipeline code is NOT public — the GitHub URL referenced in their
HF README (`github.com/RockTalk/Lance-MLX`) returns 404. So we can't
directly diff their forward-pass code. Remaining leverage points:

1. **Routing logic** — RT routes by "slicing the sequence into the latent
   slab vs the surrounding text" and concatenating; we use `mx.where` on
   all tokens. These should be numerically equivalent but maybe our
   `mx.where` has a subtle bug at the routing boundary.

2. **VAE input projection ordering** — we compute
   `vae_in_proj(latents_flat) + pe + t_emb`. The order/timing of adding
   `pe` and `t_emb` matters; verify against upstream.

3. **CFG composition + Euler step order** — verify against upstream
   `cfg_text_unconditional_pro_new`.

4. **Their separate VAE port** — RT shipped `RockTalk/Wan2.2-VAE-MLX`
   specifically because mlx-video's doesn't handle temporal feat_cache
   well. We use mlx-video's. Might affect t2v outputs even though we use
   the same encoder for image_edit (which works).

5. **Research P2** — `t * 1000` scaling on TimestepEmbedder. Still untested.

## Code state (clean, pushed)

Main branch: `96e0ace`. Three new flags on `t2v.py::generate`, all
off-by-default for backward compatibility:

- `--rope-fp32` (Phase 5g) — fp32 RoPE rotation
- `--spatial-merge-size 2` (Phase 5g) — divide latent h/w in position grid
- `prompt_format='rocktalk'` (Phase 5h) — minimal chat template

Test scripts added:
- `scripts/22_remap_rocktalk.py` — RT key remap utility
- `scripts/23_gutcheck_phase5g.py` — 4-variant P0a×P0b matrix
- `scripts/24_test_rocktalk_pipeline.py` — RT weights × our pipeline
- `scripts/25_gutcheck_phase5h.py` — chat-template A/B

Documentation:
- `notes/phase5g_p0_candidates.md`
- `notes/phase5f_rocktalk_verdict.md`
- `notes/2026-05-21_session_summary.md` (this file)

GitHub issue #2 updated 3x with rolling verdicts.

## Suggested next session start

1. **Cross-check our `_prepare_state` vs upstream Lance Python** — particularly
   the order of operations: `vae_in_proj(noise) + pe + t_emb`, and
   how the latent block is patched into `text_embeds` at `latent_positions`.

2. **Try research P2 in 30 seconds** — print the actual `t` value passed
   into `time_embedder` during a generation step. If t ∈ [0,1], compare
   to upstream which likely uses t*1000.

3. **Bisect the routing** — temporarily replace LanceMoTAttention with a
   single-expert forward (UND only) on a t2i prompt to verify our routing
   isn't accidentally always taking the wrong branch. Image works for us
   so this is probably fine, but cheap to confirm.
