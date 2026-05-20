# Phase 2 — x2t_image MVP plan

**Goal:** end-to-end VQA on Phase 0 oracle case 02 (chart → "29%") using Lance via MLX.

**Validation gate:** the decoded text from Lance MLX should at minimum produce sensible-looking text mentioning the answer. Strict ≥95% token agreement vs the PyTorch oracle is a Phase 2+1 validation step — this session just gets the pipeline running.

**Out of scope this session:** KV cache (full forward each step, slower but simple); the other 5 oracle cases; statistical parity report.

## Architecture

```
                ┌────────────────────────────────────────────────────────┐
                │  scripts/04_x2t_image_demo.py                          │
                └────────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
   Qwen2_5_VLProcessor          VisionModel (mlx-vlm)          LanceModel (ours)
   (chat template,              + vit.safetensors              + model.safetensors
    image preprocessing,        (from Lance-3B-Video-bf16/)    (from Lance-3B-bf16/)
    tokenizer)
   ←── fetched from
   Qwen/Qwen2.5-VL-3B-Instruct
   on HF (tokenizer + chat
   template + image processor
   config only, ~few MB)
```

## Pipeline module — `lance_mlx.pipeline.understanding`

```python
class UnderstandingPipeline:
    """x2t_image / x2t_video VQA via Lance MLX."""

    def __init__(self,
                 lance_weights: Path,   # /Volumes/.../Lance-3B-bf16
                 vit_weights: Path,     # /Volumes/.../Lance-3B-Video-bf16/vit.safetensors
                 hf_processor_repo: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        # 1. Load processor (tokenizer + chat template + image preprocessor)
        # 2. Load VisionModel (mlx-vlm) + vit.safetensors (apply sanitize)
        # 3. Load LanceModel + bf16 safetensors
        ...

    def generate(self, image: PIL.Image, question: str,
                 max_new_tokens: int = 256) -> str:
        # 1. Build chat-templated text with image placeholder
        # 2. processor(images=img, text=text) → input_ids, pixel_values, image_grid_thw
        # 3. ViT forward → image_features (N_post_merger, hidden_size)
        # 4. text_embeds = lance.embed_tokens(input_ids)
        # 5. Merge image_features at image_token_id positions
        # 6. position_ids = compute_position_ids(input_ids, image_grid_thw)
        # 7. position_group = all PositionGroup.TEXT (VQA is all UND)
        # 8. Greedy decode loop:
        #      h = lance(inputs_embeds=inputs_embeds, position_ids=pids, position_group=pg)
        #      logits = lance.lm_head(h[:, -1:, :])
        #      next_id = argmax(logits)
        #      append, extend inputs_embeds, repeat until EOS or max
        return decoded_text
```

## Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| ViT source | bundled `Lance-3B-Video-bf16/vit.safetensors` | Same provenance as the LLM; already on disk; matches inference_lance.sh behavior |
| Processor source | `Qwen/Qwen2.5-VL-3B-Instruct` HF repo | Stock tokenizer + image processor + chat template; small download |
| Target oracle case | 02 ("29%") | Shortest expected answer = fastest validation |
| Position IDs | Adapt mlx-vlm's `get_rope_index` | Image grid handling already correct upstream; lift the function rather than reimplement |
| Position group | All `PositionGroup.TEXT` (all UND) | VQA never touches GEN expert; MaPE is a no-op for modality 0 |
| KV cache | Deferred to Phase 2.1 | Adds complexity (sharing across 36 LanceMoTLayers); ~5s/sample without cache is acceptable for validation |
| Sampling | Greedy (argmax) | Matches Phase 0 oracle behavior (`do_sample=False`) |
| Batching | B=1 only | Phase 2 MVP scope |

## Cost estimate

Per HANDOFF: M5 Max ~10 TFLOPs bf16. One x2t_image case at T_input ≈ 1500–2500 tokens + 5–100 tokens generated:

- Forward at T=2000: ~430 GFLOPs (SDP dominant)
- 100-token greedy without cache: 100 × ~430 GFLOPs ≈ 43 TFLOPs → ~4.3s

Case 02 expected answer is just "29%" (a few tokens) → ~1s. Case 05 (Colosseum, ~120 tokens) → ~5s. All comfortably interactive.

## Known/anticipated issues

1. **ViT weight transpose**: `patch_embed.proj.weight` is 5D from the converter (shape `[1280, 3, 2, 14, 14]`). mlx-vlm's `VisionModel.sanitize()` transposes to `[1280, 2, 14, 14, 3]` on load. Use `sanitize()` when loading our `vit.safetensors`.

2. **fp32 promotion** in LanceModel forward (F32 norm scales). Output is fp32 — `lm_head` and downstream sampling work fine but consume more memory. Phase 5 task.

3. **mlx-vlm's get_rope_index** mutates state (`self._position_ids`, `self._rope_deltas`) — we'll call it as a stateful helper or extract its logic.

4. **Chat template differences**: Qwen2.5-VL's standard template might not match exactly what Lance/inference_lance.sh used. The Phase 0 oracle's prompt.json gives the EXACT expected output format; we may need to compare and adjust.

5. **First-time-load latency**: 6.2B param LanceModel at bf16 = ~12 GB. Fresh load might take a few seconds. Acceptable for demo, cache for repeated runs.

## Success criteria for this session

- ✅ Pipeline runs end-to-end without crashing
- ✅ Output is coherent text (not garbage)
- ✅ Output mentions or is "29%" for case 02

Strict token-level parity to PyTorch oracle is a follow-up.
