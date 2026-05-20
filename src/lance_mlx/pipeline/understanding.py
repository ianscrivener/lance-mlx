"""x2t_image / x2t_video — VQA via Lance MLX.

Phase 2 MVP: end-to-end pipeline composing:
  1. mlx-vlm's `Qwen2_5_VLProcessor` (chat template + image preprocessing + tokenizer)
  2. mlx-vlm's `VisionModel` (Qwen2.5-VL ViT, loaded from `vit.safetensors`)
  3. Our `LanceModel` (loaded from `model.safetensors`)
  4. Greedy decode loop (no KV cache — Phase 2.1 follow-up)

Public API: `UnderstandingPipeline.from_pretrained(...).generate(image, question)`.

Position-ID construction (`_compute_position_ids`) is adapted from
mlx-vlm's `LanguageModel.get_rope_index`. The image-grid handling is
load-bearing for mRoPE to encode 2D positions correctly inside images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen2_5_vl.config import TextConfig, VisionConfig
from mlx_vlm.models.qwen2_5_vl.vision import VisionModel
from PIL import Image

from lance_mlx.model import LanceModel
from lance_mlx.model.routing import PositionGroup


# ---------------------------------------------------------------------------
# Position-ID construction (adapted from mlx-vlm get_rope_index)
# ---------------------------------------------------------------------------

def _compute_position_ids(
    input_ids: mx.array,                         # (B, T)
    image_grid_thw: Optional[mx.array],          # (n_images, 3) or None
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    attention_mask: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """3D position IDs for mRoPE with image-grid handling.

    Adapted near-verbatim from mlx-vlm's
    `Qwen2_5_VLLanguageModel.get_rope_index`. Returns
    `(position_ids: (3, B, T), mrope_position_deltas: (B, 1))`.

    For text-only or no-image input, position IDs degrade to the standard
    1D linear positions broadcast to 3 axes. For images, the image region's
    positions are replaced by a 3D grid (T_grid, H_grid/merge, W_grid/merge).
    """
    video_grid_thw = None  # we don't handle video yet
    batch_size, seq_length = input_ids.shape

    if image_grid_thw is None and video_grid_thw is None:
        # Text-only fast path.
        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(
                attention_mask == 0, mx.ones_like(position_ids), position_ids
            )
            max_position_ids = position_ids.max(axis=-1, keepdims=True)
            position_ids = mx.broadcast_to(position_ids[None, :, :], (3, *position_ids.shape))
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(seq_length).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_length))
            mrope_position_deltas = mx.zeros([batch_size, 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas

    # Image-grid path (adapted from mlx-vlm).
    if attention_mask is None:
        attention_mask = mx.ones_like(input_ids)
    position_ids = mx.ones((3, batch_size, seq_length), dtype=input_ids.dtype)
    image_index = 0
    mrope_position_deltas = []

    for i in range(batch_size):
        ids = mx.where(attention_mask[i] == 1, input_ids[i], mx.zeros_like(input_ids[i]))
        vision_start_indices = mx.sum(
            mx.where(
                ids == vision_start_token_id,
                mx.arange(ids.shape[0]),
                mx.zeros_like(ids),
            )
        )
        vision_tokens = ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum().item()
        input_tokens = ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images = image_nums

        for _ in range(image_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed = input_tokens.index(image_token_id, st)
            else:
                ed = len(input_tokens) + 1
            t, h, w = (
                image_grid_thw[image_index][0],
                image_grid_thw[image_index][1],
                image_grid_thw[image_index][2],
            )
            image_index += 1
            remain_images -= 1
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0 else 0
            )
            index = mx.arange(text_len).reshape(1, text_len)
            index = mx.broadcast_to(index, (3, text_len)) + st_idx
            llm_pos_ids_list.append(index)

            # 3D image grid (t, h, w)
            t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
            t_index = mx.broadcast_to(t_index, (llm_grid_t, llm_grid_h * llm_grid_w)).flatten()
            h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
            h_index = mx.broadcast_to(h_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()
            w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
            w_index = mx.broadcast_to(w_index, (llm_grid_t, llm_grid_h, llm_grid_w)).flatten()
            llm_pos_ids_list.append(
                mx.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = (
                llm_pos_ids_list[-1].max() + 1
                if len(llm_pos_ids_list) > 0 else 0
            )
            text_len = len(input_tokens) - st
            t_index = mx.arange(text_len).reshape(1, text_len)
            t_index = mx.broadcast_to(t_index, (3, text_len))
            llm_pos_ids_list.append(t_index + st_idx)

        llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mask = mx.array(attention_mask[i] == 1)
        expanded_mask = mx.expand_dims(mask, axis=0)
        expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
        expanded_positions = mx.expand_dims(llm_positions, axis=1)
        new_positions = mx.where(
            expanded_mask, expanded_positions, position_ids[:, i:i+1, :]
        )
        updated_position_ids = mx.concatenate(
            [position_ids[:, :i, :], new_positions, position_ids[:, i+1:, :]],
            axis=1,
        )
        position_ids = updated_position_ids
        mrope_position_deltas.append(llm_positions.max() + 1 - len(input_tokens))

    mrope_position_deltas = mx.array(mrope_position_deltas).reshape(-1, 1)
    return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# Inputs-embeds assembly (text + ViT features merged at image-pad positions)
# ---------------------------------------------------------------------------

def _merge_text_embeds_and_image_features(
    text_embeds: mx.array,        # (B, T, D)
    image_features: mx.array,     # (N_post_merger, D)
    input_ids: mx.array,          # (B, T)
    image_token_id: int,
    video_token_id: int,
) -> mx.array:
    """Replace image-pad token embeddings with ViT features.

    Mirrors mlx-vlm's `Model.merge_input_ids_with_image_features`.
    Returns `(B, T, D)` with ViT features slotted at image-pad positions.
    """
    image_positions = input_ids == image_token_id
    if mx.sum(image_positions).item() == 0:
        image_positions = input_ids == video_token_id

    B, T = input_ids.shape
    batch_outputs = []
    feature_start_idx = 0

    for b in range(B):
        mask = image_positions[b]
        n = mx.sum(mask).item()
        if n > 0:
            features = image_features[feature_start_idx : feature_start_idx + n]
            cumsum = mx.cumsum(mask.astype(mx.int32))
            feature_indices = mx.where(mask, cumsum - 1, 0)
            gathered = features[feature_indices]
            mask_expanded = mx.expand_dims(mask, axis=-1)
            out = mx.where(mask_expanded, gathered, text_embeds[b])
            feature_start_idx += n
        else:
            out = text_embeds[b]
        batch_outputs.append(out)

    return mx.stack(batch_outputs, axis=0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class UnderstandingPipeline:
    """x2t_image / x2t_video VQA via Lance MLX.

    Loads:
      - mlx-vlm's `Qwen2_5_VLProcessor` from a HF model repo (small download)
      - mlx-vlm's `VisionModel` from a local `vit.safetensors`
      - Our `LanceModel` from a local `model.safetensors`

    Public method `generate(image, question)` returns the decoded answer string.
    """

    def __init__(
        self,
        lance_model: LanceModel,
        vision_model: VisionModel,
        processor,                     # Qwen2_5_VLProcessor or similar
        text_config: TextConfig,
        vision_config: VisionConfig,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        eos_token_id: int,
    ):
        self.lance_model = lance_model
        self.vision_model = vision_model
        self.processor = processor
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    def from_pretrained(
        cls,
        lance_weights_dir: Path | str,
        vit_safetensors: Path | str,
        hf_processor_repo: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ) -> "UnderstandingPipeline":
        """Load all three components.

        Args:
            lance_weights_dir: dir produced by `scripts/02_convert.py` for the
                LLM (contains `model.safetensors` + `config.json`).
            vit_safetensors: path to `vit.safetensors` (typically the bundled
                one from `Lance-3B-Video-bf16/`).
            hf_processor_repo: HF repo to fetch tokenizer + image processor
                + chat template from. Defaults to stock Qwen2.5-VL-3B.
        """
        lance_weights_dir = Path(lance_weights_dir)
        vit_safetensors = Path(vit_safetensors)

        # 1. Processor (tokenizer + image preprocessor + chat template).
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_processor_repo)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # 2. LanceModel from converter output.
        cfg = json.loads((lance_weights_dir / "config.json").read_text())
        text_cfg = TextConfig(
            model_type=cfg["model_type"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            vocab_size=cfg["vocab_size"],
            num_key_value_heads=cfg.get("num_key_value_heads"),
            max_position_embeddings=cfg.get("max_position_embeddings", 128000),
            rope_theta=cfg.get("rope_theta", 1e6),
            rope_scaling=cfg.get("rope_scaling"),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )
        saved_lance = mx.load(str(lance_weights_dir / "model.safetensors"))
        num_latent_positions = saved_lance["latent_pos_embed.pos_embed"].shape[0]
        lance_model = LanceModel(text_cfg, num_latent_positions=num_latent_positions)
        lance_model.load_weights(list(saved_lance.items()))
        mx.eval(lance_model.parameters())

        # 3. VisionModel from local vit.safetensors. mlx-vlm's `sanitize()`
        #    handles the patch_embed.proj.weight 5D transpose.
        # HF and mlx-vlm differ in some config field names; filter + rename.
        import inspect as _inspect
        _vc_fields = set(_inspect.signature(VisionConfig).parameters)
        _hf_vision = dict(cfg["vision_config"])
        # HF uses "in_chans"; mlx-vlm uses "in_channels".
        if "in_chans" in _hf_vision and "in_channels" not in _hf_vision:
            _hf_vision["in_channels"] = _hf_vision.pop("in_chans")
        # Drop fields mlx-vlm doesn't model (e.g. hidden_act — implicit silu).
        _vision_kwargs = {k: v for k, v in _hf_vision.items() if k in _vc_fields}
        _vision_kwargs.setdefault("model_type", "qwen2_5_vl")
        vision_cfg = VisionConfig(**_vision_kwargs)
        vision_model = VisionModel(vision_cfg)
        saved_vit = mx.load(str(vit_safetensors))
        saved_vit = vision_model.sanitize(saved_vit)
        vision_model.load_weights(list(saved_vit.items()))
        mx.eval(vision_model.parameters())

        return cls(
            lance_model=lance_model,
            vision_model=vision_model,
            processor=processor,
            text_config=text_cfg,
            vision_config=vision_cfg,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            eos_token_id=eos_token_id,
        )

    # --------- generation -------------

    def generate(
        self,
        image: Image.Image,
        question: str,
        *,
        max_new_tokens: int = 256,
        verbose: bool = False,
    ) -> str:
        """Greedy-decode an answer to `question` about `image`.

        Returns the decoded answer text (without the chat-template suffix).
        """
        # 1. Build chat-templated prompt with image placeholder.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # 2. Preprocess: expands image_token to N image_pad tokens, returns pixel_values
        #    + image_grid_thw + input_ids.
        inputs = self.processor(images=image, text=text, return_tensors="mlx")
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        if verbose:
            print(f"  prompt tokens: {input_ids.shape[-1]}")
            print(f"  image_grid_thw: {image_grid_thw.tolist()}")

        # 3. ViT forward.
        vit_dtype = self.vision_model.patch_embed.proj.weight.dtype
        image_features = self.vision_model(
            pixel_values.astype(vit_dtype), image_grid_thw,
        )
        if verbose:
            print(f"  vit features: {image_features.shape}")

        # 4. Text embeddings + merge with ViT features at image-pad positions.
        text_embeds = self.lance_model.embed_tokens(input_ids)
        inputs_embeds = _merge_text_embeds_and_image_features(
            text_embeds, image_features, input_ids,
            self.image_token_id, self.video_token_id,
        )

        # 5. Position IDs (3D for mRoPE, image-grid-aware).
        position_ids, _ = _compute_position_ids(
            input_ids, image_grid_thw,
            spatial_merge_size=self.vision_config.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
        )

        # 6. Position group: all UND (x2t never touches GEN).
        T = input_ids.shape[1]
        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)

        # 7. Greedy decode loop (no KV cache for v1).
        generated_ids: list[int] = []
        for step in range(max_new_tokens):
            h = self.lance_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                position_group=position_group,
            )
            # Logits at the LAST position only — that's where the next token comes from.
            logits = self.lance_model.lm_head(h[:, -1:, :])  # (1, 1, vocab)
            next_token = mx.argmax(logits[:, -1, :], axis=-1).item()  # scalar
            generated_ids.append(next_token)

            if verbose and step < 10:
                tok_str = self.processor.tokenizer.decode([next_token])
                print(f"  step {step}: token {next_token} ({tok_str!r})")

            if next_token == self.eos_token_id:
                break

            # Append the new token to the sequence for the next step.
            next_embed = self.lance_model.embed_tokens(
                mx.array([[next_token]], dtype=input_ids.dtype)
            )
            inputs_embeds = mx.concatenate([inputs_embeds, next_embed], axis=1)
            input_ids = mx.concatenate(
                [input_ids, mx.array([[next_token]], dtype=input_ids.dtype)], axis=1,
            )
            # Extend position_ids by one along the T axis (continuing the max).
            last_pos = position_ids[:, :, -1:]
            position_ids = mx.concatenate([position_ids, last_pos + 1], axis=2)
            position_group = mx.concatenate(
                [position_group, mx.array([int(PositionGroup.TEXT)], dtype=mx.int32)], axis=0,
            )

        # 8. Decode.
        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
