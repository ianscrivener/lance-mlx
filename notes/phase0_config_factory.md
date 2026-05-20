# Pre-flight notes — `config/config_factory.py` defaults + `inference_lance.sh` overrides

**Source:** `bytedance/Lance@main` — `config/config_factory.py` (247 LOC, 6 dataclasses), `config/__init__.py`, `config/path_default.yaml`, `inference_lance.sh`, four task example JSONs. Fetched 2026-05-19.

This document is the formal spec for what Phase 0 actually runs on the rented pod. The runbook tells you *how* to launch; this tells you *with what flags* and *with what file expectations*.

## TL;DR — three things to know before you start

1. **The shipped `inference_lance.sh` overrides several dataclass defaults in load-bearing ways.** Notably `vae_model_type` (default `"seedance"`, shell uses `"wan"`), `latent_patch_size` (default `[1,2,2]`, shell uses `[1,1,1]`), `max_num_frames` (default `25`, shell uses `121`), and `apply_qwen_2_5_vl_pos_emb` (default `False`, shell uses `true`). The dataclass defaults are *not* the operational defaults — always read the shell file alongside.
2. **The video model serves image tasks too.** `inference_lance.sh` sets `MODEL_PATH="downloads/Lance_3B_Video"` and runs `--task x2t_image` with it. So you can do **all 6 tasks from the Lance_3B_Video checkpoint alone** (~28.4 GB) — you do not have to download Lance_3B (~24.7 GB) for image tasks. This **roughly halves Phase 0 download time** if you stick to one variant. Both checkpoints share the same `llm_config.json` byte-for-byte (verified in the upstream verification pass), so they're architecturally identical, just trained on different data mixes.
3. **No custom special tokens — but `<|quad_start|>` (151657) and `<|quad_end|>` are the VAE-token insertion anchors** in the T2I chat template. They're standard Qwen2.5-VL vocab tokens repurposed as placeholders that get *replaced* with actual VAE latent tokens during sequence assembly. Earlier scaffold guesses of `BOT/EOT/BOV/EOV` markers were wrong.

## Canonical run command (verbatim from `inference_lance.sh`)

This is what actually runs at ByteDance. Reproduce it exactly to make every MLX-port output diffable.

```bash
accelerate launch \
    --num_machines $NUM_MACHINES --num_processes $TOTAL_RANK \
    --machine_rank $MACHINE_RANK --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MAIN_PROCESS_PORT --mixed_precision bf16 \
    inference_lance.py \
    --model_path                  "downloads/Lance_3B_Video" \
    --vit_type                    qwen_2_5_vl_original \
    --llm_qk_norm                 true \
    --llm_qk_norm_und             true \
    --llm_qk_norm_gen             true \
    --tie_word_embeddings         false \
    --validation_num_timesteps    30 \
    --validation_timestep_shift   3.5 \
    --copy_init_moe               true \
    --max_num_frames              121 \
    --max_latent_size             64 \
    --latent_patch_size           1 1 1 \
    --visual_und                  true \
    --visual_gen                  true \
    --vae_model_type              wan \
    --apply_qwen_2_5_vl_pos_emb   true \
    --apply_chat_template         false \
    --cfg_type                    0 \
    --validation_data_seed        42 \
    --video_height                768 \
    --video_width                 768 \
    --num_frames                  50 \
    --task                        x2t_image \
    --save_path_gen               results/... \
    --resolution                  video_480p \
    --text_template               true \
    --cfg_text_scale              4.0 \
    --use_KVcache                 true
```

For Phase 0 you'll vary `--task`, `--validation_data_seed`, and `--resolution` (and the *.json reference inputs) to capture each oracle case.

## `path_default.yaml` — where files live

Relative to wherever inference is run (typically the Lance repo root):

```
downloads/
├── Lance_3B/               (image variant; OPTIONAL — Video model handles image tasks too)
├── Lance_3B_Video/         (video variant — serves all 6 tasks per shipped script)
├── Wan2.2_VAE.pth          (2.82 GB pickle, 48-ch — Lance's bundled VAE)
└── Qwen2.5-VL-ViT/         (1.34 GB, understanding ViT)
```

## Dataclass field tables (with shell overrides)

Legend: **bold** in "Shell" column = override of the dataclass default.

### `ModelArguments`

| Field | Type | Dataclass default | Shell override | Notes |
|---|---|---|---|---|
| `model_path` | str | `""` | **`downloads/Lance_3B_Video`** | empty default — must be set |
| `llm_path` | str | `""` | (unset; embedded in `model_path`) | |
| `llm_qk_norm` | bool | `True` | `true` | matches |
| `llm_qk_norm_und` | bool | `True` | `true` | matches |
| `llm_qk_norm_gen` | bool | `True` | `true` | matches |
| `tie_word_embeddings` | bool | `False` | `false` | matches — the `llm_config.json` says `true` but the runtime overrides; load `lm_head.weight` independently |
| `layer_module` | str | `"Qwen2MoTDecoderLayer"` | (default) | the MoT layer class name upstream |
| `vit_path` | str | `""` | (unset; relative to model_path) | |
| `max_num_frames` | int | `25` | **`121`** | hard cap on output frame count |
| `max_latent_size` | int | `64` | `64` | matches — latent grid max 64×64 |
| `latent_patch_size` | `[int]` | `[1, 2, 2]` | **`[1, 1, 1]`** | **load-bearing** — sets `patch_latent_dim = pt*ph*pw*z_channels` → 48 in shell (vs 192 in default) |
| `vit_patch_size` | int | `14` | (default) | Qwen2.5-VL ViT patch size |
| `vit_patch_size_temporal` | int | `2` | (default) | |
| `vit_max_num_patch_per_side` | int | `70` | (default) | |
| `connector_act` | str | `"gelu_pytorch_tanh"` | (default) | ViT→LLM connector activation |
| `interpolate_pos` | bool | `False` | (default) | |
| `vit_select_layer` | int | `-2` | (default) | use penultimate ViT layer for features |
| `vit_rope` | bool | `False` | (default) | ViT itself doesn't use RoPE; only LLM does |
| `text_cond_dropout_prob` | float | `0.1` | (training) | inference-irrelevant |
| `vae_cond_dropout_prob` | float | `0.3` | (training) | inference-irrelevant |
| `vit_cond_dropout_prob` | float | `0.3` | (training) | inference-irrelevant |
| `vit_type` | str | `"qwen2_5_vl"` | **`qwen_2_5_vl_original`** | **DIFFERENT default vs override** — shell value (with underscores + `_original` suffix) indicates "use the original Qwen2.5-VL ViT impl rather than any Lance-modified variant". Watch this when wiring; the value the *code* dispatches on is the shell one. |
| `val_text_cond_dropout_prob` | float | `0` | (default) | |
| `val_vae_cond_dropout_prob` | float | `0` | (default) | |
| `val_vit_cond_dropout_prob` | float | `0` | (default) | |
| `cfg_text_scale` | float | `4.0` | `4.0` | matches |

### `TrainingArguments` (inherited by InferenceArguments)

Only inference-relevant fields shown; training-only fields elided.

| Field | Type | Dataclass default | Shell override | Notes |
|---|---|---|---|---|
| `apply_chat_template` | bool | `False` | `false` | matches |
| `apply_qwen_2_5_vl_pos_emb` | bool | `False` | **`true`** | enables Qwen2.5-VL multi-modal position embedding (3D mRoPE) |
| `vae_model_type` | str | `"seedance"` | **`wan`** | **important** — there are at least TWO VAE backends. Shell uses Wan2.2; the "seedance" default suggests a sibling Seedance-VAE variant exists internally |
| `visual_gen` | bool | `True` | `true` | matches |
| `visual_und` | bool | `True` | `true` | matches |
| `freeze_und` | bool | `False` | (default) | training-only — `.detach()` UND outputs |
| `copy_init_moe` | bool | `False` | **`true`** | populates GEN expert weights from UND at load time. Essential when loading from a checkpoint where only UND was initialized |
| `finetune_from_hf` | bool | `False` | (default) | |
| `use_flex` | bool | `False` | (default) | flex attention toggle |
| `global_seed` | int | `2025` | (default) | training |
| `timestep_shift` | float | `1.0` | (training default) | training schedule |
| `validation_data_seed` | int | `42` | `42` | matches — **THE seed** for Phase-0 reference captures |
| `validation_num_timesteps` | int | `30` | `30` | matches |
| `validation_timestep_shift` | float | `3.0` | **`3.5`** | shell uses 3.5 (training was 1.0, dataclass default 3.0) — **use 3.5 for parity with shipped script** |
| `validation_noise_seed` | int | `2025` | (default) | separate from data seed |
| `validation_video_saving_fps` | int | `12` | (default) | video output fps |
| `cfg_type` | int | `0` | `0` | matches — "remove text conditioning entirely" |
| `cfg_uncond_token_id` | int | `151643` | (default) | `<|endoftext|>`; only used for cfg_type=2 |
| `cfg_interval` | `[float]` | `[0.4, 1.0]` | (default) | CFG applied only on timesteps in this fraction range |
| `cfg_renorm_min` | float | `0` | (default) | |
| `cfg_renorm_type` | str | `"global"` | (default) | options: `global` / `channel` / `""` |
| `use_task_embedding` | bool | `False` | (default) | |
| `use_modality_embedding` | bool | `False` | (default) | |

### `InferenceArguments` (extends TrainingArguments)

| Field | Type | Dataclass default | Shell override | Notes |
|---|---|---|---|---|
| `save_path_gen` | str | `"tmp/results/inference/generation"` | timestamped under `results/...` | per shell |
| `video_height` | int | `480` | `768` | image_768res uses 768 |
| `video_width` | int | `480` | `768` | |
| `num_frames` | int | `50` | `50` | matches — capped at `max_num_frames=121` |
| `task` | str | `"t2v"` | **`x2t_image`** (shell default; change per case) | options: `t2v`/`t2i`/`image_edit`/`video_edit`/`x2t_image`/`x2t_video` |
| `resolution` | str | `"video_360p"` | **`video_480p`** | options: `image_256res`/`image_512res`/`image_768res`/`video_192p`/`video_360p`/`video_480p` |
| `text_template` | bool | `False` | **`true`** | system_prompt text template on/off |
| `max_duration` | float | `6.0` | (default) | seconds |
| `system_prompt_type` | str | `"SP0"` | (default) | options: `SP1`, `SP2`, … |
| `use_KVcache` | bool | `False` | **`true`** | KV cache during T2I/T2V — shipped on |

## Chat templates (`TemplateArguments`)

Two distinct templates depending on task:

**Understanding (x2t_image, x2t_video):**

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
   ...[ViT tokens injected here]...
Describe this image.<|im_end|>
<|im_start|>assistant
   ...[model autoregressively continues]...
```

`pad_token_template = "<|quad_end|>"` — used as the placeholder for ViT-token insertion at sequence assembly time.

**Generation (t2i):**

```
<|im_start|>system
Describe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
<|quad_start|>   ← VAE-token insertion anchor
<|im_end|>
<|im_start|>assistant
```

`pad_token_template_T2I = "<|quad_start|>"` — VAE-latent tokens replace this placeholder at sequence assembly. Critically: the **system prompt for T2I is a DIFFERENT prompt** that conditions the model to produce detailed visual descriptions (it's basically asking the LLM_UND to elaborate the prompt before LLM_GEN renders it).

## Example file format (`config/examples/`)

| File | Top-level | Per-entry shape |
|---|---|---|
| `t2i_example.json` | dict `{"<filename>.png": prompt}` | flat — filename keys, string prompts. ~11 prompts. |
| `t2v_example.json` | dict | non-strict JSON (Python 3.9 parser choked — has trailing-comma or JSON5 syntax). Use `json5` or hand-parse. |
| `x2t_image_example.json` | dict `{case_id: {...}}` | `interleave_array: [image_path, [question_intro, question, answer]]`, `element_dtype_array: ["image","text"]`, `istarget_in_interleave: [0, 1]` |
| `x2t_video_example.json` | dict | same shape as x2t_image but with video path |
| `image_edit_example.json` | dict | edit instruction + before/after refs |
| `video_edit_example.json` | dict | video-edit version |

For Phase 0 reference captures, use prompts from these files unmodified (or a representative subset) so that diffing against ByteDance's own published examples gives a known-good baseline.

## Surprises to flag explicitly (carry-forward to scaffold/handoff)

1. **`vae_model_type` defaults to `"seedance"`** in the dataclass, not `"wan"`. Lance therefore has internal support for at least two VAE backends. The MLX port only needs Wan2.2 (shell override), but the model loader must accept the `vae_model_type` string and dispatch — don't hardcode Wan-only.
2. **`latent_patch_size` defaults to `[1, 2, 2]`** but shell uses `[1, 1, 1]`. This changes `patch_latent_dim = pt*ph*pw*z_channels` from the default `1*2*2*48 = 192` to the shipped `1*1*1*48 = 48`. The 48-channel flow-head Linear output in our scaffold is correct for the shipped configuration; if anyone runs with default `latent_patch_size`, the dim balloons to 192 and the Linear must be re-sized.
3. **`vit_type` mismatch** — dataclass says `"qwen2_5_vl"`, shell passes `qwen_2_5_vl_original`. The code must accept both spellings or the shell value is the canonical one for inference. Worth a `--help` check or a code grep for `vit_type ==` dispatch sites during Phase 1b.
4. **`apply_qwen_2_5_vl_pos_emb` defaults to `False`** but shell enables it. The 3D mRoPE pathway is *not* on by default — flag it as required for inference.
5. **`copy_init_moe true`** is required to populate GEN expert weights from UND at load time. Without it, the GEN tower is randomly initialized and outputs garbage. This is a load-time operation, not an inference flag — it's what makes the duplicated `_moe_gen` siblings start at the same values as their UND counterparts.

## Operational notes for Phase 0

1. **Download only `Lance_3B_Video`** unless you specifically need to compare against `Lance_3B` outputs. The video model handles all 6 tasks. Saves ~24.7 GB of download.
2. **Use `--validation_data_seed 42`** for every capture — this is what the runbook's diff target is keyed off.
3. **Vary only `--task`, prompt JSON, and `--resolution`** between cases. Keep all other flags identical to the shell defaults above.
4. **For T2I cases:** `--resolution image_768res --task t2i --num_frames 1` (or omit `--num_frames`; it's unused for image tasks but the shell passes it anyway).
5. **For T2V cases:** `--resolution video_480p --task t2v --num_frames 50` (or 121 for max-length).
6. **For x2t cases:** load the question JSON, no `--num_frames`, no `--cfg_text_scale` effect (CFG is for generation only).
7. **`text_template true`** wraps the prompt in the system template; leave it on for parity with shipped behavior.

## Phase 0 capture command shortcuts

```bash
# T2I @ 768², seed 42 — case 1 from t2i_example.json (the "snowflakes girl")
bash inference_lance.sh   # after editing TASK_NAME=t2i, RESOLUTION=image_768res

# T2V @ 480p × 50 frames, seed 42
# edit TASK_NAME=t2v, RESOLUTION=video_480p, NUM_FRAMES=50

# x2t_image VQA — case 0001 (pie chart)
# edit TASK_NAME=x2t_image, RESOLUTION=video_480p (works for image-VQA too)
```

The shell expects you to edit it in place — there's no CLI override for `TASK_NAME` from outside. Either edit + run + revert, or copy the script to `phase0_t2i.sh` / `phase0_t2v.sh` / etc. with each variant baked in.

## Open question still

- The `vit_type` value `qwen_2_5_vl_original` (with underscores and `_original` suffix) — what is the alternative? Possibly a Lance-modified ViT variant exists for internal experimentation. Resolve with a `grep -rn "vit_type" modeling/` on the pod once it's running, or skim `modeling/vit/qwen2_5_vl_vit.py` for branch logic.
