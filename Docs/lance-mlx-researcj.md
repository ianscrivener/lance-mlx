# Lance (ByteDance) → MLX Port Viability Assessment

## TL;DR
- **Verdict: PORT — but with eyes open.** Lance is a real, Apache-2.0, self-contained 3B-active / ~12B-total dual-expert MoE that fuses a Qwen2.5-VL-3B understanding tower, a duplicated Qwen2.5-VL-3B generation tower, a Wan2.2 3D causal VAE, and a flow-matching head. Every component except the dual-expert routing already has a working MLX precedent in the community. On your M5 Max 128 GB, bf16 inference will fit comfortably (~30–40 GB working set including VAE & ViT), and quantized variants will be trivial. This is the most attractive port you've evaluated in this thread.
- **It would be a first.** No unified omni-modal *generation* model with image + video T2X + editing + understanding currently exists in `mlx-community` — Janus-Pro never got an official `mlx-vlm` port (only a third-party 4-bit LM-only fork by `wnma3mz`), BAGEL has no MLX port, Emu3.5 has none, Show-o2 has none. Lance at 3B active would slot in as the *flagship unified multimodal generator* on Apple Silicon.
- **The risk is scope, not feasibility.** Six task heads (t2i, t2v, image_edit, video_edit, x2t_image, x2t_video), three distinct numeric pipelines (autoregressive LM decode, flow-matching denoise over image latents, flow-matching denoise over video latents with temporal VAE), and two non-trivial sub-models to port (Wan2.2 VAE, custom MaPE-modified 3D-RoPE) put this at roughly a **3–5 week** end-to-end effort for the full feature surface vs. ~1–2 weeks for "just T2I + image understanding" as a first milestone.

---

## Key Findings

### Identity & provenance
- **Repo:** https://huggingface.co/bytedance-research/Lance (live, 57.4 GB total, 19 commits, 3 contributors, ~54 likes as of May 19 2026)
- **Code:** https://github.com/bytedance/Lance (Apache-2.0, Python 96.7% / shell 3.3%, ~50 stars, 18 commits, 2 forks — *very* new)
- **Paper:** arXiv:2605.18678 v1, "Lance: Unified Multimodal Modeling by Multi-Task Synergy," submitted **May 18 2026** (i.e., literally yesterday relative to today's date), 34 pages, 14 figures, 10 tables. PDF mirror: https://lance-project.github.io/assets/lance.pdf
- **Project page:** https://lance-project.github.io/
- **Authors:** Fengyi Fu, Mengqi Huang, Shaojin Wu, Yunsheng Jiang, Yufei Huo, Jianzhu Guo (project lead), Hao Li, Yinghang Song, Fei Ding, Qian He, Zheren Fu, Zhendong Mao, Yongdong Zhang — ByteDance **Intelligent Creation Lab** (note: *not* ByteDance-Seed, which is the org that ships BAGEL — this is a different internal team).
- **License:** **Apache-2.0** — confirmed on both the HF page and the `LICENSE` file in the GitHub repo. No carve-outs, no usage restrictions. This is critical context: ByteDance-Seed's recent releases (BAGEL, Seedance) have been Apache; this Intelligent Creation Lab release follows the same pattern. Commercial use OK.

### Architecture (resolved from paper + inference code, since the model card is silent on internals)
**Dual-stream Mixture-of-Transformer-Experts (MoT-style, BAGEL-lineage)** — NOT a per-layer FFN-experts MoE with a learned router.

- Two full transformer-decoder expert backbones share the same interleaved multimodal sequence and the same attention substrate, but tokens are **hard-routed by modality**:
  - `LLM_UND` processes text tokens + Qwen2.5-VL ViT semantic visual tokens → autoregressive next-token prediction via standard LM head.
  - `LLM_GEN` processes Wan2.2 VAE latent tokens → flow-matching velocity prediction via a flow head.
- Both experts are **initialized from Qwen2.5-VL-3B-Instruct** (literally a duplicated copy of the Qwen2.5-VL LLM tower) — the project's "trained from scratch" framing refers to the *unified multi-task objective*, not the weight init. The HF model card's `base_model: Qwen/Qwen2.5-VL-3B-Instruct` tag is the authoritative ground truth. Verbatim from Sec. 5.1 of the paper: *"Lance is implemented upon Qwen2.5-VL 3B, using its weights to initialize the visual understanding encoder and the multimodal context backbones LLM_UND and LLM_GEN."*
- Per-expert architecture inherits Qwen2.5-3B-LLM dimensions: 36 layers, hidden 2048, 16 attention heads, 2 KV heads (GQA), intermediate 11008, RMSNorm, RoPE. Per-expert separate **QK-Norm** modules (`qk_norm_und`, `qk_norm_gen` fields in `llm_config.json`).
- Attention is "**generalized 3D causal attention**": sequence partitioned into modality-specific segments; segments attend causally to earlier clean segments; *within* a segment text uses causal attention while visual tokens use bidirectional attention.

**Parameter accounting:**
- "3B active parameters" = exactly one expert (Qwen2.5-3B-LLM ≈ 3.09B) is active per token at the dual-expert layer.
- `model.safetensors` is **24.7 GB** in the `Lance_3B/` directory (plus 28.4 GB for the separate `Lance_3B_Video` model, plus a 2.82 GB `Wan2.2_VAE.pth` at the repo root, plus a 1.34 GB Qwen2.5-VL ViT in `Qwen2.5-VL-ViT/vit.safetensors`).
- 24.7 GB at bf16 ≈ **12.35B total params** on the LLM side — consistent with ~2× Qwen2.5-3B (both experts) + an LM head (untied from input embeddings per `untie_lm_head()` in the inference code), LLM-to-VAE / VAE-to-LLM MLP connectors, and a flow-prediction head.

### Encoders / decoders / heads
| Component | Source | Size | License | MLX status |
|---|---|---|---|---|
| Understanding ViT | Qwen2.5-VL-3B ViT (14× spatial patching + 2× temporal + 2×2 merge) | 1.34 GB safetensors | Apache | **Already in `mlx-vlm`** (Qwen2.5-VL family fully supported by Blaizzy/mlx-vlm 0.1.11+) |
| Generation VAE | Wan2.2 3D causal VAE (16× spatial × 4× temporal downsample) | 2.82 GB `.pth` (pickle) | Apache | **Already ported twice** in the wild — `osama-ata/Wan2.2-mlx`, `Armanoide/Wan2.2.mlx`, `kryptx/Wan2.2-mlx`, and notably **first-class in `Blaizzy/mlx-video`** which already runs Wan2.2 TI2V-5B end-to-end. DrawThings has also released a Swift implementation with 3-frame batched VAE decoding |
| LM head | Qwen2.5 standard, **untied from input embeddings** | included in 24.7 GB | Apache | Standard mlx-lm pattern, trivial |
| Flow head | Lightweight velocity-prediction head over VAE latents (paper does not specify whether MLP or DiT-block — code suggests a thin projection on top of `LLM_GEN`'s hidden states with sinusoidal latent positional embeddings) | included | Apache | Needs to be built from scratch, but small |
| MLP connectors | VAE→LLM and LLM→VAE projections | included | Apache | Trivial MLP |
| Tokenizer | Qwen2.5 BPE + Lance-specific special tokens (`BOT`, `EOT`, `BOV`, `EOV`, plus standard `<|vision_start|>` etc.) | 7.03 MB `tokenizer.json` | Apache | Use HF tokenizers directly |

### Modality-Aware Rotary Positional Encoding (MaPE)
A small but architecturally specific modification: standard Qwen2.5-VL 3D-RoPE coordinates `(t, h, w)` are augmented with a **constant temporal-axis offset Δ_m** per modality group `m ∈ {ViT-semantic, clean VAE, noisy VAE}`. Verbatim from Sec. 3.3: *"MaPE then applies a modality-specific offset Δ_m only along the temporal dimension: p^(m)_{t,h,w} = p̂^(m)_{t,h,w} + [Δ_m, 0, 0]"*. This separates the three visual token populations in positional space while preserving spatial layout and intra-group relative temporal order. In MLX this is a one-line modification to the existing `mlx-vlm` Qwen2.5-VL RoPE implementation.

### Inference pipeline (defaults, from `inference_lance.sh`)
- **Flow-matching schedule:** linear `x_t = t·x_1 + (1−t)·x_0`, velocity target `x_1 − x_0`. Inference timestep-shift = **3.5** (training used 4.0); 30 denoising steps default (50 OK).
- **CFG:** text-scale = 4.0. Code exposes `cfg_renorm_type`, `cfg_renorm_min`, `cfg_interval`, `cfg_type` — same family of CFG knobs as Wan2.2 / BAGEL.
- **Resolutions:** 768×768 for image, 480p @ 12 fps for video; max 121 frames (default 50).
- **Tasks:** `t2i`, `t2v`, `image_edit`, `video_edit`, `x2t_image`, `x2t_video` — unified through `inference_lance.sh`.
- **Hardware requirement (reference):** "**at least 40 GB VRAM** for inference" per the model card. Most of that is the Wan2.2 VAE decoding for video; the LLM at bf16 is ~25 GB and the ViT is 1.3 GB. On M5 Max 128 GB with unified memory, this is comfortable.

### Benchmarks (claimed in the paper / model card)
- **GenEval overall: 0.90** — ties TUNA-7B for SOTA among unified models, matches FLUX.1-dev (12B) and beats Janus-Pro-7B (0.80), Show-o2-7B (0.76), BAGEL-7B (0.88), OmniGen2-4B (0.80).
- **DPG-Bench overall: 84.67** — beaten by TUNA-7B (86.76) but Lance is less than half the size.
- **GEdit-Bench Avg/G_O: 7.30** — best among open unified models (vs. BAGEL 6.52, InternVL-U 6.66, Ovis-U1 6.42). Beats GPT Image 1's 7.49 only on individual axes, not overall, but for an open 3B that is impressive.
- **VBench (T2V) Total Score: 85.11** — best among unified video models (vs. Emu3 80.96, Show-o2 81.34, TUNA-1.5B 84.06), within striking distance of dedicated giants Wan2.1-T2V-14B (83.69) and Hunyuan Video (83.43). Note the † on Lance's row in the table indicates LLM-rewriter-aided prompts — worth replicating that flag during parity testing.

### Reality-check on the benchmarks
These are paper-authors' self-reported numbers on a model uploaded **three days ago** (per HF commit history showing `bfd93b2 verified 3 days ago` on the video safetensors). There is **no third-party reproduction yet**. ByteDance Intelligent Creation Lab benchmark numbers in unified-multimodal have historically been close to what other labs reproduce, but expect ±2-3 points of slop when you run the eval yourself. The MLX port's parity acceptance criteria should *not* be "match the paper's GenEval to 3 decimals" — it should be "match a fresh PyTorch run on identical prompts/seeds/CFG to within FID/CLIPScore tolerance."

---

## Details

### Lance vs. LTX-2.3 — Porting Perspective

| Dimension | LTX-2.3 (prior eval) | Lance |
|---|---|---|
| Architecture family | DiT diffusion + Gemma-3-12B text encoder | Dual-expert MoT over Qwen2.5-VL-3B init + Wan2.2 VAE |
| Active params | 22B DiT + 12B text encoder = ~34B | 3B active (one expert) at LLM layer, ~12B total LLM weight |
| Modalities | T2V/I2V/A2V (video + audio gen only) | T2I + T2V + I-edit + V-edit + I-VQA + V-VQA |
| Text encoder | External Gemma-3-12B (separate model) | **Self-contained** — Qwen2.5-VL is the model; no external encoder dependency |
| VAE | LTX video VAE (custom) + audio VAE | Wan2.2 3D causal VAE (already in mlx-video) |
| Existing MLX coverage of key pieces | None for LTX-2 (the dgrauet/ltx-2-mlx port is the *only* one) | **mlx-vlm has Qwen2.5-VL; mlx-video has Wan2.2 VAE** — ~70% of substrate already exists |
| Novel work required | Full DiT, two VAEs, cross-modal attention, audio vocoder | Dual-expert routing, MaPE, flow head, T2V/I2V conditioning glue |
| bf16 footprint | ~70 GB (DiT) + ~25 GB (text encoder) = ~95 GB | ~25 GB (LLM) + ~2 GB (VAE) + ~1.3 GB (ViT) = ~28 GB |
| M5 Max 128 GB fit | Tight (no headroom for activations on full bf16) | Roomy (3-4× headroom for activations, KV cache, video latents) |
| Quantization viability | Q8 essentially required for video gen | Q4/Q8 trivially viable; bf16 also fine |
| First-of-its-kind in mlx-community | Yes (video gen with audio) | **Yes (unified multimodal omni-generator)** |
| Estimated effort | 4–8 weeks for full feature parity | 3–5 weeks for full feature parity, 1–2 weeks for T2I-only MVP |

**Bottom line:** Lance is *materially easier* to port than LTX-2.3 because (a) two of its three sub-models (Qwen2.5-VL ViT, Wan2.2 VAE) are already in MLX, (b) it's self-contained — no external 12B text encoder dependency, (c) memory footprint is 3× smaller, and (d) the architectural novelty is concentrated in a few specific places (dual-expert routing, MaPE, flow head) that are mechanically simple even if conceptually new.

### Existing MLX Coverage Audit (relevant prior art)

What `mlx-community` and adjacent personal namespaces already cover:

- **Understanding side (✅ covered):** `mlx-vlm` (Blaizzy/Prince Canuma) has full Qwen2.5-VL support including the ViT, 3D-RoPE, and image+video understanding. The Lance understanding path is essentially "mlx-vlm Qwen2.5-VL with a slightly modified RoPE and a duplicated LLM tower used differently."
- **VAE side (✅ covered):** Three independent Wan2.2 MLX ports exist (`osama-ata/Wan2.2-mlx`, `Armanoide/Wan2.2.mlx`, `kryptx/Wan2.2-mlx`), plus first-class support in `Blaizzy/mlx-video`. DrawThings has a Swift `WanVAE.swift` with 3-frame batched decoding.
- **Flow matching (✅ covered):** `mlx-video` (Blaizzy) implements flow-matching diffusion with classifier-free guidance for Wan2.1, Wan2.2, and LTX-2. The Lance flow-matching loop is structurally identical to Wan2.2's.
- **Dual-expert MoT routing (❌ not covered):** No MLX implementation of BAGEL or BAGEL-style MoT routing exists. Lance would be the first.
- **Unified omni-modal generation (❌ not covered):** No MLX port of Janus-Pro (only `wnma3mz/Janus-Pro-7B-LM`, which is an LM-only fork via the third-party `tLLM` server, not in `mlx-community` proper), no BAGEL MLX, no Emu3/Emu3.5 MLX, no Show-o/Show-o2 MLX, no Chameleon MLX, no Lumina-mGPT MLX, no VILA-U MLX, no Anole MLX. **The gap is real and Lance is well-positioned to fill it.**
- **mlx-vlm's generation story:** mlx-vlm is *understanding-only*. The `mlx_vlm.generate` command does *text* generation conditioned on images, not image generation. There is no precedent in mlx-vlm for emitting visual latents or running a diffusion/flow head from a VLM. A Lance port would need to either (a) extend mlx-vlm with a generation pipeline (architectural change; Blaizzy collaboration likely needed), or (b) live as a standalone `lance-mlx` package that uses mlx-vlm's Qwen2.5-VL pieces as a dependency and adds its own generation loop on top, mirroring how `mlx-video` is separate from `mlx-vlm`.

**Natural collaborator:** **Prince Canuma (Blaizzy)** is the obvious owner. He maintains both `mlx-vlm` and `mlx-video`, and Lance literally sits at the intersection of those two packages. A standalone `mlx-omni` or `mlx-unified` package under his namespace (or under yours with his blessing) is the cleanest architectural answer. The model weights themselves go to `mlx-community/Lance-3B-bf16`, `mlx-community/Lance-3B-8bit`, `mlx-community/Lance-3B-4bit` per the standard `mlx-community` convention.

### Port Plan (Mirroring the LTX-2.3 Phase Structure)

#### Phase 0 — Verify (2–4 hours)
- Clone `bytedance/Lance`, download `bytedance-research/Lance` to local SSD (~60 GB total: 24.7 GB image LLM + 28.4 GB video LLM + 2.82 GB VAE + 1.34 GB ViT).
- Run `inference_lance.sh` end-to-end on a rented A100/H100 (RunPod, Lambda) to generate reference outputs for: 3× T2I prompts at seed 42, 3× T2V prompts at seed 42, 1× image edit, 1× x2t_image VQA. Save the exact prompts + seeds + CFG + step count + outputs. These are your **parity oracle**.
- Read `modeling/lance.py`, `modeling/qwen2.py`, `modeling/vae/wan/model.py`, `modeling/vit/qwen2_5_vl_vit.py` in the GitHub repo (the `modeling/` directory is the entire model source). Estimated ~3,000 LOC total based on similar repos.
- Read `llm_config.json` and `generation_config.json` directly to confirm exact hyperparameters (vocab size, expert layer count, etc.).

#### Phase 1 — Static graph + weight conversion (3–5 days)
- Fork `mlx-vlm`'s Qwen2.5-VL implementation as your starting point. Duplicate the transformer layer into two parallel `LLM_UND` / `LLM_GEN` towers sharing attention QKV but with separate FFN, output projections, and QK-Norm.
- Implement MaPE: a one-line `+ delta_m_offset` injection in the existing 3D-RoPE computation, with `delta_m_offset` looked up from the modality-group ID of each token.
- Implement the modality-router: a static, non-learned function `token_id_or_metadata → expert_id ∈ {UND, GEN}`. This is *not* a learned gate — it's deterministic from the token's segment metadata.
- Write `convert_lance.py` mapping HF safetensors keys → MLX module tree. The 24.7 GB image checkpoint has the `init_moe`-duplicated structure already baked in, so no on-the-fly duplication needed at conversion time.
- Port the Wan2.2 VAE encoder/decoder from one of the existing MLX Wan2.2 ports (Blaizzy's `mlx-video` is the most actively maintained). License is Apache, so direct vendoring is fine with attribution.
- **Acceptance:** weights load, forward pass on a dummy 768×768 latent produces a plausibly-shaped output (no shape errors), parity check pending until Phase 2.

#### Phase 2 — Understanding pipeline (x2t_image, x2t_video) (2–3 days)
- This is the easy half. Re-use mlx-vlm's autoregressive decode loop with the new dual-expert model where text+ViT tokens route to `LLM_UND` and there are no VAE tokens in the sequence.
- Build the multimodal chat template per `config/examples/x2t_image_example.json` and `x2t_video_example.json`.
- **Acceptance:** Match PyTorch reference VQA outputs token-for-token at greedy decode (do_sample=False) on the same 6 reference inputs as the model card examples (pie chart, license plate, Colosseum, market research chart, total solar eclipse, and one video VQA).

#### Phase 3 — Image generation (t2i, image_edit) (5–7 days)
- Build the flow-matching denoising loop: 30 steps, linear interpolant, timestep-shift 3.5, CFG-text-scale 4.0, optional CFG renorm.
- Route VAE-latent tokens to `LLM_GEN`, get velocity prediction back from the flow head, advance latents, repeat. Decode final latent with Wan2.2 VAE decoder to image.
- Implement image-edit conditioning (clean VAE latent of the input image + noisy target latent + edit-instruction text tokens, all in one interleaved sequence).
- **Acceptance:** Generate at seed 42 with CFG 4.0 and 30 steps from your three Phase 0 reference prompts; compare to PyTorch reference with FID (via CLIP features or DINOv2) < 0.05 and CLIPScore agreement within 0.005.

#### Phase 4 — Video generation (t2v, video_edit) (5–10 days)
- Extend the flow loop to handle the temporal dimension (3D causal VAE produces (T/4, H/16, W/16, 16) latents; flow head predicts velocity on the full 4D latent).
- Wan2.2 VAE decoding with 3-frame batched mode (per the DrawThings WanVAE.swift trick — keeps peak memory in check on 120-frame outputs).
- Handle the video_edit conditioning path (clean source-video VAE latents + noisy target + text).
- **Acceptance:** Generate 50-frame / 480p video at seed 42 from a Phase 0 reference prompt. Compare frame-by-frame to PyTorch reference; mean per-frame LPIPS < 0.02; VBench Total Score within 1.5 points of the paper's 85.11 on a small subsample (15-30 prompts, full VBench is overkill for parity check).

#### Phase 5 — Quantization + packaging (3–4 days)
- Run `mlx_lm.convert` style quantization at 4-bit, 6-bit, 8-bit, and ship bf16 reference. Per-expert quantization (UND and GEN towers can have different bit widths if quality demands it — this is a knob worth exploring since the GEN tower drives image fidelity).
- Upload to `mlx-community/Lance-3B-bf16`, `mlx-community/Lance-3B-8bit`, `mlx-community/Lance-3B-4bit`, `mlx-community/Lance-3B-Video-bf16`, etc. Ask Awni/Pedro for upload access if you don't already have it.
- Write a `lance-mlx` Python package with the same CLI surface as `inference_lance.sh`: `lance-mlx generate --task t2i --prompt "..."`, `--task t2v`, `--task image_edit`, etc.
- README, example notebooks, parity-test fixtures.

#### Phase 6 — Optional polish
- Swift port targeting macOS app integration (consistent with your RosettaCast / DubKit lineage). The dual-expert routing is the only Swift-side novelty; everything else is in `mlx-swift` already (Qwen2.5-VL via mlx-vlm-swift if/when it exists, VAE via SwiftDiffusion patterns from DrawThings).
- M5 Max Neural Accelerator profiling — per Apple Machine Learning Research's Nov 19 2025 blog "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU," the M5 NAs give "up to 4x speedup compared to a M4 baseline for time-to-first-token in language model inference" across Qwen 1.7B–30B, and *"generating a 1024x1024 image with FLUX-dev-4bit (12B parameters) with MLX is more than 3.8x faster on a M5 than it is on a M4."* So expect Lance T2I to land in the 30-60s/image range at 768×768 with bf16, possibly faster with int4. T2V at 50 frames / 480p is the gnarlier one — budget 4-8 minutes per clip is realistic before optimization.
- Distillation / few-step variants (Lance-Lightning) — out of scope for the initial port but worth flagging as a follow-up given the success of Wan2.2-Lightning.

**Realistic total effort estimate:** 3-5 weeks of focused work to ship all six tasks at parity with bf16 + Q8 + Q4 weights uploaded to mlx-community. **1-2 weeks for an MVP** that ships T2I + image understanding only (Phases 0-3, skipping video).

### Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Lance is too new (paper submitted yesterday, weights uploaded 3 days ago). Community has zero independent verification — benchmark claims could be cherry-picked, training instabilities could surface | Medium | Spend Phase 0 generating a wide variety of reference outputs (not just the cherry-picks on the project page). If T2V quality is clearly worse than VBench-85 implies on your own prompts, ship T2I-only first and revisit T2V after a community shakeout |
| Bug-bait code base — only 18 commits, 50 stars, 2 forks on GitHub as of today. Real-world usage is essentially zero so any issue you find may be unique to you | Medium | File issues upstream early; the corresponding authors (Mengqi Huang, Jianzhu Guo) have public contact info. Don't be the only person debugging |
| ByteDance pulls or relicenses the weights post-release (has happened with other ByteDance research releases, though not BAGEL or Seedance) | Low | Apache 2.0 is irrevocable — once you've downloaded the weights and republished MLX conversions on mlx-community with attribution, you're fine regardless of upstream actions. Mirror the weights to your own S3 / IPFS just in case |
| The Wan2.2 VAE has a documented version-mismatch footgun. The lilting.ch report (January 2026) documents the exact runtime error: *"weight of size [48, 48, 1, 1, 1], expected input[1, 16, 9, 60, 104] to have 48 channels, but got 16 channels instead"* — the wan2.2_vae.safetensors (1.41 GB) expects 48-channel input while T2V diffusion outputs 16-channel latents; the correct VAE for T2V remains wan_2.1_vae.safetensors (242 MB). Lance bundles a 2.82 GB `Wan2.2_VAE.pth` at the repo root — confirm channel count matches what `LLM_GEN`'s flow head outputs | Medium | First step of Phase 1: print the VAE state-dict shapes and the flow head's output projection dims. Channel mismatch = wrong VAE checkpoint. The reference inference code is the ground truth here |
| MaPE offset values (`Δ_m` per modality group) might be hard-coded constants vs. learned — paper isn't 100% explicit. If learned, they must be loaded from the checkpoint | Low | Trivially resolved by reading `llm_config.json` and the safetensors key list |
| The "untied LM head" detail (Lance untied the LM head from the input embeddings while Qwen2.5-3B-Instruct ties them) means you can't just symlink the embedding weights — you have to load lm_head separately | Low | Already noted; just don't forget |
| Lance might collide with future BAGEL-2 / Show-o3 / TUNA-3 releases that steal thunder before you ship | Medium | Move fast. The 3-day-old weights are a green-field opportunity; even a 1-week MVP for T2I alone secures the "first MLX port" narrative |
| Inference code uses `torch.distributed` / FSDP / NCCL primitives. The single-GPU inference path needs to work without distributed init | Low | The `inference_lance.py` already supports `NUM_GPUS=1`; just ensure your MLX port reads the same config without trying to call `dist.init_process_group` |
| Geopolitical/export-control: ByteDance is Chinese, the weights and code are Apache, the paper is on arXiv. No export control issue under current US BIS rules (Apache-licensed open weights are not subject to EAR §734.7-9 publicly-available exemptions) | None | Not relevant for an mlx-community port |
| The model card is informative on benchmarks and CLI usage but **silent on architectural internals** — no params count, no MoE expert count, no MaPE description. You have to read the paper (which luckily is public and detailed) | Low | Already done; resolved in this report |
| Quality of the model: the "3B-active" framing might oversell how much real capability is in `LLM_GEN`. If `LLM_GEN`'s output is visibly worse than dedicated T2I models like FLUX.1-dev or Qwen-Image, the mlx-community contribution is "interesting research artifact" rather than "production tool" | Medium | The GenEval 0.90 + DPG 84.67 numbers suggest it's at least credible. Run your own apples-to-apples comparison vs. FLUX.1-dev-mlx (if/when that exists) and Qwen-Image-mlx (also not yet ported) before declaring victory |

### Alternatives If Lance Turns Out to Be a No-Go

Ranked by mlx-community value × portability:

1. **BAGEL-7B-MoT (ByteDance-Seed)** — Apache 2.0. Per the official `ByteDance-Seed/BAGEL-7B-MoT` HF model card: *"It is finetuned from Qwen2.5-7B-Instruct and siglip-so400m-14-384-flash-attn2 model, and uses the FLUX.1-schnell VAE model, all under Apache 2.0."* 7B active / 14B total. The closest architectural analog to Lance (MoT, dual encoders) but **image-only** (no video). Strong benchmarks. **No MLX port exists.** If Lance's video pipeline turns out to be a tarpit, BAGEL is the safe fallback — fewer modalities, more mature, similar dual-expert architecture. The DFloat11/BAGEL-7B-MoT-DF11 lossless-compression fork (titled "70% Size, 100% Accuracy" — meaning the model is reduced *to* 70% of its original size, i.e., **20.2 GB down from ~29 GB bf16, bit-identical outputs**) would be a clean target for MLX 4-bit/8-bit quantization.

2. **Janus-Pro-7B (DeepSeek)** — MIT code license, DeepSeek Model License for weights (permits commercial). Decoupled SigLIP + VQ tokenizer + DeepSeek-LLM-7B-base. Image-only, no video. **Already half-ported** via `wnma3mz/Janus-Pro-7B-LM` (LM-only via `tLLM`) but not in `mlx-community`. A clean, full mlx-vlm-style port would still be a strong contribution. Lower scope than Lance.

3. **OmniGen2 (Beijing AI Academy)** — 4B params, image gen + edit + understanding. ViT + VAE dual-encoder, hidden-state-conditioned diffusion decoder. Less ambitious than Lance (no video) but more mature; could be a faster ship.

4. **Emu3.5 (BAAI)** — 34B, fully autoregressive (no flow matching), interleaved image-text + DiDA discrete diffusion adaptation for 20× inference speedup. Too big for comfortable MLX deployment on 128 GB without aggressive quantization, and the DiDA path is novel enough to be a separate research project. Skip unless you want a multi-month engagement.

5. **Show-o2 (NUS)** — 7B, autoregressive + flow matching, image + video. Strong on GenEval (0.76) but worse than Lance (0.90); no clear advantage over Lance for the mlx-community use case.

6. **HunyuanImage 3.0 / Qwen-Image / FLUX.1-dev / Wan2.2-T2V-A14B** — these are dedicated specialist generators, not unified models. Some already have MLX ports (Wan2.2 in mlx-video, FLUX.1-dev partially). Not "alternatives to Lance" in the unified-multimodal sense, but worth knowing they exist as parity targets for the GEN-only path quality comparison.

7. **TUNA-2 (paper-referenced, 7B)** — appears in the Lance benchmark tables; if/when ByteDance Intelligent Creation Lab releases TUNA-2 weights, it's a direct sibling. As of today's search, no public release.

**Recommendation among alternatives:** If Lance proves unworkable for any reason, **BAGEL-7B-MoT is the next-best mlx-community target.** Same architectural family (MoT), same lab lineage (ByteDance), same Apache 2.0 license, image-only scope is more tractable, and the architectural learnings transfer ~100% if Lance v2 ships later.

---

## Recommendations

**Decision: PORT Lance, in stages, starting with T2I MVP.**

### Immediate (this week)
1. Spend 4-6 hours reproducing Lance's reference outputs on a rented A100 to confirm the model actually works as advertised, and capture seeds/prompts/CFG/outputs as your parity oracle. **This is the single highest-value pre-port investment.**
2. Read `modeling/lance.py` and `modeling/qwen2.py` end-to-end. Confirm the architectural facts in this report (especially: expert count, FFN routing pattern, flow head structure).
3. Reach out to Prince Canuma (Blaizzy) to gauge interest in collaboration / integration with `mlx-video` or a new `mlx-omni` package. Even if he declines active collaboration, his sign-off on the package structure prevents future friction.
4. Reach out to Mengqi Huang / Jianzhu Guo on the paper's contact emails; mention the MLX port intent. ByteDance research teams have generally been responsive to community port projects and may provide informal architectural clarifications.

### Phase gate 1 (after Week 1)
**Ship `mlx-community/Lance-3B-bf16` + `lance-mlx` Python package with t2i + x2t_image only.** This is the MVP. Benefits:
- Establishes the "first unified omni-modal generator on Apple Silicon" narrative immediately.
- De-risks the dual-expert routing + MaPE + flow head implementation.
- Provides a base that the video path bolts on cleanly later.

### Phase gate 2 (after Week 3)
**Add t2v, image_edit, video_edit, x2t_video.** Full feature parity. By this point you'll know whether Lance's video quality is worth shipping or whether it's better to keep the model as "best open unified image generator on MLX, with experimental video support."

### Phase gate 3 (after Week 4-5)
**Ship 4-bit and 8-bit quantizations, write a launch blog post, submit a PR to add Lance to `Blaizzy/mlx-vlm` or `Blaizzy/mlx-video` (depending on where Prince Canuma's preference lands).** Cross-post to r/LocalLLaMA, X, HN.

### Benchmarks that would change the recommendation
- **If reference T2V at seed 42 looks visibly worse than VBench-85 implies on your own prompts (motion artifacts, blur, prompt-following failures):** Drop Phase 4, ship as image-only. Lance becomes "unified image model with bonus VQA" rather than "true omni generator."
- **If `LLM_GEN` quality at bf16 is materially below FLUX.1-dev / Qwen-Image:** Reposition the mlx-community release as "research artifact / unified model substrate" rather than "production T2I tool" — set expectations honestly.
- **If a competing port appears (Janus-Pro-mlx, BAGEL-mlx, Show-o2-mlx all show up in `mlx-community` in the next 1-2 weeks):** Lance still differentiates on video, so the port retains value. But the "first unified" narrative weakens — adjust marketing accordingly.
- **If ByteDance releases Lance-7B or Lance-14B in the next month:** Defer; port the larger model instead. Watch the `bytedance-research/Lance` HF repo's commit history and the `lance-project.github.io` for signals.

---

## Caveats

- All Lance benchmark numbers are author-reported on a 3-day-old release with no independent reproduction. Treat as plausible but not yet verified.
- Today is **May 19, 2026**, and the Lance paper was submitted **May 18, 2026**. This is the bleeding edge — community signals (Reddit, HN, X) about Lance are essentially zero. A search for "Lance ByteDance" on r/LocalLLaMA returns the GitHub README only, no community discussion yet. You will be among the first to seriously evaluate it for production-ish use.
- The architectural details (exact expert layer count, vocab size, flow head module type) in this report were extracted from the paper text and the inference code in the public GitHub repo. The HF model card itself is silent on internals. The `llm_config.json` file on HF (1.37 kB) would resolve any remaining ambiguity in a single curl but was not directly fetchable via this research run; pull it as Phase 0 task #2.
- The "3B active parameters" framing is technically true (one expert tower is Qwen2.5-3B-LLM ≈ 3.09B) but obscures that the full model is ~12-14B at bf16. This is consistent with how BAGEL markets its "7B active / 14B total." Set expectations accordingly when communicating to users.
- M5 Max Neural Accelerator support requires macOS 26.2+. Per Apple Machine Learning Research's Nov 19 2025 blog post (footnote [1]): *"To take advantage of the Neural Accelerators enhanced performance of the M5, MLX requires macOS 26.2 or later."* Confirmed independently by AppleInsider (Nov 18, 2025). If you're on an earlier macOS, M5 NA benefits will not be realized, though MLX itself works on all Apple Silicon.
- DrawThings has not yet announced Lance support but they are the most likely Swift-side ecosystem to add it given their existing Wan2.2 / WanVAE work; worth monitoring as parallel ecosystem signal.
- The phrase "trained from scratch" in the model card and abstract is misleading. Lance's LLM weights are initialized from Qwen2.5-VL-3B-Instruct (per Sec. 5.1 of the paper, verbatim), and the ViT is also from Qwen2.5-VL-3B, and the VAE is from Wan2.2. Only the dual-expert routing, MaPE, connectors, flow head, and new special-token embeddings are randomly initialized. "Trained from scratch" refers to the unified multi-task *objective*, not the parameter init. This matters for derivative-work attribution and licensing reasoning (Apache + Apache + Apache = Apache, all clean).