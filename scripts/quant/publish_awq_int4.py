#!/usr/bin/env python3
"""Publish Lance-3B-AWQ-INT4 to mlx-community.

DRY-RUN BY DEFAULT. Run once without --commit, inspect the staged
contents (especially README.md), then re-run with --commit to push.

Repo target: mlx-community/Lance-3B-AWQ-INT4

Honesty contract for the model card:
  - Explicitly scoped to VQA (x2t_image). The 4-prompt t2i sweep
    showed ~80% high-freq detail loss — naive AND AWQ quant both
    fail on Lance image generation at every tested bit-width.
  - Quantitative caveats: 4/6 VQA parity with bf16 on the diagnostic
    oracle; degrades on precision-required outputs (license plates,
    currency amounts, exact numbers).
  - Honest upside: 6-9× decode speedup on long-form VQA, 3.31 GB on
    disk fits in 8-16 GB Macs.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-AWQ-INT4"
DEFAULT_STAGE = REPO_ROOT / "outputs" / "phase5c3_publish_stage"
DEFAULT_REPO_ID = "mlx-community/Lance-3B-AWQ-INT4"
UPSTREAM = "bytedance-research/Lance"


README_MD = dedent("""\
    ---
    license: apache-2.0
    base_model: bytedance-research/Lance
    library_name: lance-mlx
    tags:
    - mlx
    - lance
    - vqa
    - image-understanding
    - quantized
    - awq
    - apple-silicon
    pipeline_tag: image-text-to-text
    ---

    # Lance-3B-AWQ-INT4

    > **Note:** "Lance" here refers to **ByteDance Intelligent Creation Lab's unified multimodal model** ([arXiv:2605.18678](https://arxiv.org/abs/2605.18678)), **not** [Lance/LanceDB](https://github.com/lancedb/lance) (the columnar data format).

    **MLX AWQ-INT4 quantization of [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) — calibrated for VQA / image-understanding use on Apple Silicon.**

    | Property | Value |
    |---|---|
    | Source | [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) — Lance_3B variant |
    | Quantization | AWQ INT4 (Reza2kn-style alpha-search + scale fusion), group_size=128 |
    | Avg bits/weight | 4.28 |
    | On-disk size (LLM only) | 3.31 GB (27% of bf16's 12.4 GB LLM) |
    | On-disk size (full repo incl. VAE + ViT) | ~5.7 GB |
    | License | Apache 2.0 |
    | MLX min RAM | ~6 GB (fits comfortably in 8–16 GB Macs) |

    ## ⚠️ Scope: VQA only — NOT for image generation

    **Use this model for:** image understanding / VQA / captioning (the `x2t_image` task family).

    **Do NOT use this model for:** text-to-image (`t2i`), image editing (`image_edit`), or video tasks. Naive AND calibrated quantization at every tested bit-width (4-bit, 8-bit, with and without AWQ, GEN-tower quantized or preserved at bf16) produce ~80% high-frequency detail loss on Lance image generation. For image generation, use the bf16 variant: [`mlx-community/Lance-3B-bf16`](https://huggingface.co/mlx-community/Lance-3B-bf16).

    ## Quality on the diagnostic VQA sweep

    Validated against 6 oracle cases (`tests/fixtures/results/x2t_image_sample_*` in the source repo). The relevant comparison is **AWQ-INT4 answer parity with the bf16 reference**, since bf16 is the calibration target.

    | Case | Question type | bf16 vs AWQ-INT4 parity |
    |---|---|---|
    | 1 | yes/no reasoning over a chart | ✓ identical |
    | 2 | percentage extraction (short numeric) | ✓ identical |
    | 3 | license plate extraction | ✗ AWQ garbles ("Bx62bfy" → "Byfky") |
    | 4 | currency amount (large number) | ✗ AWQ divergent ("1.8 million" → "198%") |
    | 5 | Colosseum description (open-ended) | ✓ semantically equivalent |
    | 6 | solar eclipse description (open-ended) | ~ marginal (same topic, different specifics) |

    **Honest summary: ~4/6 cases preserve bf16 behavior closely.** AWQ-INT4 is reliable for categorical and open-ended descriptive VQA, but **degrades on precision-required outputs**: alphanumeric extraction (license plates), exact numeric values (currency, percentages spanning units), and similar high-precision token-level reasoning. The 4-bit precision floor isn't enough to preserve fine token-level lexical relationships.

    For applications that need exact extraction of numbers / IDs / dates / proper names, use bf16. For descriptive VQA, AWQ-INT4 is a usable 4× memory + 6-9× speed win.

    ## Speed (M5 Max 128 GB, macOS 26.2, greedy decode)

    | Oracle case | Output type | bf16 latency | AWQ-INT4 latency | Speedup |
    |---|---|---|---|---|
    | 1 | "Yes" (1 token) | 0.6 s | 0.4 s | 1.5× |
    | 2 | "43" (2 tokens) | 0.6 s | 0.3 s | 2.0× |
    | 3 | License plate (short) | 1.1 s | 0.4 s | 2.8× |
    | 4 | Currency description (~30 tokens) | 6.4 s | 0.7 s | **9.1×** |
    | 5 | Colosseum description (~80 tokens) | 12.1 s | 1.4 s | **8.6×** |
    | 6 | Eclipse description (~70 tokens) | 8.6 s | 1.3 s | **6.6×** |
    | total | — | 29.4 s | 4.5 s | **6.5× wall-clock** |

    Long-form decoding sees the biggest speedup — exactly the user-visible case for descriptive VQA.

    ## Usage

    ```python
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    from PIL import Image

    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir="path/to/Lance-3B-AWQ-INT4",
        vit_safetensors="path/to/Lance-3B-AWQ-INT4/vit.safetensors",
    )
    image = Image.open("photo.jpg").convert("RGB")
    answer = pipe.generate(
        image, "What is in this image?", max_new_tokens=256,
    )
    print(answer)
    ```

    Install `lance-mlx` directly from the source repo (PyPI release pending —
    see [`xocialize/lance-mlx`](https://github.com/xocialize/lance-mlx) backlog):

    ```bash
    pip install git+https://github.com/xocialize/lance-mlx
    ```

    ## What got quantized

    - Quantization: MLX `nn.quantize` mode="affine", `bits=4`, `group_size=128`
    - Calibration: Reza2kn/lance-quant AWQ algorithm ported to MLX. Alpha-grid search ∈ [0, 1] per fusion group, scale fused into preceding RMSNorm
    - Calibration corpus: 4-prompt t2i sweep yielding 152,790 tokens of activation data per Linear (full t2i forward exercises both UND and GEN tower consumers via Lance's MoE routing)
    - Both UND and GEN towers quantized to INT4. Always-bf16 modules: `time_embedder.proj_in`, `time_embedder.proj_out`, `llm2vae`
    - Per-fusion-group alpha distribution: mean 0.37, median 0.35, range [0.25, 0.55]
    - **qk_norms preserved** (vs Reza2kn's PyTorch which drops them in their UND-only repackaging)

    Full methodology + experimental records in [`xocialize/lance-mlx`](https://github.com/xocialize/lance-mlx) under `notes/phase5n_diagnostics/phase5c3_awq_port/`.

    ## What didn't work (for the record)

    Phase 5c-2 (naive 8-bit UND-only) and Phase 5c-3e/g (AWQ-INT4 full, AWQ-INT8, AWQ-INT4-und) all produced ~80% high-frequency detail loss on Lance image generation. AWQ-INT4 was modestly better than naive 8-bit (3-15 pp HF improvement), but no quantization recipe tested closes the gap. The ~80% floor is structural, not algorithmic — see Phase 5c-3 writeups in the source repo.

    ## Attribution

    - Upstream weights: [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance) (Apache 2.0)
    - Wan2.2 VAE: Alibaba Wan-AI team (Apache 2.0)
    - Qwen2.5-VL ViT (vision encoder init): Alibaba Qwen team (Apache 2.0)
    - AWQ algorithm: [`Reza2kn/lance-quant`](https://github.com/Reza2kn/lance-quant) (alpha-search + scale fusion recipe ported to MLX)
    - MLX conversion + AWQ port: [`xocialize/lance-mlx`](https://github.com/xocialize/lance-mlx)
    - Substrate packages: [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm)

    ## Citation

    ```bibtex
    @article{fu2026lance,
      title={Lance: Unified Multimodal Modeling by Multi-Task Synergy},
      author={Fu, Fengyi and Huang, Mengqi and Wu, Shaojin and others},
      journal={arXiv preprint arXiv:2605.18678},
      year={2026}
    }
    ```
""")


NOTICE_TEMPLATE = dedent("""\
    Lance-3B-AWQ-INT4

    This product is a derivative of Lance (bytedance-research/Lance), originally
    created by ByteDance Intelligent Creation Lab and released under the Apache
    License 2.0.

    Components:

      Lance LLM weights (AWQ-INT4 quantized)
          Copyright (c) ByteDance Intelligent Creation Lab
          Licensed under the Apache License, Version 2.0
          Quantization via AWQ alpha-search algorithm
          (algorithm: github.com/Reza2kn/lance-quant — Apache 2.0)
          MLX port + AWQ implementation: xocialize/lance-mlx

      Wan2.2 3D causal VAE (bf16)
          Copyright (c) Alibaba Group / Wan-AI team
          Licensed under the Apache License, Version 2.0

      Qwen2.5-VL ViT (vision encoder, used as init, bf16)
          Copyright (c) Alibaba Group / Qwen team
          Licensed under the Apache License, Version 2.0

    MLX conversion and packaging by MVS Collective (https://github.com/xocialize/lance-mlx)
    Licensed under the Apache License, Version 2.0
""")


def stage(src: Path, stage_dir: Path) -> Path:
    """Copy all source files into a clean staging directory, append README/NOTICE/LICENSE."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    # Copy everything from src
    for entry in sorted(src.iterdir()):
        if entry.is_file():
            shutil.copy2(entry, stage_dir / entry.name)
            print(f"  copied  {entry.name}")

    # Overwrite any README with our VQA-scoped one
    (stage_dir / "README.md").write_text(README_MD)
    print(f"  wrote   README.md  ({len(README_MD)} bytes)")

    # NOTICE
    (stage_dir / "NOTICE").write_text(NOTICE_TEMPLATE)
    print(f"  wrote   NOTICE     ({len(NOTICE_TEMPLATE)} bytes)")

    # LICENSE
    license_src = REPO_ROOT / "LICENSE"
    if license_src.exists():
        shutil.copy2(license_src, stage_dir / "LICENSE")
        print(f"  copied  LICENSE")
    else:
        print(f"  WARNING: no LICENSE at {license_src}; HF repo will lack it")

    return stage_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--commit", action="store_true",
                    help="Actually push to HF (default: dry-run)")
    args = ap.parse_args()

    print(f"┏━━ Phase 5c-3 publish — {args.repo_id} ━━━━━━━━━━━━━━━")
    print(f"┃ src   : {args.src}")
    print(f"┃ stage : {args.stage}")
    print(f"┃ commit: {args.commit}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Staging ===")
    stage_dir = stage(args.src, args.stage)

    print(f"\n=== Final staged contents ===")
    total = 0
    for f in sorted(stage_dir.iterdir()):
        sz = f.stat().st_size
        total += sz
        if sz < 4096:
            print(f"  {f.name:<32s}  {sz:>10d} B")
        elif sz < 1024**2:
            print(f"  {f.name:<32s}  {sz/1024:>10.1f} KB")
        else:
            print(f"  {f.name:<32s}  {sz/1024**2:>10.1f} MB")
    print(f"  ─────────────────────────────────────────")
    print(f"  {'total':<32s}  {total/1024**3:>10.2f} GB")

    if not args.commit:
        print(f"\n=== DRY-RUN ===")
        print(f"  Stage inspected. Open {stage_dir / 'README.md'} and review.")
        print(f"  Re-run with --commit to push to https://huggingface.co/{args.repo_id}")
        return 0

    # Real push
    print(f"\n=== Pushing to {args.repo_id} ===")
    from huggingface_hub import create_repo, upload_folder, whoami
    user = whoami()
    print(f"  authenticated as: {user['name']}")
    t0 = time.perf_counter()
    create_repo(args.repo_id, repo_type="model", exist_ok=True)
    print(f"  create_repo: {time.perf_counter()-t0:.1f}s (exist_ok=True)")
    t0 = time.perf_counter()
    upload_folder(
        folder_path=str(stage_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Initial AWQ-INT4 conversion — VQA-scoped variant of bytedance-research/Lance",
    )
    print(f"  upload_folder: {time.perf_counter()-t0:.1f}s")
    print(f"\n  ✓ pushed: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
