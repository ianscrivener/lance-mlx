#!/usr/bin/env python3
"""Phase 5f — remap RockTalk's Lance-3B-Video-MLX weights → our LanceModel layout.

RockTalk's safetensors keep the upstream Lance keying conventions (kept
`language_model.` prefix, used `time_embedder.fc1/fc2` instead of our
`proj_in/proj_out`, and `vae2llm.{weight,bias}` instead of our
`vae_in_proj.vae2llm.*`). They also store F32 rather than bf16. Same
underlying numerical values (both ports derive from the same PyTorch
upstream).

This script reads RockTalk's safetensors and writes a new file with our
key layout + bf16 dtype, plus a matching config.json that's compatible
with our `load_lance_model()` loader.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/22_remap_rocktalk.py \\
        --src-dir   /Volumes/.../rocktalk-weights \\
        --out-dir   /Volumes/.../lance-mlx-models/Lance-3B-Video-bf16-RT \\
        [--include-vit] [--include-vae]

The output is drop-in compatible with our pipelines:
    HF_HUB_DISABLE_XET=1 uv run python scripts/10_t2v_demo.py \\
        --lance-weights /Volumes/.../Lance-3B-Video-bf16-RT \\
        --vae-weights   /Volumes/.../Lance-3B-Video-bf16-RT/vae.safetensors \\
        ...

If our pipeline + RockTalk's weights produces SHARP output → conversion bug
(our converter loses precision somewhere). If still WATERCOLOR → pipeline
bug (their and our converted weights are equivalent; the deviation is in
our forward-pass code).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx


def remap_key(k: str) -> str:
    """Translate a RockTalk safetensors key into our LanceModel layout."""
    # Drop the language_model.model. prefix on layers/embed
    if k.startswith("language_model.model."):
        k = k[len("language_model.model."):]
    elif k.startswith("language_model."):
        # lm_head lives directly under language_model.
        k = k[len("language_model."):]

    # TimestepEmbedder: fc1 → proj_in, fc2 → proj_out
    if k.startswith("time_embedder.fc1"):
        k = "time_embedder.proj_in" + k[len("time_embedder.fc1"):]
    elif k.startswith("time_embedder.fc2"):
        k = "time_embedder.proj_out" + k[len("time_embedder.fc2"):]

    # VAE input projection: vae2llm.* → vae_in_proj.vae2llm.*
    # (ours nests in a holder module)
    if k.startswith("vae2llm."):
        k = "vae_in_proj." + k

    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, required=True,
                    help="RockTalk weights dir (contains model.safetensors, vit.safetensors, vae.safetensors)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dir in our layout")
    ap.add_argument("--include-vit", action="store_true",
                    help="Copy vit.safetensors as-is (key layout already matches mlx-vlm).")
    ap.add_argument("--include-vae", action="store_true",
                    help="Copy vae.safetensors as-is (key layout already matches mlx-video).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Remap model.safetensors ----------------------------------------
    print(f"┏━━ RockTalk → our layout remap ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ src: {args.src_dir}")
    print(f"┃ out: {args.out_dir}")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Loading RockTalk model.safetensors ===")
    t0 = time.perf_counter()
    src = mx.load(str(args.src_dir / "model.safetensors"))
    print(f"  loaded {len(src)} tensors in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== Remapping keys + casting F32 → bf16 ===")
    out: dict[str, mx.array] = {}
    collisions = 0
    skipped_vit = 0
    for k, v in src.items():
        # RT bundles ViT inside model.safetensors AS WELL AS in vit.safetensors.
        # Our LanceModel only loads the LLM block; ViT lives in a separate
        # mlx-vlm submodule loaded by understanding.py. Skip ViT keys here so
        # load_lance_model(..., strict=True) doesn't error on unused tensors.
        if k.startswith("vit_model.") or k.startswith("vision_model."):
            skipped_vit += 1
            continue
        new_k = remap_key(k)
        if new_k in out:
            print(f"  WARNING: key collision {new_k}")
            collisions += 1
        # Cast F32 → bf16 (lance_llm internally upcasts norms; matches our converter)
        if v.dtype == mx.float32 and ("norm" not in new_k):
            out[new_k] = v.astype(mx.bfloat16)
        else:
            out[new_k] = v
    print(f"  remapped {len(out)} tensors ({collisions} collisions, skipped {skipped_vit} ViT keys)")

    print(f"\n=== Writing remapped model.safetensors ===")
    t0 = time.perf_counter()
    mx.save_safetensors(str(args.out_dir / "model.safetensors"), out)
    print(f"  wrote {(args.out_dir / 'model.safetensors').stat().st_size / 1e9:.2f} GB "
          f"in {time.perf_counter()-t0:.1f}s")

    # --- 2. Write our config.json (compatible with build_text_config) ------
    print(f"\n=== Writing config.json (our schema) ===")
    rt_cfg = json.loads((args.src_dir / "config.json").read_text())
    qwen_cfg = rt_cfg["qwen2_5_vl_config"]
    # Force tie_word_embeddings=False since we store both lm_head and embed_tokens
    # (matches our converter's behavior; RockTalk's config says tie=true but
    # they also store both weights separately).
    qwen_cfg["tie_word_embeddings"] = False
    (args.out_dir / "config.json").write_text(json.dumps(qwen_cfg, indent=2))
    print(f"  wrote {args.out_dir / 'config.json'}")

    # --- 3. Optional: copy ViT + VAE + tokenizer ---------------------------
    aux_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "generation_config.json"]
    for fn in aux_files:
        src_f = args.src_dir / fn
        if src_f.exists():
            shutil.copy(src_f, args.out_dir / fn)
            print(f"  copied {fn}")

    if args.include_vae:
        src_vae = args.src_dir / "vae.safetensors"
        if src_vae.exists():
            shutil.copy(src_vae, args.out_dir / "vae.safetensors")
            print(f"  copied vae.safetensors ({src_vae.stat().st_size / 1e9:.2f} GB)")

    if args.include_vit:
        src_vit = args.src_dir / "vit.safetensors"
        if src_vit.exists():
            shutil.copy(src_vit, args.out_dir / "vit.safetensors")
            print(f"  copied vit.safetensors ({src_vit.stat().st_size / 1e9:.2f} GB)")

    # --- 4. Write a provenance report --------------------------------------
    report = {
        "source": "RockTalk/Lance-3B-Video-MLX",
        "source_dir": str(args.src_dir),
        "remap_rules": [
            "drop language_model.{model.}? prefix",
            "time_embedder.fc1 → time_embedder.proj_in",
            "time_embedder.fc2 → time_embedder.proj_out",
            "vae2llm → vae_in_proj.vae2llm",
            "F32 (non-norm) → bf16",
        ],
        "n_tensors_in": len(src),
        "n_tensors_out": len(out),
        "n_vit_skipped": skipped_vit,
    }
    (args.out_dir / "remap_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n✓ Done. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
