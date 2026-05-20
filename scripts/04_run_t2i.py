#!/usr/bin/env python3
"""Phase 3 — run T2I against Phase 0 fixtures.

Validation gate:
    - FID (CLIP or DINOv2 features) vs PyTorch reference < 0.05
    - CLIPScore agreement within 0.005
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lance_mlx import Timer, log_run, peak_memory_gb, save_image

PHASE = "phase3"


def run_t2i(case: dict, weights_dir: Path, out_root: Path) -> None:
    """Run one T2I case.

    TODO(claude-code): wire to lance_mlx.pipeline.t2i once it lands.
        from lance_mlx.pipeline.t2i import T2IPipeline
        pipe = T2IPipeline.from_pretrained(weights_dir)
        image = pipe(prompt=case["prompt"], seed=case["seed"],
                     steps=case["steps"], cfg=case["cfg"],
                     resolution=case["resolution"], timestep_shift=3.5)
    """
    run_id = f"{PHASE}_{case['id']}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    with Timer("load") as t:
        pipe = None  # placeholder
    timings["load"] = t.elapsed

    with Timer("text_encode") as t:
        # text_tokens = pipe.encode_prompt(case["prompt"])
        pass
    timings["text_encode"] = t.elapsed

    with Timer("flow_denoise") as t:
        # latents = pipe.denoise(text_tokens, steps=case["steps"], cfg=case["cfg"], seed=case["seed"])
        latents = None
    timings["flow_denoise"] = t.elapsed

    with Timer("vae_decode") as t:
        # image = pipe.vae.decode(latents)
        image = None
    timings["vae_decode"] = t.elapsed

    peak = peak_memory_gb()
    if image is not None:
        save_image(image, out_dir / "image.png")

    log_run(
        run_id=run_id,
        model=str(weights_dir),
        task="t2i",
        prompt=case["prompt"],
        seed=case["seed"],
        resolution=(case["resolution"], case["resolution"]),
        steps=case["steps"],
        cfg=case["cfg"],
        timings=timings,
        peak_rss_gb=peak,
        extra={"phase": PHASE, "case_id": case["id"]},
    )

    print(f"  total: {sum(timings.values()):.1f}s  peak: {peak:.1f} GB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("tests/fixtures/t2i"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--prompts", type=Path, default=Path("prompts/t2i_eval.json"))
    args = parser.parse_args()

    if args.fixtures_dir.exists():
        cases = []
        for case_dir in sorted(args.fixtures_dir.iterdir()):
            cfg_path = case_dir / "config.json"
            if cfg_path.exists():
                cases.append(json.loads(cfg_path.read_text()))
    elif args.prompts.exists():
        cases = json.loads(args.prompts.read_text())["prompts"]
    else:
        print(f"ERROR: no fixtures or prompts found", file=sys.stderr)
        return 1

    out_root = args.out / PHASE
    out_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        print(f"\n=== {case['id']}: {case['prompt'][:60]} ===")
        run_t2i(case, args.weights, out_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
