#!/usr/bin/env python3
"""Phase 0 — capture PyTorch reference outputs from the official Lance inference code.

This script is meant to be run ONCE on a rented cloud GPU (RunPod A100 ~$1.50/hr,
Lambda H100 ~$2/hr). It generates a frozen set of reference outputs that every
subsequent MLX phase compares against.

Expected runtime: ~4-8 hours on an A100 for the full set below.

After running, copy outputs/ back to your local machine into tests/fixtures/:

    rsync -av --progress runpod:~/lance-fixtures/ ./tests/fixtures/

Usage on cloud GPU:

    git clone https://github.com/bytedance/Lance
    cd Lance
    pip install -r requirements.txt
    huggingface-cli download bytedance-research/Lance --local-dir ./checkpoints/Lance
    huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./checkpoints/Qwen2.5-VL-3B
    python /path/to/this/script.py --lance-root . --output-dir ./fixtures

This script is NOT runnable on Apple Silicon (Lance's CUDA + flash-attn + triton deps).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Frozen reference set. Same prompts/seeds used in every subsequent MLX phase.
REFERENCE_CASES = [
    # T2I — three diverse prompts at fixed seed
    {"task": "t2i", "id": "p01_fox_grass", "seed": 42, "steps": 30, "cfg": 4.0,
     "resolution": 768, "prompt": "A red fox cautiously stepping through tall summer grass at golden hour, naturalistic, shallow depth of field."},
    {"task": "t2i", "id": "p02_chef", "seed": 42, "steps": 30, "cfg": 4.0,
     "resolution": 768, "prompt": "A chef in a white apron plating a dish in a bright modern kitchen, soft window light."},
    {"task": "t2i", "id": "p03_coastline", "seed": 42, "steps": 30, "cfg": 4.0,
     "resolution": 768, "prompt": "Aerial view of a rugged coastline at sunset, waves crashing against cliffs, cinematic."},

    # T2V — three prompts, 50 frames @ 480p
    {"task": "t2v", "id": "v01_horse", "seed": 42, "steps": 30, "cfg": 4.0,
     "resolution": 480, "frames": 50, "fps": 12,
     "prompt": "A horse galloping across an open meadow under a partly cloudy sky, side-tracking camera."},
    {"task": "t2v", "id": "v02_reef", "seed": 42, "steps": 30, "cfg": 4.0,
     "resolution": 480, "frames": 50, "fps": 12,
     "prompt": "Tropical reef teeming with schools of fish darting between coral, sunlight filtering through clear water."},
    {"task": "t2v", "id": "v03_alley", "seed": 42, "steps": 30, "cfg": 4.0,
     "resolution": 480, "frames": 50, "fps": 12,
     "prompt": "Rain-soaked neon alley in a cyberpunk city at night, reflections in puddles, steam rising from grates."},

    # Image edit
    {"task": "image_edit", "id": "e01_remove_person", "seed": 42, "steps": 30, "cfg": 4.0,
     "input_image": "demo/people_on_beach.png",
     "prompt": "Remove the person from the scene, keep everything else identical."},

    # Video edit
    {"task": "video_edit", "id": "ve01_color", "seed": 42, "steps": 30, "cfg": 4.0,
     "input_video": "demo/sample_clip.mp4",
     "prompt": "Convert the scene to nighttime with city lights reflecting in the puddles."},

    # x2t_image (VQA)
    {"task": "x2t_image", "id": "u01_colosseum",
     "input_image": "demo/colosseum.png",
     "prompt": "What is the structure shown in the image and what is its historical significance?"},
    {"task": "x2t_image", "id": "u02_chart",
     "input_image": "demo/pie_chart.png",
     "prompt": "Summarize the main findings shown in this chart in 3 bullet points."},

    # x2t_video
    {"task": "x2t_video", "id": "uv01_caption",
     "input_video": "demo/sample_clip.mp4",
     "prompt": "Describe what happens in this video in 2-3 sentences."},
]


def run_case(case: dict, lance_root: Path, output_dir: Path) -> None:
    """Invoke Lance's inference for one case.

    TODO(claude-code): wire this to the actual `inference_lance.py` CLI once
    you've read it. The Lance README shows `bash inference_lance.sh` as the
    entry point; this likely expands to a python call with a JSON config file.
    Build that JSON config on the fly here, write it to a temp file, then
    invoke `python inference_lance.py --config /tmp/case.json`.

    Save outputs as:
        output_dir/<task>/<id>/output.{png,mp4,json}
        output_dir/<task>/<id>/config.json   (full reproducible config)
    """
    case_dir = output_dir / case["task"] / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    # Save the case config alongside the output for full reproducibility
    (case_dir / "config.json").write_text(json.dumps(case, indent=2))

    print(f"  [{case['task']}/{case['id']}] {case.get('prompt', '')[:60]}")

    # TODO(claude-code): build and invoke the actual command. Example sketch:
    # config_for_lance = {
    #     "task": case["task"],
    #     "seed": case.get("seed"),
    #     "num_timesteps": case.get("steps", 30),
    #     "cfg_text_scale": case.get("cfg", 4.0),
    #     "resolution": case.get("resolution"),
    #     "num_frames": case.get("frames"),
    #     "prompt": case["prompt"],
    #     "input_image": case.get("input_image"),
    #     "input_video": case.get("input_video"),
    #     "output_path": str(case_dir / "output"),
    # }
    # tmp_cfg = case_dir / "lance_config.json"
    # tmp_cfg.write_text(json.dumps(config_for_lance, indent=2))
    # subprocess.run(
    #     [sys.executable, str(lance_root / "inference_lance.py"), "--config", str(tmp_cfg)],
    #     check=True,
    # )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-root", type=Path, required=True,
                        help="Path to cloned bytedance/Lance repo")
    parser.add_argument("--output-dir", type=Path, default=Path("./fixtures"))
    parser.add_argument("--tasks", default="t2i,t2v,image_edit,video_edit,x2t_image,x2t_video")
    args = parser.parse_args()

    requested = set(args.tasks.split(","))
    cases = [c for c in REFERENCE_CASES if c["task"] in requested]
    print(f"Running {len(cases)} reference cases → {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        run_case(case, args.lance_root, args.output_dir)

    print(f"\nDone. Sync back to local machine:")
    print(f"  rsync -av --progress <cloud-host>:{args.output_dir}/ ./tests/fixtures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
