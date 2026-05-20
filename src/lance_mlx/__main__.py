"""CLI entry point for lance-mlx.

Usage:
    lance-mlx generate --task {t2i,t2v,image_edit,video_edit,x2t_image,x2t_video} [...]

Subcommands defer the heavy lifting to pipeline modules in lance_mlx.pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="lance-mlx", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Run a generation or understanding task")
    g.add_argument("--task", required=True,
                   choices=["t2i", "t2v", "image_edit", "video_edit", "x2t_image", "x2t_video"])
    g.add_argument("--prompt", help="Text prompt (or question for x2t_*)")
    g.add_argument("--image", type=Path, help="Input image (i2v, image_edit, x2t_image)")
    g.add_argument("--video", type=Path, help="Input video (video_edit, x2t_video)")
    g.add_argument("--weights", required=True, help="MLX weights repo or local path")
    g.add_argument("--output", type=Path, default=Path("outputs"))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--steps", type=int, default=30)
    g.add_argument("--cfg", type=float, default=4.0)
    g.add_argument("--timestep-shift", type=float, default=3.5)
    g.add_argument("--resolution", type=int, default=768)
    g.add_argument("--frames", type=int, default=50)
    g.add_argument("--fps", type=int, default=12)

    args = parser.parse_args()

    # TODO(claude-code): wire to actual pipeline modules once Phase 2+ scaffolding lands.
    # For now this is a stub that prints the dispatch.
    print(f"[lance-mlx stub] task={args.task} weights={args.weights}")
    print(f"[lance-mlx stub] Phase 0/1 incomplete — pipeline modules are stubs.")
    print(f"[lance-mlx stub] See HANDOFF.md for the phased port plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
