"""Image and video output utilities.

Conventions:
- Frames are (T, H, W, 3) uint8 RGB numpy arrays
- Images are (H, W, 3) uint8 RGB numpy arrays
- Output structure: outputs/<phase>/<run_id>/
    frames/         PNG sequence (for video)
    image.png       single image output
    video.mp4       muxed mp4 @ task-default fps
    inspection/     thumbnails (first/mid/last)
    meta.json       generation metadata
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


def save_image(image: np.ndarray, out_path: Path | str) -> Path:
    """Save a single (H, W, 3) uint8 RGB image as PNG."""
    assert image.dtype == np.uint8, f"expected uint8, got {image.dtype}"
    assert image.ndim == 3 and image.shape[-1] == 3, f"expected (H,W,3), got {image.shape}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)
    return out_path


def save_frames(frames: np.ndarray, out_dir: Path | str) -> Path:
    """Save a (T, H, W, 3) uint8 RGB array as PNG sequence."""
    assert frames.dtype == np.uint8
    assert frames.ndim == 4 and frames.shape[-1] == 3
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out_dir / f"{i:05d}.png")
    return out_dir


def save_inspection(frames: np.ndarray, out_dir: Path | str) -> Path:
    """Save first/mid/last frames as inspection thumbnails."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    for label, idx in [("first", 0), ("mid", n // 2), ("last", n - 1)]:
        Image.fromarray(frames[idx]).save(out_dir / f"{label}.png")
    return out_dir


def save_video(frames: np.ndarray, out_path: Path | str, fps: int = 12) -> Path:
    """Mux frames to MP4 via imageio-ffmpeg.

    Default fps=12 matches Lance's t2v output. Use fps=24 for cinema-rate output.
    """
    assert frames.dtype == np.uint8
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        str(out_path),
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    return out_path


def write_meta(meta: dict, out_path: Path | str) -> Path:
    """Write generation metadata as pretty-printed JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    return out_path
