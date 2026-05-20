"""lance-mlx — MLX port of Lance (ByteDance unified multimodal model).

Lance paper: arXiv:2605.18678 (Fu, Huang, Wu et al., 2026)
Upstream weights: bytedance-research/Lance on HuggingFace
"""

__version__ = "0.0.1"

from .bench import RunRecord, Timer, log_run, peak_memory_gb
from .io import save_frames, save_image, save_video

__all__ = [
    "RunRecord",
    "Timer",
    "log_run",
    "peak_memory_gb",
    "save_frames",
    "save_image",
    "save_video",
]
