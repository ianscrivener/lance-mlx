"""Timing and memory benchmarking utilities (mirrors ltx-mlx-eval pattern).

Every generation should be wrapped in Timer contexts and produce a RunRecord
appended to outputs/runs.jsonl. The log feeds README benchmark tables and
parity-check reports.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    """One generation event. Serialized as JSONL to outputs/runs.jsonl."""

    run_id: str
    model: str
    task: str  # t2i | t2v | image_edit | video_edit | x2t_image | x2t_video
    prompt: str
    seed: int
    resolution: tuple[int, int] | None  # (H, W) — None for x2t tasks
    frames: int | None  # None for image tasks
    steps: int | None  # None for x2t tasks (AR decode, not flow)
    cfg: float | None
    timings: dict[str, float]  # section -> seconds
    peak_rss_gb: float
    timestamp: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())


class Timer:
    """Context manager that measures wall-clock seconds.

    Example:
        with Timer("flow_denoise") as t:
            latents = denoise(...)
        print(t.elapsed)
    """

    def __init__(self, label: str = "timer"):
        self.label = label
        self.start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        assert self.start is not None
        self.elapsed = time.perf_counter() - self.start


def peak_memory_gb() -> float:
    """Return the process peak RSS in GB (true high-water mark).

    Uses `resource.getrusage(RUSAGE_SELF).ru_maxrss` — the maximum RSS the
    process has reached at any point — so it can be called once at the end of
    a run and still capture mid-denoise / VAE-decode spikes. The earlier
    psutil-based version returned *instantaneous* RSS, which under-reports
    the true peak after tensors are freed. ru_maxrss units differ by platform:
    bytes on macOS/BSD, kilobytes on Linux. RUSAGE_SELF excludes child
    processes (irrelevant — generation runs in-process).
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2  # darwin=bytes, linux=KB
    return maxrss / divisor


def log_run(
    run_id: str,
    model: str,
    task: str,
    prompt: str,
    seed: int,
    timings: dict[str, float],
    peak_rss_gb: float,
    resolution: tuple[int, int] | None = None,
    frames: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    extra: dict[str, Any] | None = None,
    log_path: Path | str = "outputs/runs.jsonl",
) -> RunRecord:
    """Append a structured run record to a JSONL log."""
    record = RunRecord(
        run_id=run_id,
        model=model,
        task=task,
        prompt=prompt,
        seed=seed,
        resolution=resolution,
        frames=frames,
        steps=steps,
        cfg=cfg,
        timings=timings,
        peak_rss_gb=peak_rss_gb,
        extra=extra or {},
    )
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
    return record


def load_runs(log_path: Path | str = "outputs/runs.jsonl") -> list[RunRecord]:
    """Load all run records from the JSONL log."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    records = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("resolution") is not None:
                data["resolution"] = tuple(data["resolution"])
            records.append(RunRecord(**data))
    return records
