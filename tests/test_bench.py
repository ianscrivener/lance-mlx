"""Unit tests for lance_mlx.bench."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lance_mlx.bench import RunRecord, Timer, load_runs, log_run, peak_memory_gb


def test_timer_measures_elapsed():
    with Timer("test") as t:
        time.sleep(0.01)
    assert t.elapsed >= 0.01
    assert t.elapsed < 1.0


def test_peak_memory_returns_positive():
    mem = peak_memory_gb()
    assert mem > 0
    assert mem < 1024  # sanity: less than 1 TB


def test_run_record_total_seconds():
    r = RunRecord(
        run_id="test",
        model="test-model",
        task="t2i",
        prompt="test",
        seed=42,
        resolution=(768, 768),
        frames=None,
        steps=30,
        cfg=4.0,
        timings={"a": 1.0, "b": 2.0, "c": 0.5},
        peak_rss_gb=10.0,
    )
    assert r.total_seconds == 3.5


def test_log_run_roundtrip(tmp_path: Path):
    log_path = tmp_path / "runs.jsonl"
    log_run(
        run_id="r1", model="m", task="t2i", prompt="hello", seed=42,
        timings={"section": 1.5}, peak_rss_gb=8.0,
        resolution=(768, 768), steps=30, cfg=4.0,
        log_path=log_path,
    )
    log_run(
        run_id="r2", model="m", task="x2t_image", prompt="world", seed=0,
        timings={"section": 0.5}, peak_rss_gb=12.0,
        log_path=log_path,
    )
    records = load_runs(log_path)
    assert len(records) == 2
    assert records[0].run_id == "r1"
    assert records[0].resolution == (768, 768)
    assert records[1].run_id == "r2"
    assert records[1].resolution is None
