"""Unit tests for the DPM-Solver++(2M) / Adams-Bashforth 2 scheduler.

Covers:
 - Backward-compat: first step is byte-identical to Euler (warm-up path)
 - Multi-step smoke: solver runs 12 steps without error, shape preserved
 - Invalid scheduler: generate() raises ValueError before touching any model
"""
from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from lance_mlx.scheduler.solvers import DPMSolverPlusPlus2M


# ── solver unit tests ──────────────────────────────────────────────────────────

def test_first_step_is_euler():
    """First step must be byte-identical to latents - velocity * dt (Euler)."""
    solver = DPMSolverPlusPlus2M()
    velocity = mx.array([1.0, 2.0, 3.0])
    latents = mx.array([10.0, 20.0, 30.0])
    dt = 0.1

    result = solver.step(velocity, latents, dt)
    expected = latents - velocity * dt
    mx.eval(result, expected)

    assert mx.allclose(result, expected).item(), (
        "First step must be Euler warm-up — no previous velocity available"
    )


def test_solver_runs_twelve_steps_shape_preserved():
    """Smoke: solver completes 12 steps without error; output shape matches input."""
    solver = DPMSolverPlusPlus2M()
    latents = mx.zeros((1, 2304, 48))
    dt = 1.0 / 12

    for _ in range(12):
        velocity = mx.ones_like(latents) * 0.5
        latents = solver.step(velocity, latents, dt)

    mx.eval(latents)
    assert latents.shape == (1, 2304, 48)


def test_solver_reset_clears_state():
    """reset() must restore first-step Euler behaviour."""
    solver = DPMSolverPlusPlus2M()
    v = mx.array([1.0])
    x = mx.array([5.0])

    solver.step(v, x, 0.1)   # primes _v_prev
    solver.reset()

    result = solver.step(v, x, 0.1)
    expected = x - v * 0.1
    mx.eval(result, expected)
    assert mx.allclose(result, expected).item()


# ── pipeline-level validation ──────────────────────────────────────────────────

def _make_pipeline():
    """Minimal TextToImagePipeline with mocked sub-components."""
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    return TextToImagePipeline(
        lance_model=MagicMock(),
        vae_decoder=MagicMock(),
        processor=MagicMock(),
        text_config=MagicMock(),
        image_pad_token_id=0,
        video_pad_token_id=1,
        vision_start_token_id=2,
        vision_end_token_id=3,
    )


def test_invalid_scheduler_raises():
    """scheduler='foo' must raise ValueError with 'Unknown scheduler' in the message."""
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="Unknown scheduler"):
        pipe.generate("a red apple", scheduler="foo")
