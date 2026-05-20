"""Unit tests for lance_mlx.model.routing.

These tests are runnable today — they exercise the only fully-implemented
module in the model package and lock down its behavior before the full
backbone gets wired in.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from lance_mlx.model.routing import (
    POSITION_GROUP_TO_EXPERT,
    Expert,
    PositionGroup,
    expert_mask_from_position_group,
    merge_expert_outputs,
)


def test_position_group_to_expert_mapping():
    """Text and ViT-semantic route to UND; clean and noisy VAE route to GEN."""
    assert POSITION_GROUP_TO_EXPERT[PositionGroup.TEXT] == Expert.UND
    assert POSITION_GROUP_TO_EXPERT[PositionGroup.VIT_SEMANTIC] == Expert.UND
    assert POSITION_GROUP_TO_EXPERT[PositionGroup.CLEAN_VAE] == Expert.GEN
    assert POSITION_GROUP_TO_EXPERT[PositionGroup.NOISY_VAE] == Expert.GEN


def test_expert_mask_from_position_group_all_text():
    pg = mx.array([[0, 0, 0, 0]])  # all text
    mask = expert_mask_from_position_group(pg)
    assert mask.shape == (1, 4)
    assert mx.array_equal(mask, mx.zeros_like(mask))


def test_expert_mask_from_position_group_all_vae():
    pg = mx.array([[2, 2, 3, 3]])  # all VAE (clean + noisy)
    mask = expert_mask_from_position_group(pg)
    assert mx.array_equal(mask, mx.ones_like(mask))


def test_expert_mask_from_position_group_mixed():
    pg = mx.array([[0, 1, 2, 3]])  # text, ViT, clean VAE, noisy VAE
    mask = expert_mask_from_position_group(pg)
    expected = mx.array([[0, 0, 1, 1]])
    assert mx.array_equal(mask, expected)


def test_merge_expert_outputs_selects_correctly():
    # Two-token sequence: token 0 routes to UND, token 1 routes to GEN
    out_und = mx.array([[[1.0, 1.0], [1.0, 1.0]]])  # (1, 2, 2)
    out_gen = mx.array([[[2.0, 2.0], [2.0, 2.0]]])
    mask = mx.array([[0, 1]])  # (1, 2)
    merged = merge_expert_outputs(out_und, out_gen, mask)
    expected = mx.array([[[1.0, 1.0], [2.0, 2.0]]])
    assert mx.array_equal(merged, expected)


def test_merge_expert_outputs_shape_preservation():
    B, T, D = 2, 7, 16
    out_und = mx.random.normal((B, T, D))
    out_gen = mx.random.normal((B, T, D))
    mask = mx.random.randint(0, 2, (B, T))
    merged = merge_expert_outputs(out_und, out_gen, mask)
    assert merged.shape == (B, T, D)
