"""Unit tests for lance_mlx.model.mape — verified re-anchoring behavior.

These tests lock in the *verified* upstream semantics: hardcoded re-anchoring
to 1000 (image-gen, modality 4) and 2000 (video-gen, modality 3), with other
modalities untouched.
"""

from __future__ import annotations

import mlx.core as mx

from lance_mlx.model.mape import (
    ANCHOR_IMAGE_GEN,
    ANCHOR_VIDEO_GEN,
    MODALITY_IMAGE_GEN,
    MODALITY_TEXT,
    MODALITY_VIDEO_GEN,
    shift_position_ids_mape,
)


def _make_position_grid(t_values, h_values, w_values):
    """Build a (1, 3, T) position grid from three same-length 1-D sequences."""
    return mx.array([[list(t_values), list(h_values), list(w_values)]])


def test_text_only_untouched():
    """Modality 0 (text) → no shift applied."""
    pos = _make_position_grid([0, 1, 2], [0, 0, 0], [0, 0, 0])
    mod = mx.array([MODALITY_TEXT] * 3)
    out = shift_position_ids_mape(pos, mod)
    assert mx.array_equal(out, pos)


def test_image_gen_reanchored_to_1000():
    """Modality 4 (image-gen) → first temporal position becomes ANCHOR_IMAGE_GEN."""
    pos = _make_position_grid([5, 6, 7], [3, 3, 3], [4, 4, 4])
    mod = mx.array([MODALITY_IMAGE_GEN] * 3)
    out = shift_position_ids_mape(pos, mod)
    # Temporal axis: first position re-anchored to 1000; spacing preserved.
    assert out[0, 0, 0].item() == ANCHOR_IMAGE_GEN
    assert out[0, 0, 1].item() == ANCHOR_IMAGE_GEN + 1
    assert out[0, 0, 2].item() == ANCHOR_IMAGE_GEN + 2
    # H/W untouched.
    assert mx.array_equal(out[:, 1, :], pos[:, 1, :])
    assert mx.array_equal(out[:, 2, :], pos[:, 2, :])


def test_video_gen_reanchored_to_2000():
    """Modality 3 (video-gen) → first temporal position becomes ANCHOR_VIDEO_GEN."""
    pos = _make_position_grid([10, 11, 12, 13], [0, 0, 0, 0], [0, 0, 0, 0])
    mod = mx.array([MODALITY_VIDEO_GEN] * 4)
    out = shift_position_ids_mape(pos, mod)
    assert out[0, 0, 0].item() == ANCHOR_VIDEO_GEN
    assert out[0, 0, 3].item() == ANCHOR_VIDEO_GEN + 3


def test_mixed_modalities_each_anchored_independently():
    """Text + image-gen in one sequence — text untouched, image-gen re-anchored."""
    # Layout: 2 text tokens at positions {0,1}, then 3 image-gen tokens at {2,3,4}.
    pos = _make_position_grid([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
    mod = mx.array([MODALITY_TEXT, MODALITY_TEXT,
                    MODALITY_IMAGE_GEN, MODALITY_IMAGE_GEN, MODALITY_IMAGE_GEN])
    out = shift_position_ids_mape(pos, mod)
    # Text positions unchanged.
    assert out[0, 0, 0].item() == 0
    assert out[0, 0, 1].item() == 1
    # Image-gen segment re-anchored: 2→1000, 3→1001, 4→1002.
    assert out[0, 0, 2].item() == ANCHOR_IMAGE_GEN
    assert out[0, 0, 3].item() == ANCHOR_IMAGE_GEN + 1
    assert out[0, 0, 4].item() == ANCHOR_IMAGE_GEN + 2
