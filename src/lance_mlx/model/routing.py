"""Token modality routing for Lance's dual-expert MoT.

Lance uses NO learned router (no softmax gate, no top-k). Routing is
deterministic from segment metadata constructed once at the top of the model
forward, then threaded through every layer. The verified pattern (2026-05-19):

- Routing is carried by **two integer-index tensors**, `packed_und_token_indexes`
  and `packed_gen_token_indexes`, computed from segment boundaries and task
  type in `Lance.forward`. Inside each MoT layer, tokens are dispatched by
  scatter-style index assignment to the two expert paths, with no cross-expert
  blending. This matches `Qwen2MoTDecoderLayer.forward_train` upstream.

- A `position_group` label per token (text / ViT-semantic / clean-VAE /
  noisy-VAE) drives MaPE re-anchoring (see `mape.py`) AND is the source from
  which the und/gen indexes are derived (groups 0/1 → UND, groups 2/3 → GEN).

- No tokenizer special tokens drive routing. (Earlier scaffold TODOs about
  resolving `BOT/EOT/BOV/EOV` IDs are obsolete — verification confirmed Lance
  uses straight Qwen2.5-VL vocab and routes via segment metadata only.)

This module provides both representations:
- A boolean/int **mask** (`expert_mask_from_position_group`) for `mx.where`-style
  merge — useful when both expert outputs are computed for the same tokens
  (rare under strict routing; mostly for testing/comparison).
- Pair of **index tensors** (`build_index_tensors_from_position_group`) for the
  scatter-style write-back that the upstream forward actually performs.

Both are pure functions; no parameters.
"""

from __future__ import annotations

from enum import IntEnum

import mlx.core as mx


class Expert(IntEnum):
    UND = 0
    GEN = 1


class PositionGroup(IntEnum):
    TEXT = 0
    VIT_SEMANTIC = 1
    CLEAN_VAE = 2
    NOISY_VAE = 3


POSITION_GROUP_TO_EXPERT: dict[int, int] = {
    PositionGroup.TEXT: Expert.UND,
    PositionGroup.VIT_SEMANTIC: Expert.UND,
    PositionGroup.CLEAN_VAE: Expert.GEN,
    PositionGroup.NOISY_VAE: Expert.GEN,
}


def expert_mask_from_position_group(position_group: mx.array) -> mx.array:
    """Boolean/int mask: 0 = route to LLM_UND, 1 = route to LLM_GEN.

    Args:
        position_group: (B, T) int array with values in {0..3}.

    Returns:
        (B, T) int array in {0, 1}.

    Algebraic shortcut: PositionGroup ≥ CLEAN_VAE → GEN.
    """
    return (position_group >= PositionGroup.CLEAN_VAE).astype(mx.int32)


def build_index_tensors_from_position_group(
    position_group: mx.array,
) -> tuple[mx.array, mx.array]:
    """Return `(und_idx, gen_idx)` — the index tensors upstream uses to scatter
    into each expert's path.

    For the typical inference shape where `position_group` is (T,), the returned
    indexes are 1-D and can be used directly for `array[..., und_idx, :]`-style
    advanced indexing in MLX.

    For batched layouts where `position_group` is (B, T), call this per-batch
    or rely on the mask form above — upstream's distributed packed layout flattens
    over batch anyway.
    """
    mask = expert_mask_from_position_group(position_group)
    flat = mask.reshape(-1)
    all_idx = mx.arange(flat.shape[0])
    und_idx = all_idx[flat == 0]
    gen_idx = all_idx[flat == 1]
    return und_idx, gen_idx


def merge_expert_outputs(
    out_und: mx.array,
    out_gen: mx.array,
    expert_mask: mx.array,
) -> mx.array:
    """Boolean-merge two parallel expert outputs into one tensor.

    Less efficient than the scatter pattern upstream uses (because it requires
    computing BOTH expert outputs for every token), but useful for unit tests
    and for variants that want soft routing in the future. Production forward
    should prefer `build_index_tensors_from_position_group` + scatter.

    Args:
        out_und: (B, T, D) output from LLM_UND tower.
        out_gen: (B, T, D) output from LLM_GEN tower.
        expert_mask: (B, T) int in {0, 1}.

    Returns:
        (B, T, D) merged output.
    """
    sel = expert_mask[..., None].astype(out_und.dtype)
    return out_und * (1 - sel) + out_gen * sel
