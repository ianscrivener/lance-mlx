"""Learned positional embedding over the VAE latent grid.

Discovered in Phase 1a weight inspection (2026-05-20) — NOT in the original
2026-05-19 verified findings. The Lance safetensors contains a single
embedding table named `latent_pos_embed.pos_embed`:

    safetensors key                 shape          dtype
    -------------------------------------------------------
    latent_pos_embed.pos_embed      [4096, 2048]   F32

Shape decomposition: 4096 = `max_latent_size ** 2` = 64² with the shipped
`--max_latent_size 64`. 2048 = LLM hidden size. So the table encodes 4096
distinct 2D spatial positions in the latent grid, each with a hidden_size
embedding.

Where this sits in the model:

The LatentPosEmbed lookup is ADDED into the token-embedding stream at
clean-VAE and noisy-VAE positions (modality groups 2 and 3), alongside:
  - the per-token `vae2llm`-projected latent content,
  - the broadcast `TimestepEmbedder` contribution,
  - 3D RoPE (in self-attention) modulated by MaPE re-anchoring.

Note: this is a SEPARATE position signal from 3D RoPE. RoPE encodes
*sequence* position; latent_pos_embed encodes *spatial-grid* position
within the latent modality. They coexist.

For T2V the grid is still spatial-only — the upstream config flattens
temporal × spatial latent indices into the same 4096-entry table and
uses MaPE re-anchoring to disambiguate frames. (Confirm in Phase 1b
when wiring up the t2v pipeline.)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

DEFAULT_NUM_POSITIONS = 4096        # 64×64 spatial grid (image variant)
DEFAULT_HIDDEN_SIZE = 2048

# Empirical sizes captured from the converted checkpoints (Phase 1b):
#   Lance_3B (image):       4096   = 64 × 64        spatial only
#   Lance_3B_Video:       126976   = 64 × 64 × 31   spatial × temporal slots
# The table is a flat list of positions; the caller flattens 2D or 3D grid
# coordinates into a single index before lookup.


class LatentPosEmbed(nn.Module):
    """Learned `(num_positions, hidden_size)` positional embedding table.

    Loaded from `latent_pos_embed.pos_embed`. Indexed by flat latent-grid
    position; broadcast/added into the hidden-state stream at clean/noisy-VAE
    token positions.

    The number of positions is variant-specific (4096 for Lance_3B,
    126976 for Lance_3B_Video) so the constructor takes it as a parameter.
    On load from a converted safetensors, the parameter will be overwritten
    with the actual checkpoint tensor — the initial `num_positions` only
    affects the shape of the freshly-initialized buffer.
    """

    def __init__(self, num_positions: int = DEFAULT_NUM_POSITIONS,
                 hidden_size: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.num_positions = num_positions
        self.hidden_size = hidden_size
        # Raw mx.array parameter so the safetensors key is exactly
        # `latent_pos_embed.pos_embed` (no `.weight` suffix that nn.Embedding
        # would add). MLX tracks mx.array attributes as parameters.
        self.pos_embed = mx.zeros((num_positions, hidden_size))

    def __call__(self, positions: mx.array) -> mx.array:
        """
        Args:
            positions: (B, T_vae) int — flat indices into the latent grid.
                For images: row * 64 + col. For video: temporal * 4096 + row * 64 + col
                (caller computes the right flat layout).

        Returns:
            (B, T_vae, hidden_size) — positional embeddings to be ADDED into
            the hidden-state stream at VAE-token positions.
        """
        return self.pos_embed[positions]
