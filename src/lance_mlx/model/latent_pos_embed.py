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

DEFAULT_MAX_LATENT_SIZE = 64
DEFAULT_HIDDEN_SIZE = 2048


class LatentPosEmbed(nn.Module):
    """Learned `(max_latent_size² × hidden_size)` positional embedding table.

    Loaded from `latent_pos_embed.pos_embed`. Indexed by flat latent-grid
    position; broadcast/added into the hidden-state stream at clean/noisy-VAE
    token positions.
    """

    def __init__(self, max_latent_size: int = DEFAULT_MAX_LATENT_SIZE,
                 hidden_size: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.max_latent_size = max_latent_size
        self.hidden_size = hidden_size
        # Single nn.Embedding-equivalent. Using Embedding gets us free
        # indexing semantics; the saved tensor key will be `pos_embed.weight`
        # — DIFFERENT from the safetensors key `pos_embed`. Use a raw
        # parameter instead to match the safetensors key exactly.
        self.pos_embed = mx.zeros((max_latent_size * max_latent_size, hidden_size))

    def __call__(self, positions: mx.array) -> mx.array:
        """
        Args:
            positions: (B, T_vae) int — flat indices into the latent grid
                (row * max_latent_size + col for 2D positions).

        Returns:
            (B, T_vae, hidden_size) — positional embeddings to be ADDED into
            the hidden-state stream at VAE-token positions.
        """
        return self.pos_embed[positions]
