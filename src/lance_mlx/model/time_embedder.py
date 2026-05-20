"""Timestep embedder for Lance's flow-matching path.

Verified pattern (`bytedance/Lance` `modeling/lance/modeling_utils.py`):
sinusoidal frequency embedding → 2-layer MLP → hidden_size-dim vector. The
output is ADDED INTO THE TOKEN-EMBEDDING STREAM (alongside word embeddings)
before the LLM forward pass — not consumed inside the flow head. That
propagation is what lets `flow_head.llm2vae` be a single Linear.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

DEFAULT_FREQUENCY_DIM = 256


def sinusoidal_timestep_embedding(t: mx.array, dim: int = DEFAULT_FREQUENCY_DIM,
                                  max_period: float = 10000.0) -> mx.array:
    """Standard sinusoidal embedding (same shape used by Wan2.2 / FLUX / SDXL).

    Args:
        t: (B,) timestep scalars in [0, 1] (Lance) or [0, 1000] depending on
           upstream convention — match the upstream multiplier when wiring.
        dim: frequency-embedding dimension (must be even).

    Returns:
        (B, dim) embedding.
    """
    half = dim // 2
    freqs = mx.exp(
        -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
    )
    args = t[:, None].astype(mx.float32) * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embed → SiLU MLP → hidden_size. Added into the token stream."""

    def __init__(self, hidden_size: int = 2048, freq_dim: int = DEFAULT_FREQUENCY_DIM):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden_size)
        self.proj_out = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def __call__(self, t: mx.array) -> mx.array:
        """
        Args:
            t: (B,) scalar timesteps.

        Returns:
            (B, hidden_size) timestep embedding ready to be ADDED to the
            token embedding stream (broadcast over the sequence length where
            the timestep token sits).
        """
        emb = sinusoidal_timestep_embedding(t, self.freq_dim)
        return self.proj_out(self.act(self.proj_in(emb)))
