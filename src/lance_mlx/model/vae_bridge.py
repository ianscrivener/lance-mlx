"""VAE → LLM input projection (`vae2llm`).

Discovered in Phase 1a weight inspection (2026-05-20) — NOT in the original
2026-05-19 verified findings. The Lance safetensors contains a Linear named
`vae2llm` that is the symmetric inverse of `flow_head.llm2vae`:

    vae2llm:  Linear(48,   2048, bias=True)   — input  side
    llm2vae:  Linear(2048,   48, bias=True)   — output side (lives in flow_head.py)

These two Linears are independent — they do NOT share weights — and together
they bracket the LLM forward pass. The data flow at GEN positions is:

    VAE-latent token (48-ch)
      → vae2llm    (this module)               → (B, T_vae, 2048) hidden token
      → token-embedding stream + TimestepEmbedder additive contribution
      → 36 × LanceMoTLayer (with MaPE + latent_pos_embed for positional info)
      → model.norm_moe_gen (per-expert final RMSNorm)
      → llm2vae    (FlowHead)                   → (B, T_vae, 48) velocity prediction
      → Euler step → next VAE latent

Empirical evidence from Phase 1a:

    safetensors key             shape       dtype
    -----------------------------------------------
    vae2llm.weight              [2048, 48]   F32
    vae2llm.bias                [2048]       F32

The shape `[2048, 48]` is (out_features, in_features) per the safetensors
convention used by Qwen2.5-VL / mlx-vlm. `bias=True` (confirmed empirically).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Match flow_head.py's default — Wan2.2 VAE bundled with Lance is 48-ch.
DEFAULT_LATENT_CHANNELS = 48
DEFAULT_HIDDEN_SIZE = 2048


class VAEInputProjection(nn.Module):
    """One Linear from VAE-latent channels to LLM hidden dim.

    Applied at clean-VAE and noisy-VAE token positions BEFORE the layer stack.
    Mirrors upstream `self.vae2llm = nn.Linear(48, hidden_size, bias=True)`.
    """

    def __init__(self, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 hidden_size: int = DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.vae2llm = nn.Linear(latent_channels, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T_vae, latent_channels) — VAE-latent token features at GEN positions.
                Lance feeds in BOTH clean-reference latents (for edit tasks) and
                noisy target latents (the actual denoising target) — the position_group
                metadata tells the layer stack which is which (groups 2 and 3).

        Returns:
            (B, T_vae, hidden_size) — ready to be slotted into the packed
            token-embedding stream at the GEN token positions.
        """
        return self.vae2llm(x)
