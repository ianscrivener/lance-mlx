"""Flow-matching velocity prediction head — verified against upstream source.

Surprise from the 2026-05-19 verification pass: the upstream "flow head" is
literally **one `nn.Linear(hidden_size, 48)`** named `llm2vae`, not an MLP and
not a DiT block. No AdaLN, no cross-attention, no time-conditioning inside the
head itself. Timestep enters via a `TimestepEmbedder` whose output is ADDED
INTO THE EMBEDDING STREAM before the LLM (so the timestep token rides along
through every layer's hidden state) — see `time_embedder.py`. The denoising
loop is a plain Euler step:

    x_{t-Δt} = x_t - v_t · Δt          # v_t = llm2vae(h_t)

with linear flow-matching schedule and inference timestep-shift of 3.5
(training was 4.0). CFG is applied at the velocity level (text-guided minus
unconditional, scaled by `cfg_text_scale=4.0`); not inside the head.

Output channel count: 48 — equals `latent_patch_size[0]*[1]*[2]*z_channels`
with the shipped `--latent_patch_size 1 1 1 --max_latent_size 64` and
Wan2.2 VAE `z_channels=48`. NOT 16. (The 16/48 confusion was the Wan2.2
public-distribution footgun; Lance bundles its own correct 48-ch VAE.)

⚠ Empirical correction 2026-05-20 (Phase 1a weight inspection):
HANDOFF.md's "bias=False" claim was wrong. The actual safetensors contains
BOTH `llm2vae.weight` [48, 2048] AND `llm2vae.bias` [48]. The Linear must
be instantiated with `bias=True` for the converter to find a destination
for the bias tensor. See `notes/phase1a_keys.md` D1 for evidence.

Symmetric note: `vae2llm` (the input-side projection from 48-ch VAE latents
into the 2048-hidden LLM stream) lives in `vae_bridge.py` — also has a bias.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Verified Lance defaults from `inference_lance.sh`.
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_LATENT_CHANNELS = 48           # z_channels of bundled Wan2.2 VAE
DEFAULT_NUM_TIMESTEPS = 30             # `--validation_num_timesteps 30`
DEFAULT_TIMESTEP_SHIFT = 3.5           # `--validation_timestep_shift 3.5`
DEFAULT_CFG_TEXT_SCALE = 4.0           # `--cfg_text_scale 4.0`


class FlowHead(nn.Module):
    """One Linear from LLM_GEN hidden state to flow-matching velocity.

    Mirrors upstream `self.llm2vae = nn.Linear(hidden_size, patch_latent_dim)`
    where `patch_latent_dim = 48` under the shipped config.

    bias=True: empirically confirmed by Phase 1a key inspection — both
    `llm2vae.weight [48, 2048]` and `llm2vae.bias [48]` exist in the
    actual safetensors (despite the original handoff's bias=False claim).
    """

    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 latent_channels: int = DEFAULT_LATENT_CHANNELS):
        super().__init__()
        self.llm2vae = nn.Linear(hidden_size, latent_channels, bias=True)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Args:
            h: (B, T_noisy, hidden_size) — LLM hidden states at noisy-VAE positions.

        Returns:
            (B, T_noisy, latent_channels) velocity prediction.
        """
        return self.llm2vae(h)


def euler_step(x_t: mx.array, v_t: mx.array, dt: float) -> mx.array:
    """One Euler step of the flow-matching denoising ODE: x_{t-dt} = x_t - v_t * dt.

    Lance integrates from noise (t=1) to data (t=0). `dts` are pre-computed from
    the timestep schedule once per generation (see
    `inference_lance.py::validation_gen` for the canonical loop).
    """
    return x_t - v_t * dt


def timestep_schedule(num_steps: int = DEFAULT_NUM_TIMESTEPS,
                      shift: float = DEFAULT_TIMESTEP_SHIFT) -> mx.array:
    """Linear schedule with `shift` applied — matches upstream behavior.

    Concretely: t_i = (shift * τ_i) / (1 + (shift - 1) * τ_i)  for τ_i in
    [1, 1-1/N, ..., 1/N, 0].
    """
    raw = mx.linspace(1.0, 0.0, num_steps + 1)
    return (shift * raw) / (1.0 + (shift - 1.0) * raw)


# NOTE: timestep conditioning is NOT done inside this head. Lance's TimestepEmbedder
# (see `time_embedder.py`) produces a hidden_size-dim embedding from the scalar t,
# which is ADDED into the token-embedding stream BEFORE the LLM forward pass —
# so by the time the hidden state reaches `llm2vae`, the timestep information
# has propagated through all 36 transformer layers. This is the central trick
# that lets the flow head be a single Linear.
