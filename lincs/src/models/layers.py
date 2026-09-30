"""Pieces shared by the two LINCS denoisers (`mlp.py`, `dit1d.py`).

Copied from RxRx19a/src/models/dit.py (IMPLEMENT.md §3.4; always copy, §3.1.1):
`TimestepEmbedder` verbatim, the token `modulate`, the output dataclass, and the
timestep / CFG-drop handling of `DiT2DModel.forward` (L416-438) and the
constructor guards of `DiT2DModel.__init__`. Both backbones take

    forward(sample, timestep, cond, drop=None, return_dict=True, terms_out=None)

with `sample` a (B, G) gene vector (never an image). New here:

  * `vec_modulate`: adaLN on a (B, H) vector. The token `modulate` unsqueezes
    the shift / scale to (B, 1, H) for (B, T, H) tokens; applied to a (B, H)
    vector it would broadcast to (B, B, H) without an error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.utils import BaseOutput

from .conditioning import CondSpec


@dataclass
class DenoiserOutput(BaseOutput):
    """The denoiser prediction, same shape as `sample` ((B, G) on LINCS).

    It can be eps, v, or a flow velocity, set by the training objective; the arm's `diffusion_method` in arch.json is what says how to interpret it.

    Reachable as `.sample` or as `return_dict=False)[0]`."""

    sample: torch.Tensor


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN on tokens: x (B, T, H), shift / scale (B, H)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def vec_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN on a vector: x, shift, scale all (B, H)."""
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Timestep embedder (verbatim from the reference implementation)
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Note, this is not a `CondSpec` field by nature of method"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """Create sinusoidal timestep embeddings.

        :param t: a 1-D Tensor of N indices, one per batch element (may be fractional).
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# The shared conditioning contract (from DiT2DModel)
# ---------------------------------------------------------------------------

def check_cond_spec(cond_spec: CondSpec, class_dropout_prob: float) -> None:
    """Constructor guards of DiT2DModel.__init__."""
    if len(cond_spec) == 0:
        raise ValueError(
            "cond_spec is empty. A generator with no conditioning has no "
            "estimand; pass the CondSpec from src.data.dataset.build_cond_spec.")
    if class_dropout_prob > 0 and not cond_spec.has_null:
        raise ValueError(
            f"class_dropout_prob={class_dropout_prob} but no field in the spec is "
            f"droppable (none has role 'A'), so there is nothing to drop and no "
            f"unconditional branch to guide against.")


def resolve_timestep_and_drop(model: nn.Module, sample: torch.Tensor, timestep, drop):
    """(t, drop) for `forward`: t as a (B,) tensor on sample's device, drop as a
    (B,) bool mask or None. Reads `model.cond_spec`, `model.class_dropout_prob`
    and `model.training`.
    """
    B = sample.shape[0]
    device = sample.device

    # timestep -> (B,) tensor. Infer the dtype instead of forcing long
    t = timestep
    if not torch.is_tensor(t):
        t = torch.as_tensor([t], device=device)
    elif t.dim() == 0:
        t = t[None].to(device)
    t = t.expand(B).to(device)

    # --- CFG mask: one decision per sample
    if drop is not None:
        if not model.cond_spec.has_null:
            raise ValueError(
                "drop was given but no field in the spec is droppable, so the "
                "model has no null to substitute.")
        if model.class_dropout_prob <= 0:
            raise ValueError(
                "drop requires class_dropout_prob > 0: the model was not trained "
                "for classifier-free guidance, so its nulls are untrained.")
        drop = drop.to(device).bool().expand(B)
    elif model.training and model.class_dropout_prob > 0:
        drop = torch.rand(B, device=device) < model.class_dropout_prob
    else:
        drop = None
    return t, drop
