"""Diffusion Transformer (DiT) for causal conditional generation.

A port of facebookresearch/DiT (`models.py` -- Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023)
Domain-agnostic: what the model conditions on is described by a `CondSpec` (see `conditioning.py`), so a different setting is a different spec, not different code.

UNCHANGED FROM THE ORIGINAL
---------------------------
  * `PatchEmbed` / `Attention` / `Mlp` from `timm.models.vision_transformer`.
  * `DiTBlock` is adaLN-Zero: a frozen (elementwise_affine=False) LayerNorm whose
    shift/scale/gate come from a SiLU->Linear head that is zero-initialised, so
    every block starts as the identity.
  * `FinalLayer` is adaLN + a zero-initialised Linear, then unpatchify.
  * `TimestepEmbedder` (sinusoidal -> 2-layer MLP), byte for byte.
  * Frozen 2-D sin-cos positional embedding (from MAE).
  * `initialize_weights()`: xavier-uniform basic init, N(0, 0.02) on the
    embedders, zeros on every adaLN head and on the final Linear.

**Conditioning is a spec.** 
    og DiT has exactly one conditioning input, an integer class label, so `forward(x, t, y)` suffices.
    Here it is different for counterfactual reasoning

       A  action        the estimand
       C  confounder    the adjustment set; never dropped, common between URR and outcome model
       E  environment   nuisance structure; for URR

   `CondEmbedder` turns a `{name: tensor}` dict into the adaLN vector
        One `nn.Embedding` per categorical field (og DiT's `LabelEmbedder` generalised, null row included)
        One MLP per continuous field:

       c = t_emb + sum_f term_f(cond[f])

Model sizes follow the paper (S/B/L/XL).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.utils import BaseOutput

from .conditioning import CondEmbedder, CondSpec

# adaln: every field is summed into one vector -> one modulation for all patches.
# xattn: the same adaLN path, PLUS role-A fields as cross-attention tokens, so a
# patch can weight a field by its own content. Additive and zero-initialised.
COND_MODES = ("adaln", "xattn")

try:
    from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "DiT needs timm (the reference implementation imports PatchEmbed / "
        "Attention / Mlp from timm.models.vision_transformer). "
        "Install it with `pip install timm`."
    ) from e

@dataclass
class DiT2DOutput(BaseOutput):
    """The denoiser prediction, `(B, out_channels, H, W)`.

    It can be eps, v, or a flow velocity, set by the training objective; the arm's `diffusion_method` in arch.json is what says how to interpret it.

    Reachable as `.sample` or as `return_dict=False)[0]`."""

    sample: torch.Tensor


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
# Core DiT blocks (verbatim from the reference implementation)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Image tokens (Q) attend to conditioning tokens (K, V).
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size {hidden_size} not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.kv = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, ctok):
        B, N, C = x.shape
        K = ctok.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = (self.kv(ctok).reshape(B, K, 2, self.num_heads, self.head_dim)
              .permute(2, 0, 3, 1, 4))
        o = torch.nn.functional.scaled_dot_product_attention(q, kv[0], kv[1])
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    `cross_attn=True` adds a conditioning-token sublayer. Its output projection
    is zero-initialised, so the block starts identical to the adaLN-only model.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cross_attn=False,
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm_x = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) \
            if cross_attn else None
        self.cross_attn = CrossAttention(hidden_size, num_heads) if cross_attn else None
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa: E731
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, ctok=None):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.cross_attn is not None and ctok is not None:
            x = x + self.cross_attn(self.norm_x(x), ctok)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Sin-cos positional embedding (from the MAE repo, as used by DiT)
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """grid_size: int of the grid height and width.
    returns: pos_embed of shape [grid_size*grid_size, embed_dim] (w/ or w/o cls_token)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class DiT2DModel(nn.Module):
    """Spec-based diffusion transformer.

        forward(sample, timestep, cond, drop=None, return_dict=True)

      sample    (B, in_channels, H, W) -- IMAGE channels only. 
                Conditioning is passed separately.
      timestep  int, float, 0-dim tensor, or (B,) tensor. 
                Fractional is fine and is the norm under flow matching (t = tau * num_train_timesteps).
      cond      {field_name: tensor}, one entry per field in `cond_spec`.
                Categorical -> (B,) integer indices. 
                Continuous -> (B,) or (B, dim) floats, NaN marking "not applicable".
      drop      optional (B,) bool. 
                Where True every action field is replaced by learned null, for the unconditional branch of CFG.
      returns   DiT2DOutput(sample=(B, out_channels, H, W)).
    """

    def __init__(
        self,
        sample_size: int = 128,        # H/W of input
        in_channels: int = 5,          # IMAGE channels only (80 for the latent arm)
        out_channels: int = 5,         # differs if latent
        cond_spec: CondSpec = CondSpec(),
        patch_size: int = 8,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
        cond_mode: str = "adaln",
    ):
        super().__init__()
        if cond_mode not in COND_MODES:
            raise ValueError(f"cond_mode must be one of {COND_MODES}, got {cond_mode!r}")
        if sample_size % patch_size != 0:
            raise ValueError(
                f"sample_size ({sample_size}) must be divisible by patch_size ({patch_size})"
            )
        if len(cond_spec) == 0:
            raise ValueError(
                "cond_spec is empty. A generator with no conditioning has no "
                "estimand; pass the domain's CondSpec (see src/domains/).")
        if class_dropout_prob > 0 and not cond_spec.has_null:
            raise ValueError(
                f"class_dropout_prob={class_dropout_prob} but no field in the spec is "
                f"droppable (none has role 'A'), so there is nothing to drop and no "
                f"unconditional branch to guide against.")

        self.cond_spec = cond_spec
        self.out_channels = out_channels
        self.class_dropout_prob = class_dropout_prob
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(sample_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_embedder = CondEmbedder(
            cond_spec, hidden_size, cfg_null=class_dropout_prob > 0)

        num_patches = self.x_embedder.num_patches
        # Fixed (sin-cos) positional embedding — frozen. Kept, so checkpoints can be loaded in.
        self.register_buffer(
            "pos_embed", torch.zeros(1, num_patches, hidden_size), persistent=True
        )

        # xattn is a superset of adaln: the adaLN path is unchanged and the cross-attn output projection starts at zero
        self.cond_mode = cond_mode
        self._xattn = cond_mode == "xattn"
        if self._xattn:
            n_act = sum(1 for f in cond_spec if f.role == "A")
            if n_act == 0:
                raise ValueError(
                    "cond_mode='xattn' needs at least one role-A field to use as a "
                    "conditioning token; this spec has none.")
            # A learned code per slot, so the model can tell the fields apart. 
            self.cond_token_type = nn.Parameter(torch.zeros(1, n_act, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     cross_attn=self._xattn)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self.gradient_checkpointing = False
        self.initialize_weights()

    # -- init (the reference scheme) ----------------------------------------
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Conditioning table, MLP and learned null. call `calibrate_conditioning()` next, from whoever has the training data.
        self.cond_embedder.init_weights()

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero the cross-attn output projection: the xattn pathway starts closed and opens only as it earns gradient, exactly as adaLN-Zero does.
        if self._xattn:
            nn.init.normal_(self.cond_token_type, std=0.02)
            for block in self.blocks:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def calibrate_conditioning(self, probes=None, **kw) -> dict[str, float]:
        """Passthrough to `CondEmbedder.calibrate`. 
        Call once after construction, from the trainer (needs data)
        Returns the applied factors."""
        return self.cond_embedder.calibrate(probes, **kw)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    # -- patchify / unpatchify ----------------------------------------------
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, patch_size**2 * C) -> imgs: (N, C, H, W)"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        cond: Mapping[str, torch.Tensor],
        drop: torch.Tensor | None = None,
        return_dict: bool = True,
        terms_out: dict[str, float] | None = None,
    ):
        B = sample.shape[0]
        device = sample.device

        sample = sample.contiguous()

        # timestep -> (B,) tensor. Infer the dtype instead of forcing long
        t = timestep
        if not torch.is_tensor(t):
            t = torch.as_tensor([t], device=device)
        elif t.dim() == 0:
            t = t[None].to(device)
        t = t.expand(B).to(device)

        # --- CFG mask: one decision per sample
        if drop is not None:
            if not self.cond_spec.has_null:
                raise ValueError(
                    "drop was given but no field in the spec is droppable, so the "
                    "model has no null to substitute.")
            if self.class_dropout_prob <= 0:
                raise ValueError(
                    "drop requires class_dropout_prob > 0: the model was not trained "
                    "for classifier-free guidance, so its nulls are untrained.")
            drop = drop.to(device).bool().expand(B)
        elif self.training and self.class_dropout_prob > 0:
            drop = torch.rand(B, device=device) < self.class_dropout_prob
        else:
            drop = None

        x = self.x_embedder(sample) + self.pos_embed                       # (N, T, D)
        c = self.t_embedder(t) + self.cond_embedder(cond, drop, terms_out)  # (N, D)

        # Same embedders as the adaLN sum, kept separate so attention can address each field; None under cond_mode='adaln', which skips the branch.
        ctok = None
        if self._xattn:
            ctok = (self.cond_embedder.field_tokens(cond, drop) + self.cond_token_type).to(x.dtype)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, c, ctok,
                                                      use_reentrant=False)
            else:
                x = block(x, c, ctok)                                      # (N, T, D)

        x = self.final_layer(x, c)                       # (N, T, patch**2 * C_out)
        out = self.unpatchify(x)                         # (N, C_out, H, W)

        if not return_dict:
            return (out,)
        return DiT2DOutput(sample=out)


# ---------------------------------------------------------------------------
# Paper configurations (depth / hidden_size / num_heads)
# ---------------------------------------------------------------------------

DIT_SIZES: dict[str, dict] = {
    "S":  dict(depth=12, hidden_size=384,  num_heads=6),    # ~33M
    "B":  dict(depth=12, hidden_size=768,  num_heads=12),   # ~130M
    "L":  dict(depth=24, hidden_size=1024, num_heads=16),   # ~458M
    "XL": dict(depth=28, hidden_size=1152, num_heads=16),   # ~675M
}
