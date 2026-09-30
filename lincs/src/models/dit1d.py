"""1-D Diffusion Transformer for LINCS: Y is a (B, G) gene vector read as a
sequence of gene patches (IMPLEMENT.md §3.4.2).

Copied from RxRx19a/src/models/dit.py (always copy, §3.1.1): `DiTBlock`
(adaLN-Zero; the cross-attention branch is not ported, adaln only),
`get_1d_sincos_pos_embed_from_grid`, `DIT_SIZES`, and the init. Changed for
1-d:

  * patch embed: Linear(patch_size -> H) on the zero-padded vector viewed as
    (B, T, patch_size), i.e. a stride-p Conv1d with in_channels = 1;
  * frozen 1-d sin-cos positional embedding over the T tokens;
  * FinalLayer1D: Linear(H -> patch_size) (the 2-d one is p*p*C wide);
  * unpatchify: (B, T, p) -> (B, T*p), slicing the pad off.

patch_size 10 pads 978 -> 980 (98 tokens, pad 2); 6 divides 978 (163 tokens).
The gene order is the table's (gene_order.json); the trainer records it in
arch.json. Do NOT wrap the genes into (B, 1, H, W) for DiT2DModel.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .conditioning import CondEmbedder, CondSpec
from .layers import (DenoiserOutput, TimestepEmbedder, check_cond_spec, modulate,
                     resolve_timestep_and_drop)

try:
    from timm.models.vision_transformer import Attention, Mlp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "DiT1DModel needs timm (Attention / Mlp from "
        "timm.models.vision_transformer). Install it with `pip install timm`."
    ) from e


# Model sizes follow the paper (copied from RxRx19a dit.py). S/10 is the
# LINCS default (~33M, comparable to MLP-B); B/10 is the size ablation.
DIT_SIZES: dict[str, dict] = {
    "S":  dict(depth=12, hidden_size=384,  num_heads=6),    # ~33M
    "B":  dict(depth=12, hidden_size=768,  num_heads=12),   # ~130M
    "L":  dict(depth=24, hidden_size=1024, num_heads=16),   # ~458M
    "XL": dict(depth=28, hidden_size=1152, num_heads=16),   # ~675M
}


def patch_geometry(n_genes: int, patch_size: int) -> dict[str, int]:
    """{pad, n_tokens}: pad the G genes at the end to a multiple of patch_size."""
    n_genes, patch_size = int(n_genes), int(patch_size)
    if not 1 <= patch_size <= n_genes:
        raise ValueError(f"patch_size must be in [1, {n_genes}], got {patch_size}")
    pad = (-n_genes) % patch_size
    return {"pad": pad, "n_tokens": (n_genes + pad) // patch_size}


# ---------------------------------------------------------------------------
# Core DiT blocks (verbatim from the reference implementation; no cross-attn)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa: E731
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1D(nn.Module):
    """The final layer of DiT, emitting patch_size values per token."""

    def __init__(self, hidden_size, patch_size, out_channels=1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


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
# DiT1DModel
# ---------------------------------------------------------------------------

class DiT1DModel(nn.Module):
    """Spec-based 1-d diffusion transformer over gene patches.

        forward(sample, timestep, cond, drop=None, return_dict=True)

      sample    (B, n_genes) z-scored expression (noised).
      timestep  int, float, 0-dim tensor, or (B,) tensor; fractional under
                flow matching (t = tau * num_train_timesteps).
      cond      {field_name: tensor}, one entry per field in `cond_spec`.
      drop      optional (B,) bool; where True every role-A field is replaced
                by its learned null (the unconditional branch of CFG).
      returns   DenoiserOutput(sample=(B, n_genes)).
    """

    def __init__(
        self,
        n_genes: int = 978,
        cond_spec: CondSpec = CondSpec(),
        patch_size: int = 10,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
    ):
        super().__init__()
        check_cond_spec(cond_spec, class_dropout_prob)
        geo = patch_geometry(n_genes, patch_size)

        self.cond_spec = cond_spec
        self.class_dropout_prob = class_dropout_prob
        self.n_genes = int(n_genes)
        self.patch_size = int(patch_size)
        self.pad = geo["pad"]
        self.n_tokens = geo["n_tokens"]
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(self.patch_size, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_embedder = CondEmbedder(
            cond_spec, hidden_size, cfg_null=class_dropout_prob > 0)
        # Will use fixed sin-cos embedding (frozen; saved in the state_dict):
        self.register_buffer(
            "pos_embed", torch.zeros(1, self.n_tokens, hidden_size), persistent=True
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer1D(hidden_size, self.patch_size, 1)
        self.gradient_checkpointing = False
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # xavier also covers the patch embed, which is a Linear here (DiT
        # initialises its Conv2d "like nn.Linear" with the same xavier).
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by 1-d sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1], np.arange(self.n_tokens, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Conditioning table, MLP and learned null. call `calibrate_conditioning()` next, from whoever has the training data.
        self.cond_embedder.init_weights()

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def calibrate_conditioning(self, probes=None, **kw) -> dict[str, float]:
        """Passthrough to `CondEmbedder.calibrate`. Call once after construction, from the trainer (needs data). Returns the applied factors."""
        return self.cond_embedder.calibrate(probes, **kw)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def patchify(self, sample: torch.Tensor) -> torch.Tensor:
        """(B, G) -> (B, T, p), zero-padding the last token."""
        x = F.pad(sample, (0, self.pad)) if self.pad else sample
        return x.reshape(sample.shape[0], self.n_tokens, self.patch_size)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, p) -> (B, G), slicing the pad off."""
        return x.reshape(x.shape[0], self.n_tokens * self.patch_size)[:, :self.n_genes].contiguous()

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        cond: Mapping[str, torch.Tensor],
        drop: torch.Tensor | None = None,
        return_dict: bool = True,
        terms_out: dict[str, float] | None = None,
    ):
        if sample.dim() != 2 or sample.shape[1] != self.n_genes:
            raise ValueError(f"sample must be (B, {self.n_genes}), got {tuple(sample.shape)}")
        sample = sample.contiguous()
        t, drop = resolve_timestep_and_drop(self, sample, timestep, drop)

        x = self.x_embedder(self.patchify(sample)) + self.pos_embed         # (B, T, D)
        c = self.t_embedder(t) + self.cond_embedder(cond, drop, terms_out)  # (B, D)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)                                            # (B, T, D)
        x = self.final_layer(x, c)                                         # (B, T, p)
        out = self.unpatchify(x)                                           # (B, G)

        if not return_dict:
            return (out,)
        return DenoiserOutput(sample=out)
