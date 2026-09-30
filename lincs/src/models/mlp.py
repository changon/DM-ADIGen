"""MLP denoiser for LINCS: Y is a (B, G) vector of landmark genes (IMPLEMENT.md §3.4.1).

The headline backbone (decision 4). It shares the DiT's conditioning contract
(`CondSpec` + `CondEmbedder`, CFG nulls on role A) and its adaLN-Zero recipe,
applied to a vector instead of tokens:

    x = Linear(G -> H)(sample)
    c = t_emb(t) + CondEmbedder(cond, drop)                  (B, H)
    depth x   x = x + gate * Linear(SiLU(Linear(adaLN(LN(x), c))))
    out = Linear(H -> G)(adaLN(LN(x), c))                    (B, G)

Every adaLN head and the output Linear are zero-initialised, so each block
starts as the identity and the output is exactly 0 at init.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .conditioning import CondEmbedder, CondSpec
from .layers import (DenoiserOutput, TimestepEmbedder, check_cond_spec,
                     resolve_timestep_and_drop, vec_modulate)

# hidden / depth per size (§3.4.1). B is the default train arm.
MLP_SIZES: dict[str, dict] = {
    "S": dict(hidden_size=512, depth=4),
    "B": dict(hidden_size=1024, depth=6),
    "L": dict(hidden_size=2048, depth=8),
}


class MLPBlock(nn.Module):
    """Residual adaLN-Zero block: LN -> modulate -> Linear -> SiLU -> Linear, gated."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 1.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(hidden_size, inner, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(inner, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        h = self.fc2(self.act(self.fc1(vec_modulate(self.norm(x), shift, scale))))
        return x + gate * h


class FinalLayerVec(nn.Module):
    """adaLN + zero-initialised Linear to the G genes."""

    def __init__(self, hidden_size: int, out_features: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_features, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(vec_modulate(self.norm_final(x), shift, scale))


class MLPDenoiser(nn.Module):
    """Spec-based vector denoiser.

        forward(sample, timestep, cond, drop=None, return_dict=True)

      sample    (B, n_genes) z-scored expression (noised).
      timestep  int, float, 0-dim tensor, or (B,) tensor; fractional under
                flow matching (t = tau * num_train_timesteps).
      cond      {field_name: tensor}, one entry per field in `cond_spec`
                (src.data.dataset.cond_from_batch).
      drop      optional (B,) bool; where True every role-A field is replaced
                by its learned null (the unconditional branch of CFG).
      returns   DenoiserOutput(sample=(B, n_genes)).
    """

    def __init__(
        self,
        n_genes: int = 978,
        cond_spec: CondSpec = CondSpec(),
        hidden_size: int = 1024,
        depth: int = 6,
        mlp_ratio: float = 1.0,
        class_dropout_prob: float = 0.0,
    ):
        super().__init__()
        check_cond_spec(cond_spec, class_dropout_prob)
        self.cond_spec = cond_spec
        self.class_dropout_prob = class_dropout_prob
        self.n_genes = int(n_genes)

        self.x_embedder = nn.Linear(self.n_genes, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_embedder = CondEmbedder(
            cond_spec, hidden_size, cfg_null=class_dropout_prob > 0)
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayerVec(hidden_size, self.n_genes)
        self.gradient_checkpointing = False
        self.initialize_weights()

    def initialize_weights(self):
        # Same order as DiT2DModel.initialize_weights.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Conditioning table, MLP and learned null. call `calibrate_conditioning()` next, from whoever has the training data.
        self.cond_embedder.init_weights()

        # Zero-out adaLN modulation layers in the blocks:
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

        x = self.x_embedder(sample)                                         # (B, H)
        c = self.t_embedder(t) + self.cond_embedder(cond, drop, terms_out)  # (B, H)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)
        out = self.final_layer(x, c)                                        # (B, G)

        if not return_dict:
            return (out,)
        return DenoiserOutput(sample=out)
