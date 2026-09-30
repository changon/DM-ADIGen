"""Riesz alpha network for (compound, conc) action + covariates X.

Copied from RxRx19a/src/nuisances/alpha_net.py. LINCS edit (IMPLEMENT.md §3.5):
no `infected` input -- the RxRx population bit has no LINCS analogue.
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn

# solves log(1 + exp(s)) = 1, so softplus(0 + SP_SHIFT) == 1. `warm-start` in some sense so urr is learned more easily
SP_SHIFT = math.log(math.e - 1)

class AlphaNet(nn.Module):
    def __init__(
        self,
        n_compounds: int,
        cov_dim: int,
        compound_embed_dim: int = 32,
        hidden: tuple[int, ...] = (256, 128, 64),
        cov_idx: list[int] | None = None,
        positive: bool = False,
    ):
        super().__init__()
        self.full_cov_dim = int(cov_dim)
        self.positive = bool(positive)
        idx = torch.arange(cov_dim) if cov_idx is None else torch.as_tensor(
            list(cov_idx), dtype=torch.long)
        self.register_buffer("cov_idx", idx, persistent=False)

        self.embed = nn.Embedding(n_compounds, compound_embed_dim)
        # +2: log10_conc, is_control flag
        in_dim = int(idx.numel()) + compound_embed_dim + 2
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

        self._spec = {
            "n_compounds": int(n_compounds),
            "cov_dim": int(cov_dim),
            "compound_embed_dim": int(compound_embed_dim),
            "hidden": list(hidden),
            "cov_idx": [int(i) for i in idx.tolist()],
            "positive": bool(positive),
        }

    def forward(
        self,
        cov_vec: torch.Tensor,        # (B, full_cov_dim) -- always the FULL vector
        compound_idx: torch.Tensor,   # (B,) long
        log10_conc: torch.Tensor,     # (B,)
        is_control: torch.Tensor,     # (B,) float
    ) -> torch.Tensor:
        e = self.embed(compound_idx)
        c = cov_vec.index_select(-1, self.cov_idx)
        x = torch.cat(
            [c, e, log10_conc.unsqueeze(-1), is_control.unsqueeze(-1)],
            dim=-1,
        )
        out = self.mlp(x).squeeze(-1)
        return nn.functional.softplus(out + SP_SHIFT) if self.positive else out

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        with open(spec_path(path), "w") as f:
            json.dump(self._spec, f, indent=2)

    @classmethod
    def load(cls, path: str, map_location=None) -> "AlphaNet":
        """Rebuild from the sibling `<path>.spec.json`."""
        sp = spec_path(path)
        if not os.path.exists(sp):
            raise FileNotFoundError(
                f"{path} has no {os.path.basename(sp)}; it predates the URR fix. "
                f"Re-run `python -m src.nuisances.fit_urr`.")
        with open(sp) as f:
            spec = json.load(f)

        net = cls(
            n_compounds=spec["n_compounds"],
            cov_dim=spec["cov_dim"],
            compound_embed_dim=spec.get("compound_embed_dim", 32),
            hidden=tuple(spec.get("hidden", (256, 128, 64))),
            cov_idx=spec.get("cov_idx"),
            positive=spec.get("positive", False),
        )
        sd = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        net.load_state_dict(sd)
        return net


def spec_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".spec.json"
