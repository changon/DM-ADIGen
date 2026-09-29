"""Step-C semi-synthetic covariate `syn_c` (IMPLEMENT.md §3.8.2).

Phase 0 needs only the assignment: build_dataset writes one fixed `syn_c`
column so that every consumer sees the same draw. The injected effect
(`y <- y + syn_c * beta * v`, applied in dataset.py after centring and
z-scoring) lands in Phase 5.

`syn_c` is balanced at random within every group build_dataset passes: each
treated (compound, dose_level) arm and each plate's DMSO wells. The full data
is therefore unconfounded in `syn_c`, and `syn_c` is pre-treatment by
construction.
"""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np
import pandas as pd


def assign_syn_c(groups: Sequence[str], order_keys: Sequence[str], seed: int) -> np.ndarray:
    """(N,) int8 in {0, 1}, balanced within each group.

    A group of n rows gets floor(n/2) of one level and ceil(n/2) of the other,
    with the odd row's level drawn by coin flip (a 3-well arm is a 1/2 or 2/1
    split). Each group draws from its own generator, seeded by (`seed`, group
    name), and orders its rows by `order_keys` first, so a group's assignment
    does not depend on which other groups are in the table.
    """
    groups = np.asarray(groups, dtype=object)
    order_keys = np.asarray(order_keys, dtype=object)
    if groups.shape != order_keys.shape or groups.ndim != 1:
        raise ValueError("groups and order_keys must be 1-d and the same length")
    if pd.Series(order_keys).duplicated().any():
        raise ValueError("order_keys must be unique (they fix the row order within a group)")
    out = np.full(groups.shape[0], -1, dtype=np.int8)
    for g, idx in pd.Series(groups).groupby(groups, sort=True).indices.items():
        idx = idx[np.argsort(order_keys[idx].astype(str), kind="stable")]
        rng = np.random.default_rng([int(seed), zlib.crc32(str(g).encode())])
        lab = (np.arange(idx.size) % 2).astype(np.int8)
        if idx.size % 2 and rng.random() < 0.5:
            lab = 1 - lab
        out[idx[rng.permutation(idx.size)]] = lab
    if (out < 0).any():
        raise AssertionError("syn_c left rows unassigned")
    return out


def max_group_imbalance(groups: Sequence[str], syn_c: np.ndarray) -> int:
    """max over groups of |#(syn_c=1) - #(syn_c=0)|; 1 at most by construction."""
    s = pd.Series(np.asarray(syn_c, dtype=np.int64) * 2 - 1)
    return int(s.groupby(np.asarray(groups, dtype=object)).sum().abs().max()) if len(s) else 0
