"""Per-context values of every policy, kept across stages so the report can pair
arms by compound. No PyTorch."""
from __future__ import annotations

import os

import numpy as np

from src.data.build_dataset import _atomic_write


def save_values(path: str, F: dict, block: dict, merge: bool = True) -> None:
    """Write {utility/policy: (n_ctx,) value} plus the frame's context ids; with
    `merge`, keep what an earlier stage wrote."""
    out = {}
    if merge and os.path.isfile(path):
        z = np.load(path, allow_pickle=False)
        if not np.array_equal(z["compound_idx"], F["comp"]):
            raise RuntimeError(f"{path} holds other contexts")
        out.update({k: z[k] for k in z.files if k not in ("compound_idx", "line_index", "role", "eligible")})
    for ut, blk in block.items():
        for grp in ("real", "generators"):
            for name, rec in blk.get(grp, {}).items():
                out[f"{ut}/{name}"] = rec["_v"].astype(np.float32)
    _atomic_write(path, lambda f: np.savez(f, compound_idx=F["comp"], line_index=F["ctx_line"],
                                           role=F["role"], eligible=F["eligible"], **out), mode="wb")


def load_values(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k].astype(np.float64) for k in z.files if "/" in k}
