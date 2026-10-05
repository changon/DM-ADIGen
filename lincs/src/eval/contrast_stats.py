"""The step-A statistic and its compound-clustered error (`STEP_A.md` §3). No PyTorch.

Every step-A number is a function of per-arm projections b_a:

    contrast(b)  = mean(b | high dose half) - mean(b | low dose half)    over a set of arms
    DiD          = contrast(b at gamma > 0) - contrast(b at gamma = 0)

The thinning draws independently per compound, and an arm-level correction (P1)
or an arm-level weight is shared by a compound's wells, so the COMPOUND is the
independent unit. Errors are therefore a delete-one-compound jackknife, never an
arm-level standard error. It is computed from per-compound sums, so it costs
O(compounds), and any function of several DiDs (a paired difference, a
difference of absolute values) gets its error from the same leave-one-out
replicates.
"""
from __future__ import annotations

import numpy as np


class HalfContrast:
    """Per-compound sums of b by dose half, for the full and leave-one-out contrast.

    `comp` are compound ids per arm, `half` is 0 / 1 (anything else is left out),
    `sel` selects the arms, and non-finite b are left out. `clusters` fixes the
    compound order, so several instances built on the same clusters have aligned
    leave-one-out replicates.
    """

    def __init__(self, b: np.ndarray, comp: np.ndarray, half: np.ndarray,
                 sel: np.ndarray, clusters: np.ndarray):
        b = np.asarray(b, dtype=np.float64)
        comp = np.asarray(comp, dtype=np.int64)
        half = np.asarray(half)
        self.clusters = np.asarray(clusters, dtype=np.int64)
        pos = {int(c): i for i, c in enumerate(self.clusters)}
        ci = np.array([pos.get(int(c), -1) for c in comp], dtype=np.int64)
        ok = np.asarray(sel, dtype=bool) & np.isfinite(b) & (ci >= 0)
        n = self.clusters.size
        self.S, self.N = {}, {}
        for h in (0, 1):
            m = ok & (half == h)
            self.S[h] = np.bincount(ci[m], weights=b[m], minlength=n)
            self.N[h] = np.bincount(ci[m], minlength=n).astype(np.float64)
        self.n_arms = int(ok.sum())

    @property
    def defined(self) -> bool:
        return bool(self.N[0].sum() > 0 and self.N[1].sum() > 0)

    def full(self) -> float:
        if not self.defined:
            return float("nan")
        return float(self.S[1].sum() / self.N[1].sum() - self.S[0].sum() / self.N[0].sum())

    def loo(self) -> np.ndarray:
        """(n_clusters,) the contrast with each compound left out in turn."""
        out = np.full(self.clusters.size, np.nan)
        if not self.defined:
            return out
        with np.errstate(invalid="ignore", divide="ignore"):
            hi = (self.S[1].sum() - self.S[1]) / (self.N[1].sum() - self.N[1])
            lo = (self.S[0].sum() - self.S[0]) / (self.N[0].sum() - self.N[0])
        return hi - lo


def jackknife_se(loo: np.ndarray) -> float | None:
    """Delete-one jackknife SE from the leave-one-out replicates."""
    v = np.asarray(loo, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    return float(np.sqrt((v.size - 1) / v.size * ((v - v.mean()) ** 2).sum()))


def did(b1: np.ndarray, b0: np.ndarray, comp: np.ndarray, half: np.ndarray,
        sel: np.ndarray, clusters: np.ndarray | None = None) -> dict:
    """The DiD of the high-minus-low contrast, with its jackknife replicates.

    `b1` / `b0` are the per-arm projections at gamma > 0 and gamma = 0 over the
    same arms. Returns value, se, the two contrasts, and `loo` (aligned to
    `clusters`) for building paired statistics.
    """
    if clusters is None:
        clusters = np.unique(np.asarray(comp)[np.asarray(sel, dtype=bool)])
    c1 = HalfContrast(b1, comp, half, sel, clusters)
    c0 = HalfContrast(b0, comp, half, sel, clusters)
    loo = c1.loo() - c0.loo()
    return {"value": c1.full() - c0.full(), "se": jackknife_se(loo),
            "contrast_g1": c1.full(), "contrast_g0": c0.full(),
            "n_arms": min(c1.n_arms, c0.n_arms), "n_clusters": int(np.asarray(clusters).size),
            "loo": loo, "clusters": np.asarray(clusters, dtype=np.int64)}


def paired(a: dict, b: dict, fn) -> dict:
    """value and jackknife SE of `fn(DiD_a, DiD_b)`, e.g. |b| - |a|. Both must be
    built on the same clusters."""
    if not np.array_equal(a["clusters"], b["clusters"]):
        raise ValueError("paired statistics need both DiDs on the same compounds")
    val = float(fn(a["value"], b["value"]))
    loo = fn(a["loo"], b["loo"])
    return {"value": val, "se": jackknife_se(loo)}
