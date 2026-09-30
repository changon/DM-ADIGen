"""URR overlap diagnostics: effective sample size and tail index of a weight vector.

Copied from RxRx19a/src/nuisances/knn_dr.py; fit_urr and export_urr_weights
import `ess` / `tail_index` from here. `knn_dr_weights` (kNN AIPW) is not
ported: it was retired on 2026-09-29 (IMPLEMENT.md §3.5, D1).
"""
from __future__ import annotations

import numpy as np

def tail_index(alpha: np.ndarray, frac: float = 0.05) -> float:
    """estimator of URR instability, the tail index. P(alpha > t) ~ t^(-1/k).

    E[alpha^2] finite <=> k < 0.5.  PSIS rule (Vehtari et al.): k > 0.7 => overlap failure.
    """
    # frac must stay small for well-controlled weights
    x = np.asarray(alpha, np.float64)
    x = np.sort(x[x > 0])[::-1]
    if len(x) < 20:
        return float("nan")
    m = max(10, int(frac * len(x)))
    top, thr = x[:m], x[m]
    return float(np.mean(np.log(top / thr)))

def ess(w: np.ndarray) -> float:
    """Effective sample size of a weight vector."""
    w = np.asarray(w, np.float64)
    s = w.sum()
    return float(s * s / max((w * w).sum(), 1e-12))
