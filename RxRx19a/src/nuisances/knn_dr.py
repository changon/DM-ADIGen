"""ADIGen's doubly-robust training weights

  1. DR weights via KNN
         w_i = alpha_i
               + (1/k) #{ j : i in NN(X'_pi(j), A_j) }         <- plug-in leg, PRODUCT pairs
               - (1/k) sum_{ j : i in NN(X_j, A_j) } alpha_j   <- correction,  OBSERVED pairs

     The plug-in leg queries a DONOR covariate X'_pi(j) (drawn independently of A_j via a permutation), which is what makes the product measure f_A x P_X appear.

  2. URR overlap dx checks

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

def knn_dr_weights(
    cov: np.ndarray,             # (n, d)  the covariates C  (= cov_vec)
    compound: np.ndarray,        # (n,)    treatment: compound index
    log10_conc: np.ndarray,      # (n,)    treatment: dose
    is_control: np.ndarray,      # (n,)    treatment: control flag
    alpha: np.ndarray,           # (n,)    riesz weight from fit_urr
    folds: np.ndarray,           # (n,)    cross-fit fold id (0/1)
    *,
    k: int = 12,
    bandwidth: float = 0.3,
    seed: int = 0,
    clip_neg: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Per-sample loss multiplier w for the product-measure (interventional) risk.
    Neighbours of a query (X, A) are drawn from the other cross-fit fold.
    """
    n = len(compound)
    a = np.asarray(alpha, np.float64)
    if verbose and abs(a.mean() - 1.0) > 1e-3:
        # monitor the URR mean, simply log it.
        print(f"[knn_dr] NOTE mean(alpha)={a.mean():.4f} != 1 "
              f"(score identity E[alpha]=1 does not hold on this fit)")
    X = np.asarray(cov, np.float32).reshape(n, -1)
    rng = np.random.default_rng(seed)
    tie_rng = np.random.default_rng(seed + 1_000_003)   # independent of `perm`
    perm = rng.permutation(n)          # donor covariates X'_pi(j), independent of A_j

    w = a.copy()
    n_nb = []

    for f in (0, 1):
        fit_idx = np.where(folds != f)[0]      # neighbours come from the OTHER fold
        eval_idx = np.where(folds == f)[0]
        if len(fit_idx) == 0 or len(eval_idx) == 0:
            continue

        buckets: dict[tuple[int, int], np.ndarray] = {}
        keys = list(zip(compound[fit_idx].tolist(), is_control[fit_idx].tolist()))
        order = np.argsort(np.asarray([hash(t) for t in keys]), kind="mergesort")
        for pos in range(len(fit_idx)):
            key = keys[pos]
            buckets.setdefault(key, []).append(fit_idx[pos])
        buckets = {kk: np.asarray(v, dtype=np.int64) for kk, v in buckets.items()}
        del order

        for j in eval_idx:
            cand = buckets.get((int(compound[j]), int(is_control[j])))
            if cand is None or len(cand) == 0:
                continue
            if not is_control[j]:
                # dose kernel: only neighbours within the bandwidth
                cand = cand[np.abs(log10_conc[cand] - log10_conc[j]) <= bandwidth]
                if len(cand) == 0:
                    continue
            # Randomise tie-breaking: a low-cardinality C gives few distinct distances, so argpartition would hand every query in a bucket the same k donors. permute to induce variation
            if len(cand) > k:
                cand = cand[tie_rng.permutation(len(cand))]
            Xc = X[cand]
            d_obs = ((Xc - X[j]) ** 2).sum(1)                 # query at OBSERVED X_j
            d_prod = ((Xc - X[perm[j]]) ** 2).sum(1)          # query at DONOR X'_pi(j)
            nn_o = cand[np.argpartition(d_obs, min(k, len(cand)) - 1)[:k]]
            nn_p = cand[np.argpartition(d_prod, min(k, len(cand)) - 1)[:k]]
            n_nb.append(len(nn_p))
            w[nn_p] += 1.0 / len(nn_p)                        # + zeta_hat at the product measure
            w[nn_o] -= a[j] / len(nn_o)                  # - alpha * zeta_hat at P

    if verbose:
        neg = -(w[w < 0].sum()) / max(w.sum(), 1e-12)
        print(f"[knn_dr] neighbours/query mean={np.mean(n_nb) if n_nb else 0:.1f} (k={k})  "
              f"mean(w)={w.mean():.4f}  neg mass={neg:.3f}  ESS={ess(np.abs(w)):,.0f}/{n:,}")

    if clip_neg: # ensure postiive
        w = np.clip(w, 0.0, None)
    return w.astype(np.float32)
