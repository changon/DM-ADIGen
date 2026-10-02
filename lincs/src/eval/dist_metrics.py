"""Distributional-difference metrics on LINCS gene vectors.

Copied from `RxRx19a/src/eval/dist_metrics.py` (IMPLEMENT.md §3.1.1: always
copy, never import across the two trees), then cut down and fixed for a
`(n, 978)` outcome:

  - **numpy only.** The Inception extractor, the 5->3 channel hack, `_extract`,
    and the image orchestration (`compute_all` / `compute_per_slice`) are gone
    with their `torch` / `torchvision` imports, so the `--source real` oracle
    runs as a CPU job that never loads torch (IMPLEMENT.md §3.7, §3.9).
  - **`rbf_mmd2` no longer materialises `(n, n, d)`.** RxRx built
    `((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)` in both the kernel and the
    median-heuristic block. At d = 978 and its own `max_samples=4000` that is
    ~125 GB. Both now go through `_pairwise_sq_dists` (the Gram trick, which
    RxRx already had for PRDC). The bandwidth is returned so callers can record
    it, and a degenerate median no longer silently becomes 1.0 via `or`.
  - **`seed` is a parameter**, not a hard-coded `default_rng(0)` inside each
    function.
  - **Two Fréchet variants** (E5, §3.14). A full-covariance Fréchet in 978-d
    needs n >> 978 to estimate the covariance, so:
      `marginal_frechet`  per-gene closed form; usable on small groups, blind
                          to gene-gene correlations.
      `pc_frechet`        full covariance on top-k PCs fit on real TRAIN Y;
                          large pools only. `k` is chosen by explained variance
                          and recorded.
  - RxRx's `dose_bin` / `DOSE_BIN_EDGES_LOG10` are dropped: LINCS groups by
    `dose_level` (decision 3) or `splits.dose_half`.

`kid` and `prdc` are carried over unchanged (both are Gram-product only, so
they are memory-safe at d = 978) and sit behind `evaluate --extra_metrics`.

Convention everywhere: a float64-able `(n, n_genes)` matrix of plate-centred,
z-scored expression. Never clamped (IMPLEMENT.md §3.3).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers (copied)
# ---------------------------------------------------------------------------

def _stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def assert_blas_ok() -> float:
    """Refuse to compute anything on a numpy whose BLAS gets matrix products wrong.

    numpy's bundled OpenBLAS selects its Sapphire Rapids kernels on
    zabih-compute-01 and on the login node, and their dgemm returns garbage
    (relative error ~1.5; eigh and svd fail with it), while matrix-vector
    products stay correct (§5, step C2 scoring). Which shapes fail depends on
    the thread count (100x300 @ 300x100 fails at 4 threads but not at 1), but a
    300x300 product failed 12/12 draws at 1, 4 and all threads, so that is the
    shape tested here. Every
    MMD and PC / full Frechet computed there before 2026-10-02 is therefore
    invalid. `OPENBLAS_CORETYPE=Haswell` fixes it but must be set before numpy is
    imported, so it lives in the job scripts; this check is the backstop, like
    the CUDA preflight. Returns the measured relative error.
    """
    rng = np.random.default_rng(12345)
    a, b = rng.normal(size=(300, 300)), rng.normal(size=(300, 300))
    ref = np.einsum("ik,kj->ij", a, b, optimize=False)    # loops, no BLAS
    err = float(np.abs(a @ b - ref).max() / np.abs(ref).max())
    if not err < 1e-10:
        import os
        raise RuntimeError(
            f"numpy's BLAS computes matrix products wrongly on this node (relative "
            f"error {err:.2e} on a 300x300 @ 300x300 dgemm; "
            f"OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE')!r}). Set "
            f"OPENBLAS_CORETYPE=Haswell before Python starts (the job scripts do).")
    return err


# How often `_eigh_psd` had to leave numpy's eigh, per fallback. evaluate.py
# records it in the quality block, so a number computed on a fallback is traceable.
EIGH_FALLBACKS = {"scipy_evr": 0, "svd": 0}


def _eigh_psd(mat: np.ndarray, vectors: bool = True):
    """Eigendecomposition of a symmetric PSD matrix (ascending), VERIFIED.

    On zabih-compute-01, numpy's eigh (OpenBLAS dsyevd in this env) is wrong:
    on the step-C2 train covariance it returned finite eigenvalues with NaN
    eigenvectors, so every PC projection was NaN and the eval died; and on
    every Phase 4 / step-C eval it returned FINITE garbage, which surfaced only
    as "negative Frechet" warnings (a cross term of 476,123 against traces of
    ~7,000; §5, step C2 scoring). The same matrices decompose correctly on
    bindel. So a result is accepted only if it is finite, orthonormal, and
    reconstructs `mat`; otherwise this falls back to scipy's MRRR driver
    (scipy bundles its own LAPACK), then to an SVD (for a PSD matrix the
    singular values and vectors ARE the eigenpairs), and raises if all fail.
    """
    mat = (np.asarray(mat, dtype=np.float64) + np.asarray(mat, dtype=np.float64).T) * 0.5
    if not np.isfinite(mat).all():
        raise ValueError("_eigh_psd: the matrix itself is not finite")
    scale = max(float(np.abs(mat).max()), np.finfo(np.float64).tiny)

    def ok(w, v):
        if not (np.isfinite(w).all() and np.isfinite(v).all()):
            return False
        orth = float(np.abs(v.T @ v - np.eye(v.shape[1])).max())
        rec = float(np.abs((v * w) @ v.T - mat).max()) / scale
        return orth < 1e-7 and rec < 1e-7

    w, v = np.linalg.eigh(mat)
    if ok(w, v):
        return (w, v) if vectors else w
    try:
        from scipy.linalg import eigh as _s_eigh
        w, v = _s_eigh(mat, driver="evr")
        if ok(w, v):
            EIGH_FALLBACKS["scipy_evr"] += 1
            print(f"[dist_metrics] WARNING: numpy eigh failed verification on a "
                  f"{mat.shape[0]}-d matrix; used scipy eigh(driver='evr') "
                  f"(fallback #{sum(EIGH_FALLBACKS.values())})", file=sys.stderr)
            return (w, v) if vectors else w
    except Exception as e:                        # noqa: BLE001
        print(f"[dist_metrics] WARNING: scipy eigh(evr) failed too ({e})", file=sys.stderr)
    u, sv, _ = np.linalg.svd(mat)
    w, v = sv[::-1], u[:, ::-1]                    # ascending, like eigh
    if ok(w, v):
        EIGH_FALLBACKS["svd"] += 1
        print(f"[dist_metrics] WARNING: numpy and scipy eigh failed verification on a "
              f"{mat.shape[0]}-d matrix; used its SVD", file=sys.stderr)
        return (w, v) if vectors else w
    raise FloatingPointError(f"no verified eigendecomposition of a {mat.shape[0]}-d matrix "
                             f"(numpy eigh, scipy evr and SVD all failed)")


def _matrix_sqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable PSD-matrix square root via eigendecomposition (real, symmetric)."""
    mat = (mat + mat.T) * 0.5
    w, v = _eigh_psd(mat)
    w = np.clip(w, a_min=0.0, a_max=None)
    return (v * np.sqrt(w + eps)) @ v.T


def _drop_nonfinite(feats: np.ndarray, name: str) -> np.ndarray:
    """Drop rows containing any NaN/Inf."""
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"{name} must be 2-d (n, d), got shape {feats.shape}")
    ok = np.isfinite(feats).all(axis=1)
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"[dist_metrics] WARNING: dropping {n_bad}/{feats.shape[0]} "
              f"non-finite rows from {name}.", file=sys.stderr)
    return feats[ok]


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape (n_a, n_b). float64 for stability.

    The Gram form (a2 + b2 - 2 a.b) keeps this O(n_a * n_b), never O(n_a * n_b * d).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.clip(d2, 0.0, None)


def _kth_nn_radii(feats: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest neighbour within `feats` (excluding self)."""
    d2 = _pairwise_sq_dists(feats, feats)
    np.fill_diagonal(d2, np.inf)
    kth = np.partition(d2, kth=k - 1, axis=1)[:, k - 1]
    return np.sqrt(kth)


def _subsample(x: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples and x.shape[0] > max_samples:
        return x[rng.choice(x.shape[0], max_samples, replace=False)]
    return x


# ---------------------------------------------------------------------------
# Frechet distances
# ---------------------------------------------------------------------------

def frechet_distance(feat_a: np.ndarray, feat_b: np.ndarray, eps: float = 1e-6) -> float:
    """Full-covariance Frechet distance between two feature matrices.

    WARNING for gene space: the covariance is (d, d), so this needs n >> d rows
    per side to be meaningful. At d = 978 use it only on the large pools, or on
    `pc_frechet`'s projection. `marginal_frechet` is the small-group variant.
    """
    feat_a = _drop_nonfinite(feat_a, "feat_a")
    feat_b = _drop_nonfinite(feat_b, "feat_b")
    mu_a, sig_a = _stats(feat_a)
    mu_b, sig_b = _stats(feat_b)
    diff = mu_a - mu_b
    # tr( A + B - 2*sqrt(AB) ); compute sqrt(A) B sqrt(A) for symmetry.
    sa = _matrix_sqrt(sig_a, eps)
    inner = sa @ sig_b @ sa
    inner = (inner + inner.T) * 0.5
    ev = np.clip(_eigh_psd(inner, vectors=False), 0.0, None)
    tr_cross = float(np.sqrt(ev).sum())
    mean_term = float(diff @ diff)
    fid = mean_term + float(np.trace(sig_a)) + float(np.trace(sig_b)) - 2.0 * tr_cross
    if fid < -1e-3:
        print(f"[dist_metrics] WARNING: negative Frechet={fid:.4f} "
              f"(||du||^2={mean_term:.4f}, trA={np.trace(sig_a):.4f}, "
              f"trB={np.trace(sig_b):.4f}, 2*tr_cross={2 * tr_cross:.4f}, "
              f"d={sig_a.shape[0]}, n_a={feat_a.shape[0]}, n_b={feat_b.shape[0]}); "
              f"clamping to mean term.", file=sys.stderr)
        fid = mean_term
    return float(fid)


def marginal_frechet(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """Per-gene Frechet distance, summed over genes (E5).

    The 1-d Gaussian W2^2 has a closed form, so for diagonal covariances the
    Frechet distance is  sum_g [(mu_a - mu_b)^2 + (sd_a - sd_b)^2].  No (d, d)
    covariance is formed, so this is usable down to a handful of rows per side.
    It is blind to gene-gene correlations by construction.
    """
    a = _drop_nonfinite(feat_a, "feat_a")
    b = _drop_nonfinite(feat_b, "feat_b")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"gene count differs: {a.shape[1]} vs {b.shape[1]}")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError(f"need >= 2 rows per side, got {a.shape[0]} and {b.shape[0]}")
    # ddof=1: an unbiased SD, so two samples of the same distribution do not
    # separate just because one side has fewer rows.
    d_mu = a.mean(0) - b.mean(0)
    d_sd = a.std(0, ddof=1) - b.std(0, ddof=1)
    return float((d_mu * d_mu).sum() + (d_sd * d_sd).sum())


# ---------------------------------------------------------------------------
# PCA, fit on real TRAIN Y (E5)
# ---------------------------------------------------------------------------

def fit_pca(y_train: np.ndarray, var: float = 0.90, max_k: int | None = None) -> dict:
    """Top-k PCs of real TRAIN Y, k chosen by cumulative explained variance.

    Returns a dict (JSON-friendly apart from the arrays) recording `k`, the
    realised `explained`, and the rows the fit used, so a PC Frechet number can
    be traced to its basis.
    """
    y = _drop_nonfinite(y_train, "y_train")
    if not 0.0 < var <= 1.0:
        raise ValueError(f"var must be in (0, 1], got {var}")
    n, d = y.shape
    if n <= d:
        print(f"[dist_metrics] WARNING: fitting PCs on n={n} <= d={d} rows; "
              f"the tail eigenvalues are not identified.", file=sys.stderr)
    mean = y.mean(0)
    cov = np.cov(y, rowvar=False)
    w, v = _eigh_psd(cov)                 # ascending; never non-finite
    w = np.clip(w[::-1], 0.0, None)       # descending
    v = v[:, ::-1]
    total = float(w.sum())
    if total <= 0:
        raise ValueError("train Y has zero total variance")
    cum = np.cumsum(w) / total
    k = int(np.searchsorted(cum, var) + 1)
    k = min(k, d if max_k is None else min(d, int(max_k)))
    return {"mean": mean, "components": v[:, :k].T.copy(),   # (k, d)
            "k": k, "var_target": float(var),
            "explained": float(cum[k - 1]), "n_fit": int(n), "n_genes": int(d)}


def apply_pca(y: np.ndarray, pca: dict) -> np.ndarray:
    """Project rows onto the fitted PCs -> (n, k)."""
    y = _drop_nonfinite(y, "y")
    if y.shape[1] != pca["n_genes"]:
        raise ValueError(f"expected {pca['n_genes']} genes, got {y.shape[1]}")
    return (y - pca["mean"]) @ pca["components"].T


def pc_frechet(feat_a: np.ndarray, feat_b: np.ndarray, pca: dict,
               eps: float = 1e-6) -> float:
    """Full-covariance Frechet on the top-k PCs. Large pools only (n >> k)."""
    return frechet_distance(apply_pca(feat_a, pca), apply_pca(feat_b, pca), eps)


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

def median_bandwidth(feat_a: np.ndarray, feat_b: np.ndarray, *,
                     n_sub: int = 1000, seed: int = 0) -> float:
    """Median-heuristic RBF bandwidth over a subsample of the pooled rows.

    Unlike RxRx's version this uses the Gram trick, and a degenerate median
    (every sampled pair identical) raises instead of silently becoming 1.0.
    """
    rng = np.random.default_rng(seed)
    pool = np.concatenate([np.asarray(feat_a, dtype=np.float64),
                           np.asarray(feat_b, dtype=np.float64)], axis=0)
    sub = _subsample(pool, n_sub, rng)
    d2 = _pairwise_sq_dists(sub, sub)
    pos = d2[d2 > 0]
    if pos.size == 0:
        raise ValueError("median heuristic: every sampled pair is identical, "
                         "so there is no scale to set a bandwidth from")
    return float(np.sqrt(np.median(pos) / 2.0))


def rbf_mmd2(feat_a: np.ndarray, feat_b: np.ndarray,
             sigma: float | None = None, max_samples: int = 4000,
             seed: int = 0) -> tuple[float, float]:
    """Unbiased squared MMD with an RBF kernel. Returns (mmd2, sigma).

    `sigma` defaults to the median heuristic over the pooled subsample. Both
    sides are subsampled to `max_samples` because the kernel matrix is O(n^2) --
    but, unlike RxRx's version, no O(n^2 d) temporary is ever formed.
    """
    rng = np.random.default_rng(seed)
    a = _subsample(_drop_nonfinite(feat_a, "feat_a"), max_samples, rng)
    b = _subsample(_drop_nonfinite(feat_b, "feat_b"), max_samples, rng)
    n, m = a.shape[0], b.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"unbiased MMD needs >= 2 rows per side, got {n} and {m}")
    if sigma is None:
        sigma = median_bandwidth(a, b, seed=seed)
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    inv = -1.0 / (2.0 * sigma * sigma)

    kxx = np.exp(inv * _pairwise_sq_dists(a, a)); np.fill_diagonal(kxx, 0.0)
    kyy = np.exp(inv * _pairwise_sq_dists(b, b)); np.fill_diagonal(kyy, 0.0)
    kxy = np.exp(inv * _pairwise_sq_dists(a, b))
    mmd2 = (kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean())
    return float(mmd2), float(sigma)


# ---------------------------------------------------------------------------
# KID and PRDC (copied; both are Gram-product only, so safe at d = 978)
# ---------------------------------------------------------------------------

def kid(feat_a: np.ndarray, feat_b: np.ndarray, n_subsets: int = 100,
        subset_size: int = 1000, degree: int = 3,
        seed: int = 0) -> tuple[float, float]:
    """Unbiased KID with a polynomial kernel; returns (mean, std) across subsets."""
    rng = np.random.default_rng(seed)
    feat_a = _drop_nonfinite(feat_a, "feat_a")
    feat_b = _drop_nonfinite(feat_b, "feat_b")
    n_a, n_b = feat_a.shape[0], feat_b.shape[0]
    m = min(subset_size, n_a, n_b)
    if m < 2:
        raise ValueError(f"KID needs >= 2 rows per side, got {n_a} and {n_b}")
    d = feat_a.shape[1]
    c = 1.0
    gamma = 1.0 / d
    vals = np.empty(n_subsets, dtype=np.float64)
    for i in range(n_subsets):
        a = feat_a[rng.choice(n_a, m, replace=False)]
        b = feat_b[rng.choice(n_b, m, replace=False)]
        kaa = (gamma * a @ a.T + c) ** degree
        kbb = (gamma * b @ b.T + c) ** degree
        kab = (gamma * a @ b.T + c) ** degree
        np.fill_diagonal(kaa, 0.0); np.fill_diagonal(kbb, 0.0)
        vals[i] = (kaa.sum() / (m * (m - 1)) + kbb.sum() / (m * (m - 1))
                   - 2.0 * kab.mean())
    return float(vals.mean()), float(vals.std())


def prdc(real: np.ndarray, gen: np.ndarray, nearest_k: int = 5,
         max_samples: int = 10000, seed: int = 0) -> dict[str, float]:
    """Precision, Recall, Density, Coverage between two feature matrices."""
    rng = np.random.default_rng(seed)
    real = _subsample(_drop_nonfinite(real, "real"), max_samples, rng)
    gen = _subsample(_drop_nonfinite(gen, "gen"), max_samples, rng)

    n_real, n_gen = real.shape[0], gen.shape[0]
    k = min(nearest_k, n_real - 1, n_gen - 1)
    if k < 1:
        return {"precision": float("nan"), "recall": float("nan"),
                "density": float("nan"), "coverage": float("nan"),
                "nearest_k": 0, "n_real": int(n_real), "n_gen": int(n_gen)}

    real_radii = _kth_nn_radii(real, k)
    gen_radii = _kth_nn_radii(gen, k)
    d_rg = np.sqrt(_pairwise_sq_dists(real, gen))  # (n_real, n_gen)

    precision = float((d_rg <= real_radii[:, None]).any(axis=0).mean())
    recall = float((d_rg <= gen_radii[None, :]).any(axis=1).mean())
    density = float((1.0 / k) * (d_rg <= real_radii[:, None]).sum(axis=0).mean())
    coverage = float((d_rg <= real_radii[:, None]).any(axis=1).mean())
    return {"precision": precision, "recall": recall, "density": density,
            "coverage": coverage, "nearest_k": int(k),
            "n_real": int(n_real), "n_gen": int(n_gen)}


# ---------------------------------------------------------------------------
# Result container (copied)
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    name: str
    value: float
    extra: dict | None = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "value": self.value}
        if self.extra:
            d.update(self.extra)
        return d


def quality_metrics(real: np.ndarray, gen: np.ndarray, *,
                    pca: dict | None = None, mmd_max_samples: int = 4000,
                    extra: bool = False, seed: int = 0,
                    min_rows: int = 2) -> list[MetricResult]:
    """The E5 metric set for one group: marginal Frechet, MMD, and -- when a PCA
    basis is given and the group is large enough -- the top-k PC Frechet.

    Returns a `skipped` marker rather than raising when a group is too small, so
    a per-dose-level sweep does not die on its thinnest cell.
    """
    out: list[MetricResult] = []
    n_r, n_g = int(np.shape(real)[0]), int(np.shape(gen)[0])
    if n_r < min_rows or n_g < min_rows:
        return [MetricResult("skipped", float("nan"),
                            {"n_real": n_r, "n_gen": n_g, "min_rows": int(min_rows)})]
    out.append(MetricResult("frechet_marginal", marginal_frechet(real, gen),
                            {"n_real": n_r, "n_gen": n_g}))
    mmd2, sigma = rbf_mmd2(real, gen, max_samples=mmd_max_samples, seed=seed)
    out.append(MetricResult("mmd2_rbf", mmd2, {"sigma": sigma,
                                               "max_samples": int(mmd_max_samples)}))
    if pca is not None:
        k = int(pca["k"])
        # A (k, k) covariance per side needs more rows than k to be estimable.
        if n_r > k and n_g > k:
            out.append(MetricResult("frechet_pc", pc_frechet(real, gen, pca),
                                    {"k": k, "explained": pca["explained"]}))
        else:
            out.append(MetricResult("frechet_pc", float("nan"),
                                    {"k": k, "skipped": "n <= k",
                                     "n_real": n_r, "n_gen": n_g}))
    if extra:
        kid_mean, kid_std = kid(real, gen, seed=seed)
        out.append(MetricResult("kid", kid_mean, {"std": kid_std}))
        # One row per PRDC component: a single row would have to carry four
        # numbers in `extra` and a meaningless `value`.
        d = prdc(real, gen, seed=seed)
        meta = {k: d[k] for k in ("nearest_k", "n_real", "n_gen")}
        for nm in ("precision", "recall", "density", "coverage"):
            out.append(MetricResult(nm, d[nm], dict(meta)))
    return out
