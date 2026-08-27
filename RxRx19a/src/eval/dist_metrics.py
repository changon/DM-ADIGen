"""Distributional-difference metrics for RxRx19a generative evaluation.

Some metrics:

1.  FID + KID computed on Inception-V3 (pool3) features.

2.  FID + RBF-MMD computed on the domain ResNet18 features from `src.eval.feature_extractor`. 

3.  Conditional density evaluations

Image convention everywhere: float tensors in [-1, 1], shape (N, C, H, W),
C = n_channels (5 for RxRx).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Inception feature extractor (3-ch input)
# ---------------------------------------------------------------------------

class _InceptionFeatures(nn.Module):
    """Returns the 2048-d pool3 features used by canonical FID."""

    def __init__(self):
        super().__init__()
        # `weights=None` would skip the pretrained download; use IMAGENET1K_V1.
        net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                  aux_logits=True)
        net.fc = nn.Identity()
        net.eval()
        self.net = net

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1]. Inception wants 299x299.
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # to [-1, 1] which matches the model's expected normalize
        out = self.net(x)
        return out if isinstance(out, torch.Tensor) else out[0]


def _five_to_three(x: torch.Tensor, channels: Sequence[int] = (1, 2, 3)) -> torch.Tensor:
    """Pick three channels out of the 5-channel stack to feed Inception."""
    if len(channels) != 3:
        raise ValueError("channels must be a 3-tuple")
    return x[:, list(channels), :, :]


def _to_inception_input(x_minus1_1: torch.Tensor,
                        channels: Sequence[int] = (1, 2, 3)) -> torch.Tensor:
    """[-1, 1] 5-ch -> [0, 1] 3-ch."""
    x = (x_minus1_1.clamp(-1, 1) + 1.0) * 0.5
    return _five_to_three(x, channels)


# ---------------------------------------------------------------------------
# Generic Frechet (FID) + KID on a precomputed feature matrix
# ---------------------------------------------------------------------------

def _stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _matrix_sqrt(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable PSD-matrix square root via eigendecomposition (real, symmetric)."""
    mat = (mat + mat.T) * 0.5
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, a_min=0.0, a_max=None)
    return (v * np.sqrt(w + eps)) @ v.T


def _drop_nonfinite(feats: np.ndarray, name: str) -> np.ndarray:
    """Drop rows containing any NaN/Inf."""
    feats = np.asarray(feats, dtype=np.float64)
    ok = np.isfinite(feats).all(axis=1)
    n_bad = int((~ok).sum())
    if n_bad:
        import sys
        print(f"[dist_metrics] WARNING: dropping {n_bad}/{feats.shape[0]} "
              f"non-finite rows from {name}.", file=sys.stderr)
    return feats[ok]


def frechet_distance(feat_a: np.ndarray, feat_b: np.ndarray, eps: float = 1e-6) -> float:
    """Standard FID formula on two feature matrices. """
    feat_a = _drop_nonfinite(feat_a, "feat_a")
    feat_b = _drop_nonfinite(feat_b, "feat_b")
    mu_a, sig_a = _stats(feat_a)
    mu_b, sig_b = _stats(feat_b)
    diff = mu_a - mu_b
    # tr( A + B - 2*sqrt(AB) ); compute sqrt(A) B sqrt(A) for symmetry.
    sa = _matrix_sqrt(sig_a, eps)
    inner = sa @ sig_b @ sa
    inner = (inner + inner.T) * 0.5
    ev = np.clip(np.linalg.eigvalsh(inner), 0.0, None)
    tr_cross = float(np.sqrt(ev).sum())
    mean_term = float(diff @ diff)
    fid = mean_term + float(np.trace(sig_a)) + float(np.trace(sig_b)) - 2.0 * tr_cross
    if fid < -1e-3:
        import sys
        print(f"[dist_metrics] WARNING: negative FID={fid:.4f} "
              f"(||du||^2={mean_term:.4f}, trA={np.trace(sig_a):.4f}, "
              f"trB={np.trace(sig_b):.4f}, 2*tr_cross={2*tr_cross:.4f}, "
              f"d={sig_a.shape[0]}, n_a={feat_a.shape[0]}, n_b={feat_b.shape[0]}); "
              f"clamping to mean term.", file=sys.stderr)
        fid = mean_term
    return float(fid)


def kid(feat_a: np.ndarray, feat_b: np.ndarray, n_subsets: int = 100,
        subset_size: int = 1000, degree: int = 3) -> tuple[float, float]:
    """Unbiased KID with a polynomial kernel; returns (mean, std) across subsets."""
    rng = np.random.default_rng(0)
    n_a, n_b = feat_a.shape[0], feat_b.shape[0]
    m = min(subset_size, n_a, n_b)
    d = feat_a.shape[1]
    c = 1.0
    gamma = 1.0 / d
    vals = np.empty(n_subsets, dtype=np.float64)
    for i in range(n_subsets):
        ia = rng.choice(n_a, m, replace=False)
        ib = rng.choice(n_b, m, replace=False)
        a = feat_a[ia]; b = feat_b[ib]
        kaa = (gamma * a @ a.T + c) ** degree
        kbb = (gamma * b @ b.T + c) ** degree
        kab = (gamma * a @ b.T + c) ** degree
        # Unbiased: drop diagonal in kaa, kbb.
        np.fill_diagonal(kaa, 0.0); np.fill_diagonal(kbb, 0.0)
        sa = kaa.sum() / (m * (m - 1))
        sb = kbb.sum() / (m * (m - 1))
        sab = kab.mean()
        vals[i] = sa + sb - 2.0 * sab
    return float(vals.mean()), float(vals.std())


def rbf_mmd2(feat_a: np.ndarray, feat_b: np.ndarray,
             sigma: float | None = None, max_samples: int = 4000) -> float:
    """Unbiased squared MMD with an RBF kernel.

    `sigma` defaults to the median pairwise distance over a sample (median heuristic). 
    Subsamples to `max_samples` per side because the kernel matrix is O(n^2).
    """
    rng = np.random.default_rng(0)
    if feat_a.shape[0] > max_samples:
        feat_a = feat_a[rng.choice(feat_a.shape[0], max_samples, replace=False)]
    if feat_b.shape[0] > max_samples:
        feat_b = feat_b[rng.choice(feat_b.shape[0], max_samples, replace=False)]

    if sigma is None:
        # Median heuristic on a 1000-pt subsample of the combined set.
        pool = np.concatenate([feat_a, feat_b], axis=0)
        idx = rng.choice(pool.shape[0], min(1000, pool.shape[0]), replace=False)
        sub = pool[idx]
        d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
        sigma = float(np.sqrt(np.median(d2[d2 > 0]) / 2.0)) or 1.0

    def k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2.0 * sigma * sigma))

    kxx = k(feat_a, feat_a); np.fill_diagonal(kxx, 0.0)
    kyy = k(feat_b, feat_b); np.fill_diagonal(kyy, 0.0)
    kxy = k(feat_a, feat_b)
    n, m = feat_a.shape[0], feat_b.shape[0]
    return float(
        kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()
    )


# ---------------------------------------------------------------------------
# Precision / Recall / Density / Coverage (Naeem et al., 2020)...
# ---------------------------------------------------------------------------

def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape (n_a, n_b). float64 for stability."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.clip(d2, 0.0, None)

def _kth_nn_radii(feats: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest neighbour within `feats` (excluding self)."""
    d2 = _pairwise_sq_dists(feats, feats)
    np.fill_diagonal(d2, np.inf)
    # k-th smallest (1-indexed k) -> index k-1 after partial sort.
    kth = np.partition(d2, kth=k - 1, axis=1)[:, k - 1]
    return np.sqrt(kth)

def prdc(real: np.ndarray, gen: np.ndarray, nearest_k: int = 5,
         max_samples: int = 10000) -> dict[str, float]:
    """Precision, Recall, Density, Coverage between two feature matrices.

    Subsamples to `max_samples` per side (kernel/NN matrices are O(n^2)).
    """
    rng = np.random.default_rng(0)
    if real.shape[0] > max_samples:
        real = real[rng.choice(real.shape[0], max_samples, replace=False)]
    if gen.shape[0] > max_samples:
        gen = gen[rng.choice(gen.shape[0], max_samples, replace=False)]

    n_real, n_gen = real.shape[0], gen.shape[0]
    k = min(nearest_k, n_real - 1, n_gen - 1)
    if k < 1:
        return {"precision": float("nan"), "recall": float("nan"),
                "density": float("nan"), "coverage": float("nan"),
                "nearest_k": 0}

    real_radii = _kth_nn_radii(real, k)
    gen_radii = _kth_nn_radii(gen, k)
    d_rg = np.sqrt(_pairwise_sq_dists(real, gen))  # (n_real, n_gen)

    # Precision: gen point inside any real sphere.
    precision = float((d_rg <= real_radii[:, None]).any(axis=0).mean())
    # Recall: real point inside any gen sphere.
    recall = float((d_rg <= gen_radii[None, :]).any(axis=1).mean())
    # Density: avg number of real spheres each gen point falls in, /k.
    density = float((1.0 / k) * (d_rg <= real_radii[:, None]).sum(axis=0).mean())
    # Coverage: fraction of real spheres containing >=1 gen point.
    coverage = float((d_rg <= real_radii[:, None]).any(axis=1).mean())
    return {"precision": precision, "recall": recall, "density": density,
            "coverage": coverage, "nearest_k": int(k),
            "n_real": int(n_real), "n_gen": int(n_gen)}


# ---------------------------------------------------------------------------
# Feature extraction over image tensors (batched, on GPU when available)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract(
    model: nn.Module,
    images: torch.Tensor,
    *,
    pre: callable | None = None,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or next(model.parameters()).device
    feats = []
    n = images.shape[0]
    model.eval()
    for i in range(0, n, batch_size):
        chunk = images[i : i + batch_size].to(device, non_blocking=True)
        if pre is not None:
            chunk = pre(chunk)
        f = model(chunk).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Public API
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


def compute_all(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    domain_model: nn.Module | None = None,
    inception_channels: Sequence[int] = (1, 2, 3),
    device: torch.device | str = "cuda",
    batch_size: int = 64,
) -> list[MetricResult]:
    """Compute the full metric panel given paired real/generated image tensors.

    real_images, gen_images: (N, C, H, W) in [-1, 1].

    Returns a list of MetricResult — convertible to JSON-friendly dicts.
    """
    device = torch.device(device if isinstance(device, str) else device)

    out: list[MetricResult] = []

    # --- Inception (3-ch projection) -----------------------------------------
    incept = _InceptionFeatures().to(device).eval()
    pre = lambda x: _to_inception_input(x, inception_channels)  # noqa: E731
    f_real = _extract(incept, real_images, pre=pre, batch_size=batch_size, device=device)
    f_gen = _extract(incept, gen_images, pre=pre, batch_size=batch_size, device=device)
    out.append(MetricResult("fid_inception", frechet_distance(f_real, f_gen),
                            {"channels": list(inception_channels),
                             "n_real": int(f_real.shape[0]),
                             "n_gen": int(f_gen.shape[0])}))
    kid_mean, kid_std = kid(f_real, f_gen)
    out.append(MetricResult("kid_inception", kid_mean, {"std": kid_std}))
    prdc_incept = prdc(f_real, f_gen)
    for key in ("precision", "recall", "density", "coverage"):
        out.append(MetricResult(f"{key}_inception", prdc_incept[key],
                                {"nearest_k": prdc_incept["nearest_k"]}))
    del incept; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Domain ResNet18 (5-ch native) ---------------------------------------
    if domain_model is not None:
        domain_model = domain_model.to(device).eval()
        emb = lambda batch: domain_model.embed(batch)  # noqa: E731

        class _EmbedWrap(nn.Module):
            def __init__(self, base): super().__init__(); self.base = base
            def forward(self, x): return self.base.embed(x)

        wrap = _EmbedWrap(domain_model).to(device).eval()
        f_real_d = _extract(wrap, real_images, batch_size=batch_size, device=device)
        f_gen_d = _extract(wrap, gen_images, batch_size=batch_size, device=device)
        out.append(MetricResult("fid_domain", frechet_distance(f_real_d, f_gen_d),
                                {"embed_dim": int(f_real_d.shape[1])}))
        out.append(MetricResult("mmd_rbf_domain", rbf_mmd2(f_real_d, f_gen_d)))
        prdc_domain = prdc(f_real_d, f_gen_d)
        for key in ("precision", "recall", "density", "coverage"):
            out.append(MetricResult(f"{key}_domain", prdc_domain[key],
                                    {"nearest_k": prdc_domain["nearest_k"]}))
        del wrap

    return out


def compute_per_slice(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    real_slice_id: np.ndarray,
    gen_slice_id: np.ndarray,
    slice_names: dict[int, str],
    domain_model: nn.Module,
    *,
    device: torch.device | str = "cuda",
    batch_size: int = 64,
    min_per_slice: int = 50,
) -> dict[str, dict]:
    """Compute domain FID + MMD per slice (e.g. per dose-bin).

    real_slice_id / gen_slice_id give the slice membership for each row.
    Slices with fewer than `min_per_slice` samples on either side are skipped.
    """
    device = torch.device(device if isinstance(device, str) else device)
    domain_model = domain_model.to(device).eval()

    class _EmbedWrap(nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return self.base.embed(x)

    wrap = _EmbedWrap(domain_model).to(device).eval()
    f_real = _extract(wrap, real_images, batch_size=batch_size, device=device)
    f_gen = _extract(wrap, gen_images, batch_size=batch_size, device=device)

    results: dict[str, dict] = {}
    for sid, sname in slice_names.items():
        r_mask = real_slice_id == sid
        g_mask = gen_slice_id == sid
        n_r, n_g = int(r_mask.sum()), int(g_mask.sum())
        if n_r < min_per_slice or n_g < min_per_slice:
            results[sname] = {"skipped": True, "n_real": n_r, "n_gen": n_g}
            continue
        results[sname] = {
            "n_real": n_r,
            "n_gen": n_g,
            "fid_domain": frechet_distance(f_real[r_mask], f_gen[g_mask]),
            "mmd_rbf_domain": rbf_mmd2(f_real[r_mask], f_gen[g_mask]),
        }
    return results


# ---------------------------------------------------------------------------
# Dose-bin helper 
# ---------------------------------------------------------------------------

DOSE_BIN_EDGES_LOG10 = (-2.0, -0.5, 0.5)  # control, low, mid, high

def dose_bin(log10_conc: Iterable[float], is_control: Iterable[int]) -> np.ndarray:
    """Return an int array in {0,1,2,3} = (control, low, mid, high).

    Log10 thresholds match the standard RxRx dose ladder breakpoints (~0.01 µM and ~0.3 µM).
    """
    out = np.empty(len(list(is_control) if not hasattr(is_control, "__len__") else is_control), dtype=np.int64)
    ic_arr = np.asarray(list(is_control), dtype=np.int64)
    lx_arr = np.asarray(list(log10_conc), dtype=np.float32)
    out[ic_arr == 1] = 0
    non_ctrl = ic_arr == 0
    lx = lx_arr[non_ctrl]
    bin_id = np.ones_like(lx, dtype=np.int64)  # low
    bin_id[lx > DOSE_BIN_EDGES_LOG10[1]] = 2  # mid
    bin_id[lx > DOSE_BIN_EDGES_LOG10[2]] = 3  # high
    out[non_ctrl] = bin_id
    return out


DOSE_BIN_NAMES = {0: "control", 1: "low", 2: "mid", 3: "high"}
