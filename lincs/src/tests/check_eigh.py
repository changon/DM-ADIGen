"""Regression check for `dist_metrics._eigh_psd` (IMPLEMENT.md §5, step C2 scoring). No PyTorch.

Every step-C2 eval on zabih-compute-01 died in the quality block: numpy's eigh
(OpenBLAS dsyevd) returned finite eigenvalues but NaN eigenvectors on the C2
train covariance, so every PC projection was NaN. This rebuilds that covariance
the way evaluate does, reports what numpy's eigh does with it on THIS node, and
checks that the robust path (`fit_pca` -> `pc_frechet`) is finite. It also forces
each fallback (scipy evr, then SVD) on a random PSD matrix and checks they agree
with a good decomposition.

    python -m src.tests.check_eigh --data_dir data/mcf7_24h --syn_meta syn_meta_compound_r1.json
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.data.synthetic import inject_meta, load_syn_meta, table_syn_seed  # noqa: E402
from src.eval import dist_metrics as dm  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, add_syn_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser()
    add_syn_cli(p)
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    from dataclasses import replace
    cfg.outcome = replace(cfg.outcome, syn_effect=1.0, syn_seed=table_syn_seed(cfg))
    print(f"[eigh] host {socket.gethostname()}  numpy {np.__version__}  "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}  "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    # ---- 1. the forced fallbacks, on a random PSD matrix ----------------------
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 40))
    m = a.T @ a / 300
    w_ref, v_ref = np.linalg.eigh(m)
    real_eigh, real_svd = np.linalg.eigh, np.linalg.svd
    import scipy.linalg as sl
    real_s = sl.eigh
    try:
        # finite garbage (what zabih-compute-01 returned in Phase 4 / step C)
        np.linalg.eigh = lambda x: (real_eigh(x)[0],
                                    np.random.default_rng(1).normal(size=x.shape))
        w0, v0 = dm._eigh_psd(m)
        check(dm.EIGH_FALLBACKS["scipy_evr"] == 1 and np.allclose(w0, w_ref),
              "numpy eigh with FINITE garbage vectors -> rejected by verification, scipy evr")
        dm.EIGH_FALLBACKS.update(scipy_evr=0, svd=0)
        np.linalg.eigh = lambda x: (real_eigh(x)[0], np.full_like(x, np.nan))
        w1, v1 = dm._eigh_psd(m)
        check(dm.EIGH_FALLBACKS["scipy_evr"] == 1 and np.allclose(w1, w_ref)
              and np.allclose(np.abs(v1.T @ v_ref), np.eye(40), atol=1e-8),
              "numpy eigh with NaN vectors -> scipy evr, same eigenpairs")
        sl.eigh = lambda *x, **k: (np.full(40, np.nan), np.full((40, 40), np.nan))
        w2, v2 = dm._eigh_psd(m)
        check(dm.EIGH_FALLBACKS["svd"] == 1 and np.allclose(w2, w_ref)
              and np.allclose(np.abs(v2.T @ v_ref), np.eye(40), atol=1e-8),
              "numpy and scipy both non-finite -> SVD, same eigenpairs (ascending)")
        np.linalg.svd = lambda x, *q, **k: tuple(np.full_like(r, np.nan) for r in real_svd(x))
        try:
            dm._eigh_psd(m)
            check(False, "all three non-finite -> raises")
        except FloatingPointError:
            check(True, "all three non-finite -> raises FloatingPointError")
    finally:
        np.linalg.eigh, np.linalg.svd, sl.eigh = real_eigh, real_svd, real_s
    check(np.linalg.eigh is real_eigh and np.linalg.svd is real_svd and sl.eigh is real_s,
          "the real solvers are restored before the data checks")
    dm.EIGH_FALLBACKS.update(scipy_evr=0, svd=0)

    # ---- 2. the real C2 train covariance --------------------------------------
    syn = load_syn_meta(cfg, n_genes=cfg.outcome.n_genes)
    splits = load_splits(cfg)
    em = load_expr_meta(cfg, None, splits=splits)
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "det_plate", "syn_c", "is_control"]).to_pandas())
    expr = np.load(cfg.paths.expr_npy, mmap_mode="r")
    y = normalize_expr(np.asarray(expr), plate_codes(meta["det_plate"].values, em["plates"]), em)
    y = inject_meta(y, meta["syn_c"].values.astype(np.int64),
                    meta["compound_idx"].values.astype(np.int64), syn)
    tr = np.asarray(splits["train_idx"], dtype=np.int64)
    yt = y[tr].astype(np.float64)
    cov = np.cov(yt, rowvar=False)
    w, v = np.linalg.eigh(cov)
    fin = bool(np.isfinite(w).all() and np.isfinite(v).all())
    orth = float(np.abs(v.T @ v - np.eye(v.shape[1])).max()) if fin else float("nan")
    rec = (float(np.abs((v * w) @ v.T - cov).max() / np.abs(cov).max()) if fin
           else float("nan"))
    print(f"[eigh] {syn['name']}: RAW numpy eigh on the {cov.shape[0]}-d train covariance: "
          f"finite {fin} ({int((~np.isfinite(v)).sum())} non-finite vector entries), "
          f"orthonormality error {orth:.1e}, reconstruction error {rec:.1e} -> "
          f"{'GOOD' if fin and orth < 1e-7 and rec < 1e-7 else 'BAD'} on this node",
          flush=True)
    pca = dm.fit_pca(yt, var=0.90)
    check(np.isfinite(pca["components"]).all() and np.isfinite(pca["mean"]).all(),
          f"fit_pca: k={pca['k']}, explained {pca['explained']:.3f}, basis finite "
          f"(fallbacks used: {dict(dm.EIGH_FALLBACKS)})")
    c = pca["components"]
    check(float(np.abs(c @ c.T - np.eye(pca["k"])).max()) < 1e-8, "the PC basis is orthonormal")
    dmso = np.flatnonzero(meta["is_control"].values == 1)
    half = dmso.size // 2
    fa, fb = dm.apply_pca(y[dmso[:half]], pca), dm.apply_pca(y[dmso[half:]], pca)
    fr = dm.frechet_distance(fa, fb)
    sa, sb = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    # FD = |du|^2 + trA + trB - 2 tr sqrt(A B) >= 0, since tr sqrt(AB) <= (trA + trB)/2.
    check(np.isfinite(fr) and fr >= 0,
          f"pc_frechet between the two halves of the DMSO wells is finite and >= 0 "
          f"({fr:.3f}; trA {np.trace(sa):.1f}, trB {np.trace(sb):.1f})")
    print(f"      fallbacks used over the whole check: {dict(dm.EIGH_FALLBACKS)}")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILURE(S)"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
