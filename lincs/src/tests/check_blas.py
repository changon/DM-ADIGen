"""Probe numpy's BLAS/LAPACK on this node against loop references (IMPLEMENT.md §5). No PyTorch.

zabih-compute-01 (Xeon Gold 6426Y) gave wrong eigendecompositions under
OpenBLAS's default kernel choice and correct ones with OPENBLAS_CORETYPE=Haswell.
This checks whether plain matrix products are affected too, at the shapes
evaluate.py uses: dgemm (A @ B), dgemv (A @ x, the step-C projection), np.dot,
np.cov, and eigh/svd verified with loop-based (einsum, optimize=False) products.

    python -m src.tests.check_blas
"""
import os
import socket
import sys

import numpy as np

FAILS = []


def check(ok, msg):
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def mm_ref(a, b):
    return np.einsum("ik,kj->ij", a, b, optimize=False)


def rel(x, ref):
    return float(np.abs(x - ref).max() / max(np.abs(ref).max(), 1e-300))


print(f"[blas] host {socket.gethostname()} numpy {np.__version__} "
      f"OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE')} "
      f"threads={os.environ.get('OPENBLAS_NUM_THREADS')}", flush=True)
rng = np.random.default_rng(0)
for n, k, m in ((40, 40, 40), (100, 300, 100), (659, 659, 659), (978, 978, 978),
                (2064, 978, 659), (10446, 978, 1)):
    a, b = rng.normal(size=(n, k)), rng.normal(size=(k, m))
    check(rel(a @ b, mm_ref(a, b)) < 1e-12, f"dgemm {n}x{k} @ {k}x{m}: rel err {rel(a @ b, mm_ref(a, b)):.1e}")
a, x = rng.normal(size=(10446, 978)), rng.normal(size=978)
r = np.einsum("ag,g->a", a, x, optimize=False)
check(rel(a @ x, r) < 1e-12, f"dgemv 10446x978 @ 978 (the step-C projection): rel err {rel(a @ x, r):.1e}")
check(rel(np.array([a[i] @ x for i in range(50)]), r[:50]) < 1e-12, "ddot row-wise")
y = rng.normal(size=(5000, 978))
yc = y - y.mean(0)
cref = np.einsum("ni,nj->ij", yc, yc, optimize=False) / (y.shape[0] - 1)
check(rel(np.cov(y, rowvar=False), cref) < 1e-12, f"np.cov 5000x978: rel err {rel(np.cov(y, rowvar=False), cref):.1e}")
for d in (40, 659, 978):
    s = rng.normal(size=(3 * d, d))
    m = s.T @ s / (3 * d)
    w, v = np.linalg.eigh(m)
    rec = rel(mm_ref(v * w, v.T), m)
    orth = float(np.abs(mm_ref(v.T, v) - np.eye(d)).max())
    check(np.isfinite(w).all() and np.isfinite(v).all() and rec < 1e-10 and orth < 1e-10,
          f"eigh {d}-d: reconstruction {rec:.1e}, orthonormality {orth:.1e}")
    u, sv, vt = np.linalg.svd(m)
    rec = rel(mm_ref(u * sv, vt), m)
    check(np.isfinite(sv).all() and rec < 1e-10, f"svd {d}-d: reconstruction {rec:.1e}")
print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILURE(S)"))
sys.exit(1 if FAILS else 0)
