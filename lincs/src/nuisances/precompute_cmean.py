"""Precompute mu_hat(a) = E[y | a] in the trainer's z-space, for the tau=0 conditional-mean auxiliary loss (P9).

TRAINING ROWS ONLY.

Copied from RxRx19a/src/nuisances/precompute_cmean.py, then adapted
(IMPLEMENT.md §3.5, P2, P9):
  - y is the plate-centred, z-scored landmark vector, computed by
    `expr_stats.normalize_expr` from the split's expr_meta, the same function
    LincsDataset applies (not VAE latents). numpy only: no torch.
  - An action is the dose_level arm (`splits.arm_keys`, P2); every vehicle row
    is one key. RxRx keyed on round(log10_conc, 3), which splits float
    variants of one dose into separate arms.
  - --min_n is configurable; the RxRx default of 8 keeps 34 of 10,479 arms
    on mcf7_24h (plus the vehicle key). Coverage is logged.
  - row_gid: each TRAIN row's index into `keys` (-1 = its arm has < min_n
    train rows), so the trainer reads the grouping instead of re-deriving the
    key (RxRx re-implemented group_key inside the trainer).

    python -m src.nuisances.precompute_cmean [--min_n 8] [--plate_center ...] [--data_dir ...] [--nuisance_dir ...]

Writes <nuisance_dir>/<out>: keys (K,) str, mu (K, G) float32, counts (K,),
train_idx, row_gid (n_train,), min_n, plate_center, split_fingerprint. The
trainer (--cmean_lambda > 0) checks train_idx, split_fingerprint and plate_center.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import _atomic_write  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import CONTROL_ARM, arm_keys, load_splits  # noqa: E402
from src.spec import (  # noqa: E402
    PLATE_CENTER_MODES, add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)


def group_means(y: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(uniq keys, per-key mean of y (K, G) float64, counts) over the rows of y."""
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    sums = np.add.reduceat(y[order].astype(np.float64), starts, axis=0)
    return uniq, sums / counts[:, None], counts


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="cmean.npz")
    p.add_argument("--min_n", type=int, default=8, help="Skip arms with fewer than this many TRAIN rows -- their mean is noisier than the effect it supervises. RxRx default; the Phase 4 ablation picks the value used.")
    p.add_argument("--plate_center", default=None, choices=PLATE_CENTER_MODES, help="Default: OutcomeSpec.plate_center. Must match the trainer's --plate_center.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    a = p.parse_args()
    if a.min_n < 1:
        p.error("--min_n must be >= 1")
    cfg = apply_paths_args(config_from_args(a), a)
    if cfg.outcome.syn_effect != 0:
        raise SystemExit("[cmean] syn_effect != 0: the step-C injection lands in Phase 5 (IMPLEMENT.md §3.8.2); "
                         "mu_hat must then include it, as LincsDataset will")
    plate_center = a.plate_center or cfg.outcome.plate_center

    splits = load_splits(cfg)
    train_idx = np.asarray(splits["train_idx"], dtype=np.int64)
    if not np.all(np.diff(train_idx) > 0):
        raise ValueError("train_idx must be sorted ascending and unique")
    m = load_expr_meta(cfg, plate_center, splits)

    meta = load_from_disk(cfg.paths.tabular_dataset_dir).select_columns(
        ["compound_idx", "dose_level", "is_control", "det_plate"])
    comp = np.asarray(meta["compound_idx"], dtype=np.int64)
    dl = np.asarray(meta["dose_level"], dtype=np.float64)
    ic = np.asarray(meta["is_control"]).astype(bool)
    plates = np.asarray(meta["det_plate"], dtype=object)
    expr = np.load(cfg.paths.expr_npy, mmap_mode="r")
    G = cfg.outcome.n_genes
    if expr.shape != (comp.size, G):
        raise ValueError(f"expr.npy {expr.shape} does not match the table ({comp.size:,} x {G})")

    # y in the trainer's z-space, TRAIN rows only
    y = normalize_expr(np.asarray(expr[train_idx]), plate_codes(plates[train_idx], m["plates"]), m)
    keys_tr = arm_keys(comp, dl, ic)[train_idx].astype(str)
    uniq, mu, counts = group_means(y, keys_tr)
    print(f"[cmean] plate_center={plate_center}  train rows {train_idx.size:,}  arms {uniq.size:,} "
          f"(incl. the vehicle key)  n/arm min {counts.min()} median {int(np.median(counts))} max {counts.max()}",
          flush=True)

    keep = counts >= a.min_n
    remap = np.full(uniq.size, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    inv = np.searchsorted(uniq, keys_tr)
    row_gid = remap[inv]

    is_ctl_key = uniq == CONTROL_ARM
    ic_tr = ic[train_idx]
    cov = row_gid >= 0
    print(f"[cmean] --min_n {a.min_n}: keeping {int((keep & ~is_ctl_key).sum()):,}/{int((~is_ctl_key).sum()):,} "
          f"treated arms{' + the vehicle key' if bool((keep & is_ctl_key).any()) else ''}; train rows covered "
          f"{100.0 * cov.mean():.1f}% (treated {100.0 * cov[~ic_tr].mean():.1f}%, "
          f"DMSO {100.0 * cov[ic_tr].mean():.1f}%)", flush=True)
    if bool((keep & is_ctl_key).any()):
        vm = np.abs(mu[is_ctl_key][0]).max()
        print(f"[cmean] vehicle mu max |.| {vm:.2e} (0 by construction when the z-mean is the train DMSO mean)",
              flush=True)

    dst = os.path.join(cfg.paths.nuisance_dir, a.out)
    _atomic_write(dst, lambda f: np.savez(
        f, keys=uniq[keep], mu=mu[keep].astype(np.float32), counts=counts[keep], train_idx=train_idx,
        row_gid=row_gid, min_n=np.array(a.min_n), plate_center=np.array(plate_center),
        split_fingerprint=np.array(splits["split_fingerprint"])), mode="wb")
    print(f"[cmean] wrote {dst}  mu {mu[keep].shape}", flush=True)


if __name__ == "__main__":
    main()
