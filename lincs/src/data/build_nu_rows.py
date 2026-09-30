"""Build nu_rows.npy for a split dir: the UNTHINNED train pool.

Rewritten from RxRx19a/src/data/build_nu_rows.py (IMPLEMENT.md §3.5, §3.8.1).
nu = population rows minus the holdout minus the reserve wells: the design
action distribution BEFORE thinning. `export_urr_weights --mode counts` reads
it, so in a thinning instance w = n_nu / n_kept per positivity cell is the
post-stratified design weight (P12); `expr_stats` fits its statistics on it;
`fit_urr --nu_rows` uses it in v1. In v1 (no reserve, no thinning) nu is
exactly the train rows. build_tiered_split calls this.

    python -m src.data.build_nu_rows [--data_dir ...] --nuisance_dir data/mcf7_24h/nuisances
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

from src.data.splits import load_splits, population_rows  # noqa: E402
from src.spec import (  # noqa: E402
    CaseConfig, add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

NU_ROWS_FILENAME = "nu_rows.npy"


def nu_rows(cfg: CaseConfig) -> np.ndarray:
    """Sorted table row ids of nu for the split in cfg.paths.nuisance_dir."""
    from datasets import load_from_disk
    s = load_splits(cfg)
    meta = load_from_disk(cfg.paths.tabular_dataset_dir).select_columns(["pert_id", "is_control"])
    pop = population_rows(cfg, meta["pert_id"], meta["is_control"])
    if len(pop) != s["n_total"]:
        raise ValueError(f"table has {len(pop):,} rows; the split was built on {s['n_total']:,}")
    keep = pop.copy()
    keep[s["holdout_idx"]] = False
    keep[s["reserve_idx"]] = False
    nu = np.flatnonzero(keep).astype(np.int64)
    if not np.isin(s["train_idx"], nu).all():
        raise AssertionError("train rows outside nu: the split is not a subset of population - holdout - reserve")
    return nu


def build_nu_rows(cfg: CaseConfig, out: str = NU_ROWS_FILENAME) -> str:
    nu = nu_rows(cfg)
    n_train = load_splits(cfg)["train_idx"].size
    dst = os.path.join(cfg.paths.nuisance_dir, out)
    np.save(dst, nu)
    print(f"[nu] wrote {dst}  n={nu.size:,}  (train {n_train:,}; thinning dropped "
          f"{nu.size - n_train:,} rows that stay in nu)")
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=NU_ROWS_FILENAME)
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    a = p.parse_args()
    cfg = apply_paths_args(config_from_args(a), a)
    build_nu_rows(cfg, a.out)


if __name__ == "__main__":
    main()
