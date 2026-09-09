"""Build nu_rows_design.npy for a tiered split dir: the UNTHINNED train pool.

nu = population rows minus the holdout minus the reserve wells -- the design
action distribution BEFORE thinning. fit_urr --nu_rows points at it so the
learned alpha targets nu_design(a)/f_thinned(a), i.e. the design weights.
Identical across pg0/pg2 instances of one seed (shared reserve + split).

    python -m src.data.build_nu_rows --nuisance_dir data/nuisances_tier_k2_pg0_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.spec import default_config  # noqa: E402


def build_nu_rows(nuisance_dir: str, out: str = "nu_rows_design.npy") -> str:
    cfg = default_config()
    meta = load_from_disk(cfg.paths.tabular_dataset_dir)
    dis = np.array([str(x) for x in meta["disease_condition"]])
    ct = np.array([str(x) for x in meta["cell_type"]])
    wid = np.array([str(x) for x in meta["well_id"]])
    pop = (dis == str(cfg.population.disease_condition)) \
        & (ct == str(cfg.population.cell_type))

    splits = json.load(open(os.path.join(nuisance_dir, "splits.json")))
    res = json.load(open(os.path.join(nuisance_dir, "reserve.json")))
    hold = np.zeros(len(pop), dtype=bool)
    hold[np.asarray(splits["holdout_idx"], dtype=np.int64)] = True
    rwells = {w for arm in res["arms"] for w in arm["reserve_wells"]}
    in_res = np.isin(wid, sorted(rwells))

    nu = np.where(pop & ~hold & ~in_res)[0]
    dst = os.path.join(nuisance_dir, out)
    np.save(dst, nu)
    n_thin_dropped = len(nu) - len(splits["train_idx"])
    print(f"[nu] wrote {dst}  n={len(nu):,}  "
          f"(train {len(splits['train_idx']):,}; thinning dropped "
          f"{n_thin_dropped:,} rows that stay in nu)")
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nuisance_dir", required=True)
    p.add_argument("--out", default="nu_rows_design.npy")
    a = p.parse_args()
    build_nu_rows(a.nuisance_dir, a.out)


if __name__ == "__main__":
    main()
