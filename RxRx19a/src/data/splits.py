"""Stratified train / holdout split, shared by every downstream stage.

`fit_urr` and `train_diffusion` see only `train_idx`; `evaluate` uses `holdout_idx`
Stratified on joint (disease_condition, compound_idx): each stratum puts `round(holdout_frac * n)` rows to holdout, and other in train

    python -m src.data.splits --holdout-frac 0.10
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spec import (
    CaseConfig, add_adjustment_set_cli, config_from_args, default_config)  # noqa: E402


SPLITS_FILENAME = "splits.json"


def population_filters(cfg) -> dict:
    """{column: required value} for cfg.population. Used to check a cached split and to create a new one."""
    out = {k: str(v) for k, v in (
        ("disease_condition", cfg.population.disease_condition),
        ("cell_type", cfg.population.cell_type)) if v is not None}
    # Part of the cache key, not an equality filter: a split built for a
    # different action space must not be silently reused. Handled separately
    # in the mask loop below.
    if cfg.population.compounds:
        out["compounds"] = f"{len(cfg.population.compounds)}:" + hashlib.sha1(
            "|".join(sorted(cfg.population.compounds)).encode()).hexdigest()[:12]
    return out


def _splits_path(cfg: CaseConfig) -> str:
    return os.path.join(cfg.paths.nuisance_dir, SPLITS_FILENAME)


def _stratified_indices(
    strata_key: np.ndarray, holdout_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-stratum random holdout. `strata_key` is an int array (one stratum id per row); returns (train_idx, holdout_idx) over the original positions."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    holdout_parts: list[np.ndarray] = []
    for s in np.unique(strata_key):
        rows = np.where(strata_key == s)[0]
        rng.shuffle(rows)
        n = rows.shape[0]
        n_hold = int(round(holdout_frac * n))
        if n < 2:
            n_hold = 0  # can't hold out a singleton stratum
        holdout_parts.append(rows[:n_hold])
        train_parts.append(rows[n_hold:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    holdout_idx = np.concatenate(holdout_parts) if holdout_parts else np.array([], dtype=np.int64)
    # Sorted indices are friendlier to HF dataset.select.
    train_idx.sort()
    holdout_idx.sort()
    return train_idx.astype(np.int64), holdout_idx.astype(np.int64)


def make_splits(
    cfg: CaseConfig | None = None,
    holdout_frac: float = 0.20,
    seed: int | None = None,
    force: bool = False,
) -> dict:
    """Compute (or reuse) the stratified split.

    Returns train_idx, holdout_idx, holdout_frac, seed, n_total, n_strata.
    """
    cfg = cfg or default_config()
    seed = cfg.seed if seed is None else seed
    out_path = _splits_path(cfg)

    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            cached = json.load(f)
        # define the pop of interest. not just all samples.
        if (cached.get("holdout_frac") == holdout_frac
                and cached.get("seed") == seed
                and cached.get("population", {}) == population_filters(cfg)):
            return {
                "train_idx": np.asarray(cached["train_idx"], dtype=np.int64),
                "holdout_idx": np.asarray(cached["holdout_idx"], dtype=np.int64),
                "holdout_frac": cached["holdout_frac"],
                "seed": cached["seed"],
                "n_total": cached["n_total"],
                "n_strata": cached["n_strata"],
            }

    ds = load_from_disk(cfg.paths.tabular_dataset_dir)
    n_total = len(ds)

    # Population restriction (cfg.population). Rows outside it never enter train OR holdout; they stay on disk for eval to use as real references.
    pop_filters = population_filters(cfg)
    pop_mask = np.ones(n_total, dtype=bool)
    for col, want in pop_filters.items():
        if col == "compounds":
            continue          # membership, not equality -- applied below
        pop_mask &= (np.array([str(x) for x in ds[col]]) == want)
    if cfg.population.compounds:
        _want = set(cfg.population.compounds)
        _treat = np.array([str(x) for x in ds["treatment"]])
        _isctl = np.asarray(ds["is_control"], dtype=np.int64) == 1
        _keep = np.isin(_treat, list(_want)) | _isctl
        _missing = _want - set(_treat[np.isin(_treat, list(_want))].tolist())
        if _missing:
            raise ValueError(
                f"population.compounds names {len(_missing)} compound(s) absent "
                f"from the dataset, e.g. {sorted(_missing)[:3]}.")
        pop_mask &= _keep
        print(f"[splits] action space restricted to {len(_want)} compounds "
              f"(+controls): {int(_keep.sum()):,} rows pass")
    if pop_filters:
        print(f"[splits] population {pop_filters}: "
              f"{int(pop_mask.sum()):,}/{n_total:,} rows")

    disease = np.array(
        ["<missing>" if d is None else str(d) for d in ds["disease_condition"]]
    )
    compound = np.asarray(ds["compound_idx"], dtype=np.int64)

    # (disease, compound) -> a single int stratum id.
    disease_levels, disease_codes = np.unique(disease, return_inverse=True)
    strata_key = disease_codes.astype(np.int64) * (compound.max() + 1) + compound

    # Stratify within the population only
    pop_rows = np.where(pop_mask)[0]
    _tr, _ho = _stratified_indices(strata_key[pop_rows], holdout_frac, seed)
    train_idx, holdout_idx = pop_rows[_tr], pop_rows[_ho]
    n_strata = int(np.unique(strata_key[pop_rows]).shape[0])

    payload = {
        "holdout_frac": holdout_frac,
        "seed": seed,
        "n_total": n_total,
        "n_strata": n_strata,
        "stratify_by": ["disease_condition", "compound_idx"],
        "population": pop_filters, #pop filters
        "disease_levels": [str(x) for x in disease_levels.tolist()],
        "n_train": int(train_idx.shape[0]),
        "n_holdout": int(holdout_idx.shape[0]),
        "train_idx": train_idx.tolist(),
        "holdout_idx": holdout_idx.tolist(),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[splits] wrote {out_path}: "
          f"n_total={n_total} train={payload['n_train']} "
          f"holdout={payload['n_holdout']} ({holdout_frac:.0%}) "
          f"n_strata={n_strata}")

    return {
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
        "holdout_frac": holdout_frac,
        "seed": seed,
        "n_total": n_total,
        "n_strata": n_strata,
    }


def load_splits(cfg: CaseConfig | None = None) -> dict:
    """Load an existing splits.json. Calls `make_splits` to create one if it doesn't exist yet (using defaults).
    """
    cfg = cfg or default_config()
    out_path = _splits_path(cfg)
    if not os.path.isfile(out_path):
        base = os.path.basename(cfg.paths.nuisance_dir)
        if base.startswith("nuisances_"):
            raise FileNotFoundError(
                f"no splits.json in {cfg.paths.nuisance_dir}. load_splits() will "
                f"not create one in a prebuilt split dir: build it with "
                f"src.data.build_tiered_split.")
        return make_splits(cfg)
    with open(out_path) as f:
        cached = json.load(f)
    return {
        "train_idx": np.asarray(cached["train_idx"], dtype=np.int64),
        "holdout_idx": np.asarray(cached["holdout_idx"], dtype=np.int64),
        "holdout_frac": cached["holdout_frac"],
        "seed": cached["seed"],
        "n_total": cached["n_total"],
        "n_strata": cached["n_strata"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=None,
                   help="Defaults to cfg.seed.")
    p.add_argument("--force", action="store_true")
    add_adjustment_set_cli(p)
    a = p.parse_args()
    cfg = config_from_args(a)
    make_splits(cfg=cfg, holdout_frac=a.holdout_frac, seed=a.seed, force=a.force)


if __name__ == "__main__":
    main()
