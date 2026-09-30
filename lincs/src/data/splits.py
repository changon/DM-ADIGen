"""Stratified train / holdout split, shared by every downstream stage.

Copied from RxRx19a/src/data/splits.py (`load_splits` + the stratification
engine), then adapted (IMPLEMENT.md §3.2 Splits; P3, P4):

  - The table is already filtered, so the population is a CACHE KEY, never an
    equality mask: `population_key` = PopulationSpec.key() + the table
    fingerprint build_dataset recorded. `load_splits` refuses a split whose key
    differs, so a rebuilt table or another compound subset cannot reuse it.
  - Strata: each treated well's arm (cell_id, compound_idx, dose_level), and
    each plate's DMSO wells (so every plate keeps train DMSO for its centre).
    Each stratum of n >= 2 wells holds out round(holdout_frac * n); singletons
    stay in train. At 0.2 a 3-well arm holds out 1 and a 2-well arm none: the
    realised holdout is ~28.5% on mcf7_24h, and it is logged.
  - v1's split is the split layer of build_tiered_split (no reserve, no
    thinning; P4), which writes splits.json. `load_splits` only reads and
    validates; unlike RxRx it never creates one.
  - Each stratum draws from its own generator, seeded by (seed, stratum name),
    not from one stream over all strata: a stratum's holdout depends only on
    its own rows, so reserving wells of scored arms (k_reserve > 0) leaves every
    other stratum's holdout -- DMSO, hence the plate centres, included -- as in v1.

`fit_urr` and `train_diffusion` see only `train_idx`; `evaluate` uses `holdout_idx`.

    python -m src.data.splits [--data_dir ...] [--nuisance_dir ...]   # print a split's summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spec import (  # noqa: E402
    CaseConfig, add_adjustment_set_cli, add_paths_cli, apply_paths_args,
    config_from_args, default_config)

SPLITS_FILENAME = "splits.json"
CONTROL_ARM = "0|ctl"   # every vehicle row: compound_idx 0, no dose


def table_fingerprint(cfg: CaseConfig) -> str:
    """The fingerprint build_dataset recorded for the table under cfg.paths.data_dir."""
    with open(cfg.paths.population_qc_json) as f:
        return json.load(f)["table_fingerprint"]


def population_key(cfg: CaseConfig) -> dict:
    """Cache key of a split: the population plus the table it indexes."""
    return {**cfg.population.key(), "table_fingerprint": table_fingerprint(cfg)}


def _splits_path(cfg: CaseConfig) -> str:
    return os.path.join(cfg.paths.nuisance_dir, SPLITS_FILENAME)


def resolve_split_file(nz: str, path: str) -> str:
    """`path` inside the split dir `nz` when it exists there, else as given."""
    inside = os.path.join(nz, path)
    return inside if not os.path.isabs(path) and os.path.isfile(inside) else path


def _fmt_dose(v) -> str:
    return f"{float(v):.6g}"


def arm_keys(compound_idx, dose_level, is_control) -> np.ndarray:
    """(N,) the dose_level arm of each row, "<compound_idx>|<dose_level>" (P2).

    Every vehicle row is the one arm CONTROL_ARM. `dose_level` (not
    log10_conc) keys the arm, so float variants of a dose (0.37 vs 0.3704 uM)
    are one arm: 10,479 arms on mcf7_24h, not 10,944.
    """
    comp = np.asarray(compound_idx, dtype=np.int64)
    ctl = np.asarray(is_control).astype(bool)
    dl = np.asarray(dose_level, dtype=np.float64)
    return np.array([CONTROL_ARM if k else f"{c}|{_fmt_dose(d)}"
                     for c, d, k in zip(comp, dl, ctl)], dtype=object)


def dose_half(compound_idx, dose_level, is_control) -> np.ndarray:
    """(N,) int8: 1 for the high half of each compound's dose levels, 0 for the
    low half, -1 for vehicles. The levels are the compound's distinct treated
    dose_levels in the table; of n levels the top n - n//2 are high (4-6 of 6;
    with an odd count the middle level is high)."""
    comp = np.asarray(compound_idx, dtype=np.int64)
    ctl = np.asarray(is_control).astype(bool)
    dl = np.asarray(dose_level, dtype=np.float64)
    out = np.full(comp.shape[0], -1, dtype=np.int8)
    for c in np.unique(comp[~ctl]):
        m = (comp == c) & ~ctl
        lv = np.unique(dl[m])
        out[m] = np.searchsorted(lv, dl[m]) >= lv.size // 2
    return out


# The design's positivity cell per confounder (IMPLEMENT.md §3.8.1-3; decision
# 2026-09-30): thinning keeps >= 1 train well in each, the design weight is
# constant within each, and step C / A estimate the DR weights at this key.
POSITIVITY_KEYS = {"syn_c": ("compound", "dose_half", "syn_c"),
                   "cell_id": ("compound", "dose_level", "cell_id")}


def positivity_cells(confounder: str, compound_idx, dose_level, is_control, c_values) -> np.ndarray:
    """(N,) the positivity cell of each row for `confounder`; vehicles are CONTROL_ARM.
        syn_c   -> "<compound_idx>|h<dose half>|syn_c=<v>"
        cell_id -> "<arm>|cell_id=<v>"   (arm = compound_idx|dose_level)"""
    ctl = np.asarray(is_control).astype(bool)
    cv = np.asarray(c_values).astype(str)
    if confounder == "syn_c":
        comp = np.asarray(compound_idx, dtype=np.int64)
        half = dose_half(comp, dose_level, ctl)
        return np.array([CONTROL_ARM if k else f"{c}|h{h}|syn_c={v}"
                         for c, h, v, k in zip(comp, half, cv, ctl)], dtype=object)
    if confounder == "cell_id":
        arm = arm_keys(compound_idx, dose_level, ctl)
        return np.array([CONTROL_ARM if k else f"{a}|cell_id={v}"
                         for a, v, k in zip(arm, cv, ctl)], dtype=object)
    raise ValueError(f"no positivity cell declared for confounder {confounder!r}; have {list(POSITIVITY_KEYS)}")


def split_strata(cell_id, compound_idx, dose_level, is_control, det_plate) -> np.ndarray:
    """(N,) stratum of each row: "arm|<cell_id>|<arm>" for treated wells,
    "dmso|<det_plate>" for vehicles (DMSO stratified by plate, P3). cell_id is
    constant on mcf7_24h; it keeps a multi-line population's arms per line."""
    arm = arm_keys(compound_idx, dose_level, is_control)
    ctl = np.asarray(is_control).astype(bool)
    return np.array([f"dmso|{p}" if k else f"arm|{c}|{a}"
                     for c, a, k, p in zip(cell_id, arm, ctl, det_plate)], dtype=object)


def population_rows(cfg: CaseConfig, pert_id, is_control) -> np.ndarray:
    """(N,) bool: rows in cfg.population. The table is the population, except
    that --population_compounds (pert_ids) keeps only those compounds plus the
    controls."""
    ctl = np.asarray(is_control).astype(bool)
    if not cfg.population.compounds:
        return np.ones(ctl.shape[0], dtype=bool)
    pid = np.asarray(pert_id, dtype=object)
    want = set(cfg.population.compounds)
    missing = want - set(pid[~ctl].tolist())
    if missing:
        raise ValueError(f"population.compounds names {len(missing)} pert_id(s) absent from the "
                         f"table, e.g. {sorted(missing)[:3]}")
    return np.isin(pid, sorted(want)) | ctl


def _stratified_indices(
    strata_key: np.ndarray, holdout_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-stratum random holdout. `strata_key` holds one stratum NAME per row; returns (train_idx, holdout_idx) over the original positions.

    Rows of a stratum are taken in ascending position and shuffled by that
    stratum's own generator, default_rng([seed, crc32(name)]).
    """
    train_parts: list[np.ndarray] = []
    holdout_parts: list[np.ndarray] = []
    uniq, inv, counts = np.unique(strata_key, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    ends = np.cumsum(counts)
    for s in range(uniq.shape[0]):
        rows = order[ends[s] - counts[s]:ends[s]].copy()
        rng = np.random.default_rng([int(seed), zlib.crc32(str(uniq[s]).encode())])
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


def split_layer(strata: np.ndarray, pool_rows: np.ndarray, holdout_frac: float,
                seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Stratified split of `pool_rows` (table row ids). Returns (train_idx,
    holdout_idx, n_strata), both sorted table row ids."""
    pool_rows = np.asarray(pool_rows, dtype=np.int64)
    names = strata[pool_rows]
    tr, ho = _stratified_indices(names, holdout_frac, seed)
    return pool_rows[tr], pool_rows[ho], int(np.unique(names).size)


def split_report(strata: np.ndarray, is_control, train_idx, holdout_idx) -> dict:
    """The realised holdout (P3): overall, treated and DMSO; arms with no holdout
    well and with no train well; train DMSO wells per plate."""
    ctl = np.asarray(is_control).astype(bool)
    n = strata.shape[0]
    tr = np.zeros(n, dtype=bool)
    ho = np.zeros(n, dtype=bool)
    tr[train_idx] = True
    ho[holdout_idx] = True
    used = tr | ho
    keys, inv = np.unique(strata[used], return_inverse=True)
    n_tr = np.bincount(inv, weights=tr[used], minlength=keys.size).astype(np.int64)
    n_ho = np.bincount(inv, weights=ho[used], minlength=keys.size).astype(np.int64)
    is_dmso = np.array([k.startswith("dmso|") for k in keys], dtype=bool)
    dmso_tr = n_tr[is_dmso]
    frac = lambda m: float(ho[used & m].sum() / max(int((used & m).sum()), 1))
    return {
        "n_train": int(tr.sum()), "n_holdout": int(ho.sum()),
        "holdout_frac_realised": frac(np.ones(n, dtype=bool)),
        "holdout_frac_treated": frac(~ctl),
        "holdout_frac_dmso": frac(ctl),
        "n_arms": int((~is_dmso).sum()),
        "n_arms_without_holdout": int(((n_ho == 0) & ~is_dmso).sum()),
        "n_arms_without_train": int(((n_tr == 0) & ~is_dmso).sum()),
        "n_plates": int(is_dmso.sum()),
        "train_dmso_per_plate": {"min": int(dmso_tr.min()) if dmso_tr.size else 0,
                                 "median": float(np.median(dmso_tr)) if dmso_tr.size else 0.0,
                                 "max": int(dmso_tr.max()) if dmso_tr.size else 0},
    }


def split_fingerprint(train_idx, holdout_idx, reserve_idx=()) -> str:
    """Identifies one split; downstream artifacts (expr_meta, folds, weights) record it."""
    h = hashlib.sha1()
    h.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    h.update(b"#holdout#")
    h.update(np.asarray(holdout_idx, dtype=np.int64).tobytes())
    h.update(b"#reserve#")
    h.update(np.asarray(reserve_idx, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def load_splits(cfg: CaseConfig | None = None) -> dict:
    """Load and validate splits.json from cfg.paths.nuisance_dir.

    Raises when it is missing (build it with src.data.build_tiered_split), when
    its population / table key differs from cfg's, or when its indices were
    edited (fingerprint mismatch).
    """
    cfg = cfg or default_config()
    out_path = _splits_path(cfg)
    if not os.path.isfile(out_path):
        raise FileNotFoundError(
            f"no {SPLITS_FILENAME} in {cfg.paths.nuisance_dir}. Build it with "
            f"`python -m src.data.build_tiered_split` (the v1 split is that builder "
            f"with no scored compounds and no thinning).")
    with open(out_path) as f:
        cached = json.load(f)
    want = population_key(cfg)
    if cached.get("population") != want:
        raise ValueError(
            f"{out_path} was built for population {cached.get('population')}, but the "
            f"current config and table are {want}. Rebuild the split with "
            f"src.data.build_tiered_split (into its own --out_dir for a compound subset).")
    train_idx = np.asarray(cached["train_idx"], dtype=np.int64)
    holdout_idx = np.asarray(cached["holdout_idx"], dtype=np.int64)
    reserve_idx = np.asarray(cached.get("reserve_idx", []), dtype=np.int64)
    if split_fingerprint(train_idx, holdout_idx, reserve_idx) != cached["split_fingerprint"]:
        raise ValueError(f"{out_path}: train/holdout/reserve indices do not match the recorded split_fingerprint")
    return {
        "train_idx": train_idx,
        "holdout_idx": holdout_idx,
        "reserve_idx": reserve_idx,
        "holdout_frac": cached["holdout_frac"],
        "seed": cached["seed"],
        "n_total": cached["n_total"],
        "n_strata": cached["n_strata"],
        "population": cached["population"],
        "split_fingerprint": cached["split_fingerprint"],
        "report": cached.get("report", {}),
        "tier": cached.get("tier", {"active": False}),
    }


def main():
    p = argparse.ArgumentParser(description="Print and validate an existing split.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    a = p.parse_args()
    cfg = apply_paths_args(config_from_args(a), a)
    s = load_splits(cfg)
    print(f"[splits] {_splits_path(cfg)}  population {s['population']}")
    print(f"[splits] train {s['train_idx'].size:,}  holdout {s['holdout_idx'].size:,}  "
          f"reserve {s['reserve_idx'].size:,}  of {s['n_total']:,} rows; {s['n_strata']:,} strata; "
          f"seed {s['seed']}; tier active={s['tier'].get('active')}")
    print(f"[splits] report {json.dumps(s['report'])}")


if __name__ == "__main__":
    main()
