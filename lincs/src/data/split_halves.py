"""Two training halves of a thinning instance (POLICY_LEARNING.md §6, P3). No PyTorch.

Algorithm 1 of the paper trains one generator per half of the data and
retargets each half's generator toward the pilot policy learned on the other
half. This writes `<tier>_h1` and `<tier>_h2`: split dirs whose `train_idx` is
one half of the instance's KEPT train wells, everything else inherited:

  - the halves partition the kept train wells, stratified on (compound, line,
    dose) for treated wells and (plate, line) for vehicles: a stratum's wells
    alternate between the halves after a seeded shuffle, so a 2-well cell gives
    one well to each half and a 1-well cell goes to one half at random;
  - the holdout (and reserve) are the parent's, so a half scores on the same
    pool and trains on no holdout row;
  - `expr_meta.json` keeps the parent's z-scale under the half's fingerprint
    (the tier dirs already share the base split's scale);
  - every `dr_weights_*.npz` of the parent is subset to the half's rows. The
    counts weights are per-cell ratios of the parent (n_unthinned / n_kept), and
    the halving is stratified by cell, so the ratios between cells are
    preserved in expectation; `--dr_weight_norm group` re-balances within each
    group exactly in any case;
  - `nu_rows.npy`, `tier_meta.json`, the vocab and the encoders are copied, so
    `check_arm_against_data` recovers the unthinned base split from a half.

    python -m src.data.split_halves --data_dir data/core5_24h --nuisance_dir <tier dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import _atomic_write, _write_json  # noqa: E402
from src.data.build_nu_rows import NU_ROWS_FILENAME  # noqa: E402
from src.data.build_tiered_split import COPIED_FROM_BASE  # noqa: E402
from src.data.expr_stats import EXPR_META  # noqa: E402
from src.data.splits import SPLITS_FILENAME, load_splits, split_fingerprint  # noqa: E402
from src.spec import add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args  # noqa: E402

N_HALVES = 2


def half_dir(tier_dir: str, j: int) -> str:
    """The one definition of a half's name: <tier>_h<j>, j in 1..N_HALVES."""
    if not 1 <= int(j) <= N_HALVES:
        raise ValueError(f"half {j}: 1..{N_HALVES}")
    return os.path.normpath(tier_dir) + f"_h{int(j)}"


def assign_halves(train_idx: np.ndarray, strata: np.ndarray, seed: int) -> np.ndarray:
    """(n_train,) the half (1-based) of each train row: within each stratum the
    rows are shuffled and dealt in turn, from a random starting half."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    strata = np.asarray(strata).astype(str)
    if strata.shape[0] != train_idx.shape[0]:
        raise ValueError("strata must be aligned to train_idx")
    rng = np.random.default_rng([int(seed), 7_171])
    out = np.zeros(train_idx.shape[0], dtype=np.int8)
    order = np.argsort(strata, kind="stable")
    ss = strata[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[starts[1:], ss.size]
    for a, b in zip(starts, ends):
        rows = order[a:b]
        perm = rng.permutation(rows.size)
        start = int(rng.integers(N_HALVES))
        out[rows[perm]] = 1 + (start + np.arange(rows.size)) % N_HALVES
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0, help="Seed of the within-stratum deal.")
    p.add_argument("--overwrite", action="store_true", help="Replace halves built from another parent or seed.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    parent = os.path.normpath(cfg.paths.nuisance_dir)
    sp = load_splits(cfg)                                   # validates the parent
    if not sp["tier"].get("active"):
        raise SystemExit(f"[halves] {parent} is not a thinning instance")
    if sp["tier"].get("half"):
        raise SystemExit(f"[halves] {parent} is itself a half")
    with open(os.path.join(parent, SPLITS_FILENAME)) as fh:
        payload = json.load(fh)
    from datasets import load_from_disk
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "cell_id", "det_plate"]).to_pandas())
    if len(meta) != sp["n_total"]:
        raise SystemExit(f"[halves] table has {len(meta):,} rows; the split was built on {sp['n_total']:,}")
    tr = sp["train_idx"]
    comp = meta["compound_idx"].values.astype(np.int64)[tr]
    dl = meta["dose_level"].values.astype(np.float64)[tr]
    ctl = meta["is_control"].values.astype(bool)[tr]
    line = meta["cell_id"].values.astype(str)[tr]
    plate = meta["det_plate"].values.astype(str)[tr]
    strata = np.where(ctl, np.char.add("veh|", np.char.add(plate, np.char.add("|", line))),
                      np.array([f"{c}|{d:.6g}|{l}" for c, d, l in zip(comp, dl, line)]))
    half_of = assign_halves(tr, strata, args.seed)
    n_str = np.unique(strata).size
    print(f"[halves] {parent}: {tr.size:,} kept train rows in {n_str:,} strata -> "
          + ", ".join(f"h{j} {int((half_of == j).sum()):,}" for j in range(1, N_HALVES + 1)), flush=True)

    weight_files = sorted(f for f in os.listdir(parent) if f.startswith("dr_weights_") and f.endswith(".npz"))
    for j in range(1, N_HALVES + 1):
        out = half_dir(parent, j)
        idx = tr[half_of == j]                              # sorted, since tr is sorted
        fp = split_fingerprint(idx, sp["holdout_idx"], sp["reserve_idx"])
        existing = os.path.join(out, SPLITS_FILENAME)
        if os.path.isfile(existing):
            with open(existing) as fh:
                old = json.load(fh)
            if old.get("split_fingerprint") == fp and not args.overwrite:
                print(f"[halves] {out} already holds this half; nothing to do")
                continue
            if not args.overwrite:
                raise SystemExit(f"[halves] {out} holds another half ({old.get('split_fingerprint')}); pass --overwrite")
        os.makedirs(out, exist_ok=True)
        doc = dict(payload)
        doc["train_idx"] = idx.tolist()
        doc["n_train"] = int(idx.size)
        doc["split_fingerprint"] = fp
        doc["tier"] = dict(payload["tier"], half={"index": j, "of": N_HALVES, "seed": int(args.seed),
                                                   "parent": parent, "parent_split_fingerprint": sp["split_fingerprint"],
                                                   "strata": "treated: (compound, dose_level, cell_id); vehicle: (det_plate, cell_id)"})
        for fn in COPIED_FROM_BASE + ("tier_meta.json", NU_ROWS_FILENAME):
            src = os.path.join(parent, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out, fn))
        for fn in os.listdir(parent):                       # the parent's expr_meta (any plate_center)
            if fn.startswith(EXPR_META[:-5]) and fn.endswith(".json"):
                with open(os.path.join(parent, fn)) as fh:
                    em = json.load(fh)
                if em.get("split_fingerprint") != sp["split_fingerprint"]:
                    raise SystemExit(f"[halves] {fn} in {parent} is for split {em.get('split_fingerprint')}, "
                                     f"not the parent's {sp['split_fingerprint']}")
                em["split_fingerprint"] = fp
                em["z_scale_from"] = {"split_fingerprint": sp["split_fingerprint"], "dir": parent}
                _write_json(os.path.join(out, fn), em)
        pos = {int(r): i for i, r in enumerate(tr)}
        take = np.array([pos[int(r)] for r in idx], dtype=np.int64)
        for fn in weight_files:
            z = np.load(os.path.join(parent, fn), allow_pickle=False)
            if not np.array_equal(z["row_id"].astype(np.int64), tr):
                raise SystemExit(f"[halves] {fn} in {parent} is not aligned to its train_idx")
            extra = {k: z[k] for k in z.files if k not in ("row_id", "w", "split_fingerprint")}
            w = z["w"][take]
            _atomic_write(os.path.join(out, fn),
                          lambda f, w=w, extra=extra: np.savez(f, row_id=idx, w=w, split_fingerprint=np.array(fp), **extra),
                          mode="wb")
        _atomic_write(os.path.join(out, SPLITS_FILENAME), lambda f, doc=doc: json.dump(doc, f))
        print(f"[halves] wrote {out}: train {idx.size:,}, fingerprint {fp}, weights {weight_files}", flush=True)


if __name__ == "__main__":
    main()
