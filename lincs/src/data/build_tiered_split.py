#!/usr/bin/env python3
"""Build a split dir, drawn in three independent layers (IMPLEMENT.md §3.8.1; P4, P6).

Rewritten from RxRx19a/src/data/build_tiered_split.py: the three layers and
`_calibrate_pi` are kept; RxRx's panel, outcome-derived position score and
image arms are replaced.

  1. RESERVE (--k_reserve, default 0): k wells per scored arm, drawn before
     anything else and never trained on (out-of-sample truth for high-n arms).
  2. SPLIT: src.data.splits on the rest (arm strata, DMSO by plate,
     --holdout_frac 0.2). Each stratum draws from its own (seed, stratum)
     generator, so every instance of one seed shares v1's holdout on every
     unscored stratum (DMSO included), and at k_reserve = 0 on all of them.
  3. THINNING: for each scored compound, every train well i gets retention
     pi_i ~ exp(-gamma z_i), calibrated to sum(pi) = keep_frac * n and clipped
     to [pmin, 1]. Survivors are redrawn until every positivity cell keeps >= 1
     train well. gamma = 0 is the uniform (MCAR) control at the same keep_frac.
     Design weight: the redraw conditions on "cell c kept >= 1", so well i is
     kept with probability pi_i / P_c, P_c = 1 - prod_{j in c} (1 - pi_j) (the
     cells are independent). Kept wells get the design weight P_c / pi_i, the
     inverse of that realised inclusion probability; every other row 1. (The
     plan's 1/pi ignores the conditioning and would over-weight small
     low-pi cells up to ~4x at gamma = 1.)

Selection score, within a scored compound: z = standardize(z_dose * z_C), with
z_dose = +1 / -1 for the high / low half of the compound's dose levels (levels
4-6 vs 1-3 of 6; with an odd count the middle level is in the high half) and
z_C = +1 / -1 for the two levels of C. A scored compound needs >= 2 dose levels,
so the single-dose 20 uM proteasome plate controls cannot be scored. Positivity cells:
    syn_c   (step C): (compound, dose half, syn_c)
    cell_id (step A): (compound, dose_level, cell_id); needs the line groups
                      declared in spec.py (Phase 6), so it raises for now.

v1 = no scored compounds, hence no reserve and no thinning: splits.json and
nu_rows.npy in the base nuisance dir. A thinning instance writes
<nuisance_dir>_tier_C<c>_k<k>_g<gamma>_s<seed>/ with splits.json,
dr_weights_design.npz, nu_rows.npy, tier_meta.json and copies of
nuisance_meta.json, compound_vocab.json, covariate_encoder.json. Scored
compounds come from --scored_compounds (Phase 5 passes the responders); the
--plan bias table needs the Phase 4 oracle and lands in Phase 5.

    python -m src.data.build_tiered_split                          # v1 -> data/mcf7_24h/nuisances/
    python -m src.data.build_tiered_split --scored_compounds responders.json \\
        --confounder syn_c --gamma 1 --keep_frac 0.4               # a step-C instance
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import _atomic_write, _write_json  # noqa: E402
from src.data.build_nu_rows import NU_ROWS_FILENAME, build_nu_rows  # noqa: E402
from src.data.splits import (  # noqa: E402
    POSITIVITY_KEYS, SPLITS_FILENAME, arm_keys, dose_half, population_key, population_rows,
    split_fingerprint, split_layer, split_report, split_strata)
from src.spec import (  # noqa: E402
    CaseConfig, add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

TABLE_COLS = ["pert_id", "pert_iname", "cell_id", "compound_idx", "is_control",
              "dose_level", "det_plate", "syn_c"]
COPIED_FROM_BASE = ("nuisance_meta.json", "compound_vocab.json", "covariate_encoder.json")
WRITTEN = (SPLITS_FILENAME, NU_ROWS_FILENAME, "dr_weights_design.npz", "tier_meta.json") + COPIED_FROM_BASE
CONFOUNDERS = ("syn_c", "cell_id")
DEFAULT_KEEP_FRAC = {"syn_c": 0.4, "cell_id": 0.6}   # §3.8.2 / §3.8.3
RESERVE_MIN_EXTRA = 3                                # a reserved arm keeps >= 3 wells for the split


def _calibrate_pi(q: np.ndarray, m: float, pmin: float) -> np.ndarray:
    """pi_i proportional to q_i with sum(pi) = m, clipped to [pmin, 1]."""
    pi = np.full(q.shape, np.nan)
    free = np.ones(q.shape, dtype=bool)
    for _ in range(50):
        clipped_mass = np.where(~free, pi, 0.0).sum()
        rem = m - clipped_mass
        if rem <= 0 or not free.any():
            break
        scale = rem / q[free].sum()
        cand = q * scale
        newly = free & ((cand < pmin) | (cand > 1.0))
        pi = np.where(free, np.clip(cand, pmin, 1.0), pi)
        if not newly.any():
            break
        free = free & ~newly
    return np.clip(pi, pmin, 1.0)


def read_compound_list(raw: str) -> list[str]:
    """pert_ids from a .json file (a list, or {"compounds": [...]}), a text file
    (one per line), or a comma-separated string."""
    if os.path.isfile(raw):
        if raw.endswith(".json"):
            with open(raw) as f:
                obj = json.load(f)
            names = obj["compounds"] if isinstance(obj, dict) else obj
        else:
            with open(raw) as f:
                names = [ln.strip() for ln in f]
    else:
        names = raw.split(",")
    return sorted({str(n).strip() for n in names if str(n).strip()})


def standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x, dtype=np.float64)


def default_out_dir(cfg: CaseConfig, base: str, scored: list[str], confounder: str | None,
                    k: int, gamma: float, seed: int) -> str:
    """v1 -> the base nuisance dir; anything else gets its own tagged sibling dir."""
    tag = ""
    if cfg.population.compounds:
        tag += "_pop" + cfg.population.key()["compounds"].replace(":", "-")
    if scored:
        tag += f"_tier_C{confounder}_k{k}_g{gamma:g}_s{seed}"
    elif seed != cfg.seed:
        tag += f"_s{seed}"
    return os.path.normpath(base) + tag


def thin_compound(rows: np.ndarray, half: np.ndarray, c_val: np.ndarray,
                  confounder: str, gamma: float, keep_frac: float, pmin: float,
                  rng: np.random.Generator, max_redraws: int) -> dict:
    """Layer 3 for one scored compound: pi, the kept mask, and its positivity cells.
    `half` is splits.dose_half over the table (1 = high half of the compound's levels)."""
    high = half[rows] == 1
    if confounder != "syn_c":
        raise NotImplementedError(
            f"--confounder {confounder}: the step-A line groups (z_C) and (compound, dose_level, "
            f"cell_id) positivity cells are declared in spec.py in Phase 6")
    cv = c_val[rows].astype(np.int64)
    z = standardize(np.where(high, 1.0, -1.0) * np.where(cv == 1, 1.0, -1.0))
    pi = _calibrate_pi(np.exp(-gamma * z), keep_frac * rows.size, pmin)
    cells = np.array([f"{'high' if h else 'low'}|syn_c={c}" for h, c in zip(high, cv)])
    want = {f"{h}|syn_c={c}" for h in ("low", "high") for c in (0, 1)}
    missing = sorted(want - set(cells.tolist()))
    if missing:
        raise ValueError(f"positivity cells {missing} have no train well before thinning")
    for attempt in range(1, max_redraws + 1):
        kept = rng.random(rows.size) < pi
        if all(kept[cells == c].any() for c in want):
            break
    else:
        raise RuntimeError(f"no draw in {max_redraws} kept >= 1 train well in every positivity cell")
    # realised inclusion probability under the redraw: pi_i / P(cell of i keeps >= 1)
    p_cell = {c: 1.0 - float(np.prod(1.0 - pi[cells == c])) for c in want}
    incl = pi / np.array([p_cell[c] for c in cells])
    return {"pi": pi, "incl": incl, "p_cell": p_cell, "kept": kept, "cells": cells, "z": z, "n_draws": attempt}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=None, help="Defaults to cfg.seed.")
    ap.add_argument("--k_reserve", type=int, default=0, help="Wells reserved per scored arm (default 0: the truth is the full-pool oracle).")
    ap.add_argument("--scored_compounds", default=None, help="pert_ids to thin (json / text file or comma list). Omit for v1.")
    ap.add_argument("--confounder", choices=CONFOUNDERS, default=None, help="C the thinning selects on (required with --scored_compounds).")
    ap.add_argument("--gamma", type=float, default=0.0, help="Selection strength in exp(-gamma * z); 0 = uniform (MCAR) control.")
    ap.add_argument("--keep_frac", type=float, default=None, help="Expected kept fraction of a scored compound's train wells (default 0.4 syn_c, 0.6 cell_id).")
    ap.add_argument("--pmin", type=float, default=0.05, help="Positivity floor on per-well retention probability.")
    ap.add_argument("--max_redraws", type=int, default=1000)
    ap.add_argument("--out_dir", default=None, help="Default: the base nuisance dir for v1, else <nuisance_dir>_tier_C<c>_k<k>_g<gamma>_s<seed>.")
    ap.add_argument("--overwrite", action="store_true", help="Replace a split built with other parameters (refused while downstream artifacts exist).")
    add_adjustment_set_cli(ap)
    add_paths_cli(ap)
    args = ap.parse_args()

    cfg = apply_paths_args(config_from_args(args), args, require_splits=False)
    seed = cfg.seed if args.seed is None else args.seed
    base_nz = cfg.paths.nuisance_dir
    scored = read_compound_list(args.scored_compounds) if args.scored_compounds else []
    if scored and args.confounder is None:
        ap.error("--scored_compounds needs --confounder")
    if not scored and (args.k_reserve or args.confounder or args.gamma):
        ap.error("--k_reserve / --confounder / --gamma act on scored compounds; pass --scored_compounds")
    confounder = args.confounder if scored else None
    keep_frac = (args.keep_frac if args.keep_frac is not None else DEFAULT_KEEP_FRAC[confounder]) if scored else None
    if scored and not 0 < keep_frac < 1:
        ap.error("--keep_frac must be in (0, 1)")
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else default_out_dir(
        cfg, base_nz, scored, confounder, args.k_reserve, args.gamma, seed)

    params = {"population": population_key(cfg), "holdout_frac": args.holdout_frac, "seed": seed,
              "k_reserve": args.k_reserve, "scored_compounds": scored, "confounder": confounder,
              "gamma": args.gamma if scored else None, "keep_frac": keep_frac,
              "pmin": args.pmin if scored else None}
    sp = os.path.join(out_dir, SPLITS_FILENAME)
    if os.path.isfile(sp):
        with open(sp) as f:
            old = json.load(f).get("params")
        if old == json.loads(json.dumps(params)) and not args.overwrite:
            print(f"[tier] {sp} already holds this split (same parameters); nothing to do")
            if not os.path.isfile(os.path.join(out_dir, NU_ROWS_FILENAME)):
                cfg.paths.nuisance_dir = out_dir
                build_nu_rows(cfg)
            return
        if not args.overwrite:
            raise SystemExit(f"[tier] {sp} holds a split with other parameters ({old}); pass --overwrite")
        downstream = sorted(f for f in os.listdir(out_dir) if f not in WRITTEN and not f.endswith(".tmp"))
        if downstream:
            raise SystemExit(f"[tier] {out_dir} holds artifacts built on the old split ({downstream[:6]}); "
                             f"move or delete them before rebuilding the split")

    # --- table -------------------------------------------------------------
    from datasets import load_from_disk
    df = load_from_disk(cfg.paths.tabular_dataset_dir).select_columns(TABLE_COLS).to_pandas()
    n_total = len(df)
    ctl = df["is_control"].values.astype(bool)
    comp = df["compound_idx"].values.astype(np.int64)
    dose_level = df["dose_level"].values.astype(np.float64)
    pop = population_rows(cfg, df["pert_id"].values, ctl)
    strata = split_strata(df["cell_id"].values, comp, dose_level, ctl, df["det_plate"].values)
    arm = arm_keys(comp, dose_level, ctl)
    print(f"[tier] population {params['population']}: {int(pop.sum()):,}/{n_total:,} rows -> {out_dir}")

    with open(os.path.join(base_nz, "compound_vocab.json")) as f:
        vocab = json.load(f)
    unknown = [c for c in scored if c not in vocab]
    if unknown:
        raise ValueError(f"{len(unknown)} scored pert_ids are not in the compound vocab, e.g. {unknown[:3]}")
    scored_ci = sorted(int(vocab[c]) for c in scored)
    scored_mask = np.isin(comp, scored_ci) & ~ctl & pop
    if scored and not np.isin(scored_ci, comp[pop & ~ctl]).all():
        raise ValueError("a scored compound has no treated row in the population")

    # --- LAYER 1: reserve --------------------------------------------------
    res_rng = np.random.default_rng(seed + 10_000)
    reserve_idx = np.array([], dtype=np.int64)
    reserve_arms = []
    if args.k_reserve > 0:
        parts = []
        for a in sorted(set(arm[scored_mask])):
            rows = np.flatnonzero(scored_mask & (arm == a))
            if rows.size < args.k_reserve + RESERVE_MIN_EXTRA:
                raise ValueError(f"scored arm {a} has {rows.size} wells; --k_reserve {args.k_reserve} "
                                 f"needs >= {args.k_reserve + RESERVE_MIN_EXTRA}")
            r = np.sort(res_rng.choice(rows, args.k_reserve, replace=False))
            parts.append(r)
            reserve_arms.append({"arm": a, "rows": r.tolist()})
        reserve_idx = np.sort(np.concatenate(parts)).astype(np.int64)
        print(f"[tier] reserve: {reserve_idx.size} wells locked out of training "
              f"(k={args.k_reserve} x {len(reserve_arms)} arms)")

    # --- LAYER 2: split ----------------------------------------------------
    pool = pop.copy()
    pool[reserve_idx] = False
    train_idx, holdout_idx, n_strata = split_layer(strata, np.flatnonzero(pool), args.holdout_frac, seed)
    report = split_report(strata, ctl, train_idx, holdout_idx)
    print(f"[tier] split: train {train_idx.size:,}  holdout {holdout_idx.size:,}  ({n_strata:,} strata)  "
          f"realised holdout {report['holdout_frac_realised']:.3f} (target {args.holdout_frac}; treated "
          f"{report['holdout_frac_treated']:.3f}, DMSO {report['holdout_frac_dmso']:.3f}); "
          f"{report['n_arms_without_holdout']:,}/{report['n_arms']:,} arms have no holdout well; "
          f"train DMSO per plate {report['train_dmso_per_plate']}")
    if report["n_arms_without_train"]:
        raise AssertionError(f"{report['n_arms_without_train']} arms have no train well")

    # --- LAYER 3: thinning -------------------------------------------------
    w_train = np.ones(train_idx.size, dtype=np.float32)
    final_train = train_idx
    tier_arms = []
    if scored:
        thin_rng = np.random.default_rng(seed + 20_000)
        in_train = np.zeros(n_total, dtype=bool)
        in_train[train_idx] = True
        c_val = df[confounder].values
        half = dose_half(comp, dose_level, ctl)
        drop = np.zeros(n_total, dtype=bool)
        w_of = {}
        inv_vocab = {v: k for k, v in vocab.items()}
        for ci in scored_ci:
            rows = np.flatnonzero(in_train & scored_mask & (comp == ci))
            levels = sorted(set(dose_level[scored_mask & (comp == ci)].tolist()))
            if len(levels) < 2:
                raise ValueError(f"scored compound {inv_vocab[ci]} has {len(levels)} dose level(s); "
                                 f"the dose-half selection needs >= 2")
            try:
                t = thin_compound(rows, half, c_val, confounder, args.gamma,
                                  keep_frac, args.pmin, thin_rng, args.max_redraws)
            except (ValueError, RuntimeError) as e:
                raise type(e)(f"scored compound {inv_vocab[ci]}: {e}") from None
            drop[rows[~t["kept"]]] = True
            w_of.update({int(r): 1.0 / float(q) for r, q, k in zip(rows, t["incl"], t["kept"]) if k})
            cells = {c: [int((t["cells"] == c).sum()), int(((t["cells"] == c) & t["kept"]).sum())]
                     for c in sorted(set(t["cells"].tolist()))}
            tier_arms.append({
                "pert_id": inv_vocab[ci], "pert_iname": str(df.loc[comp == ci, "pert_iname"].iat[0]),
                "compound_idx": ci, "dose_levels": levels, "n_train": int(rows.size),
                "n_kept": int(t["kept"].sum()), "sum_pi": float(t["pi"].sum()),
                "expected_kept": float(t["incl"].sum()),
                "pi_min": float(t["pi"].min()), "pi_max": float(t["pi"].max()), "n_draws": t["n_draws"],
                "cells_train_kept": cells, "p_cell": t["p_cell"],
                "rows": rows.tolist(), "z": [float(v) for v in t["z"]], "pi": [float(p) for p in t["pi"]],
                "incl": [float(q) for q in t["incl"]], "kept": t["kept"].astype(int).tolist()})
        final_train = train_idx[~drop[train_idx]]
        w_train = np.array([w_of.get(int(r), 1.0) for r in final_train], dtype=np.float32)
        n_sc = sum(a["n_train"] for a in tier_arms)
        n_kept = sum(a["n_kept"] for a in tier_arms)
        n_exp = sum(a["expected_kept"] for a in tier_arms)
        ess_n = float(w_train.sum() ** 2 / np.square(w_train).sum() / w_train.size)
        print(f"[tier] thinning ({confounder}, gamma={args.gamma:g}, keep_frac={keep_frac}, pmin={args.pmin}): "
              f"{len(scored_ci)} compounds, kept {n_kept:,}/{n_sc:,} train wells ({n_kept / max(n_sc, 1):.3f}; "
              f"expected {n_exp / max(n_sc, 1):.3f} under the positivity redraw); "
              f"final train {final_train.size:,}; {int((w_train != 1).sum()):,} rows carry design weights, "
              f"w max {w_train.max():.1f}, ESS/n {ess_n:.3f}")

    # --- write -------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    tier = {"active": bool(scored), "k_reserve": args.k_reserve}
    if scored:
        tier.update({
            "confounder": confounder, "gamma": args.gamma, "keep_frac": keep_frac, "pmin": args.pmin,
            "selection": "z = standardize(z_dose * z_C) within compound; z_dose = +/-1 high/low dose half, z_C = +/-1",
            "positivity_cells": "(compound, dose half, syn_c)",
            "positivity_key": list(POSITIVITY_KEYS[confounder]),
            "scored_compounds": {a["pert_id"]: a["compound_idx"] for a in tier_arms},
            "n_thinned": int(train_idx.size - final_train.size),
        })
    payload = {
        "params": params,
        "population": params["population"],
        "holdout_frac": args.holdout_frac, "seed": seed, "n_total": n_total, "n_strata": n_strata,
        "stratify_by": "treated: arm (cell_id, compound_idx, dose_level); vehicle: det_plate",
        "n_train": int(final_train.size), "n_holdout": int(holdout_idx.size), "n_reserve": int(reserve_idx.size),
        "split_fingerprint": split_fingerprint(final_train, holdout_idx, reserve_idx),
        "report": report,
        "tier": tier,
        "train_idx": final_train.tolist(), "holdout_idx": holdout_idx.tolist(),
        "reserve_idx": reserve_idx.tolist(),
    }
    if os.path.normpath(out_dir) != os.path.normpath(base_nz):
        for fn in COPIED_FROM_BASE:
            shutil.copy2(os.path.join(base_nz, fn), os.path.join(out_dir, fn))
    if scored:
        _atomic_write(os.path.join(out_dir, "dr_weights_design.npz"),
                      lambda f: np.savez(f, row_id=final_train.astype(np.int64), w=w_train,
                                         split_fingerprint=np.array(payload["split_fingerprint"])), mode="wb")
        _write_json(os.path.join(out_dir, "tier_meta.json"), {
            "gamma": args.gamma, "seed": seed, "k_reserve": args.k_reserve, "pmin": args.pmin,
            "keep_frac": keep_frac, "confounder": confounder,
            "realised_kept_frac": float(sum(a["n_kept"] for a in tier_arms) / max(sum(a["n_train"] for a in tier_arms), 1)),
            "reserve": reserve_arms,
            "design_weight": "P_c / pi_i on kept scored wells (inverse realised inclusion probability under the positivity redraw)",
            "bias": "expected naive vs IPW bias: Phase 5 (--plan, needs the Phase 4 oracle)",
            "compounds": tier_arms})
    if not scored:   # a v1 rebuild must not leave a thinning instance's design files behind
        for fn in ("dr_weights_design.npz", "tier_meta.json"):
            if os.path.isfile(os.path.join(out_dir, fn)):
                os.remove(os.path.join(out_dir, fn))
                print(f"[tier] removed stale {fn}")
    _atomic_write(os.path.join(out_dir, SPLITS_FILENAME), lambda f: json.dump(payload, f))
    cfg.paths.nuisance_dir = out_dir
    build_nu_rows(cfg)
    print(f"[tier] wrote {out_dir}")


if __name__ == "__main__":
    main()
