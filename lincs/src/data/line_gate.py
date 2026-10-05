"""Phase 6 gate (IMPLEMENT.md §3.8.3): does cell line MODIFY the compound response? No PyTorch.

§3.8.3 requires this measurement before `core5_24h` is built:

    read the responders' wells in the 5 lines (a small h5py read); compare
    per-line centred tau_hat across lines against the 3-well noise floor. If
    lines respond alike, cell_id has no effect on centred Y, and step A
    collapses into another null check.

It runs on raw Level 3 and does NOT need the `core5_24h` build (which needs the
`--population` switch, not written yet): it selects the population from
`inst_info`, reads the wells it needs straight from the GCTX, and applies the
same outcome rules the build would (plate QC with the 3x rule on each line's
median spread, DMSO-median plate centring, then one pooled z-scale).

**The statistic.** For an arm a = (compound, dose_level) and a line L,
tau_hat_L(a) = mean Y over L's wells of a, minus the mean over L's DMSO wells,
in that z-space. Two independent estimates of the SAME tau correlate at the
full-sample reliability r_full = 2 r_half / (1 + r_half) (Spearman-Brown on the
within-line split-half r_half; E11, `evaluate.corrected_ceiling` is its square
root). So, for arms measured in two lines:

    r_cross ~ r_full   ->  the lines share one tau: NO effect modification
    r_cross <  r_full  ->  cell line modifies the response

and the §3.8.3 wording, ||tau_hat_L1 - tau_hat_L2|| against the floor: under one
shared tau the difference of two independent n-well estimates has norm
~sqrt(2) x floor(n), so the ratio to that is ~1 with no modification.

**MCF7 pairs are reported apart from the rest.** The responder list is MCF7's
(`responders.json`), so MCF7's tau_hat is selected on being large and its pairs
carry that selection. The four other lines were never selected on, so the
headline is the median over their 6 pairs.

    python -m src.data.line_gate                      # all 641 responders, 5 lines
    python -m src.data.line_gate --max_compounds 40   # a quick subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import (  # noqa: E402
    MIN_DMSO_PER_PLATE, _write_json, encode_action, gctx_well_positions, landmark_genes,
    parse_plate, plate_qc, read_gctx_rows, read_tsv)
from src.eval.evaluate import (  # noqa: E402
    _cos, _q, floor_for_sizes, noise_floor, split_half_reliability)
from src.nuisances.precompute_cmean import group_means  # noqa: E402
from src.spec import (  # noqa: E402
    add_paths_cli, apply_paths_args, config_from_args, default_config)

# §3.8.3: compounds present in all five; HELA and YAPC miss ~10% of arms.
CORE5 = ("MCF7", "HT29", "HA1E", "A375", "PC3")
SELECTED_LINE = "MCF7"      # the line responders.json was derived from


def _pairs(lines):
    return [(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lines", default=",".join(CORE5))
    p.add_argument("--responders", default="data/mcf7_24h/nuisances/responders.json",
                   help="The MCF7 responder set (its `compounds`); the entry criterion.")
    p.add_argument("--max_compounds", type=int, default=0,
                   help="Use only the first N responders (by max ||tau_hat||); 0 = all.")
    p.add_argument("--min_wells", type=int, default=2,
                   help="Minimum wells an (arm, line) cell needs to enter the comparison.")
    p.add_argument("--n_floor_draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_ratio_alike", type=float, default=0.8,
                   help="Verdict threshold: r_cross / r_full below this on the unselected "
                        "pairs means cell line modifies the response.")
    p.add_argument("--out", default="runs/core5_24h_line_gate.json")
    p.add_argument("--population_compounds", default=None)   # unused; add_paths_cli's sibling
    p.add_argument("--adjustment_set", default=None)
    p.add_argument("--environment_set", default=None)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    t0 = time.time()
    lines = [s.strip() for s in args.lines.split(",") if s.strip()]
    cfg = apply_paths_args(config_from_args(args), args, require_splits=False)
    rng_seed = int(args.seed)

    # ---- 1. the population, from inst_info (no build needed) -------------
    inst = read_tsv(cfg.paths.raw("inst_info"))
    pop = default_config().population
    t = pd.to_numeric(inst["pert_time"], errors="coerce").values
    m = (inst["pert_type"].isin(pop.pert_types).values
         & np.isclose(t, pop.pert_time_h) & (inst["pert_time_unit"] == "h").values
         & inst["cell_id"].isin(lines).values)
    df = inst.loc[m].copy()
    with open(args.responders) as fh:
        rj = json.load(fh)
    detail = sorted(rj["detail"], key=lambda d: -float(d["max_tau_norm_real"]))
    if args.max_compounds:
        detail = detail[:args.max_compounds]
    resp = [d["pert_id"] for d in detail]
    print(f"[gate] lines {lines}; {len(df):,} wells at {pop.pert_time_h:g} h; "
          f"{len(resp):,} responder compounds from {os.path.basename(args.responders)}",
          flush=True)

    is_ctl_all = df[cfg.action.control_col].isin(cfg.action.control_values).values
    keep_trt = df["pert_id"].isin(resp).values & ~is_ctl_all
    # Every DMSO well of a plate that carries a responder well: centring and the
    # noise floor are both per plate / per line, and need the vehicles.
    plates = set(df.loc[keep_trt, "det_plate"].unique())
    on_plate = df["det_plate"].isin(plates).values
    df = df.loc[keep_trt | (on_plate & is_ctl_all)].copy()
    df = df.sort_values("inst_id", kind="stable").reset_index(drop=True)
    df = pd.concat([df, parse_plate(df["det_plate"])], axis=1)
    df = encode_action(df, cfg)
    is_ctl = df["is_control"].values.astype(bool)
    print(f"[gate] {len(df):,} wells on {len(plates):,} plates "
          f"({int((~is_ctl).sum()):,} treated, {int(is_ctl.sum()):,} DMSO)", flush=True)

    # Plates that cannot be centred at all (they would raise in plate_dmso_spread).
    n_dmso_p = df.loc[is_ctl].groupby("det_plate").size().reindex(
        sorted(df["det_plate"].unique()), fill_value=0)
    short = n_dmso_p.index[n_dmso_p < MIN_DMSO_PER_PLATE].tolist()
    if short:
        print(f"[gate] dropping {len(short)} plates with < {MIN_DMSO_PER_PLATE} DMSO wells",
              flush=True)
        df = df.loc[~df["det_plate"].isin(short)].reset_index(drop=True)
        is_ctl = df["is_control"].values.astype(bool)

    # ---- 2. the outcome ---------------------------------------------------
    gctx = cfg.paths.raw("gctx")
    gene_info = read_tsv(cfg.paths.raw("gene_info"))
    genes = landmark_genes(gctx, gene_info, cfg.outcome.n_genes)
    cols = genes["gctx_gene_pos"].values
    rows = gctx_well_positions(gctx, df["inst_id"].values)
    expr = read_gctx_rows(gctx, rows, cols)          # (n, 978) float32
    del rows

    keep, qc = plate_qc(df, expr, cfg.outcome.plate_qc_max_spread_ratio)
    print(f"[gate] plate QC: dropped {len(qc['dropped_plates'])} plates, "
          f"{qc['n_wells_dropped']:,} wells", flush=True)
    df, expr = df.loc[keep].reset_index(drop=True), expr[keep]
    is_ctl = df["is_control"].values.astype(bool)

    # Centre each plate on its own DMSO median, then one pooled z-scale
    # (decision 5's analogue; no split exists yet, so all DMSO wells are used --
    # recorded, because the built pipeline uses TRAIN DMSO only).
    plate_lv = sorted(df["det_plate"].unique())
    pc = pd.Series(df["det_plate"].values).map({p: i for i, p in enumerate(plate_lv)}).values
    for i in range(len(plate_lv)):
        sel = (pc == i) & is_ctl
        expr[pc == i] -= np.median(expr[sel].astype(np.float64), axis=0).astype(np.float32)
    mean = expr[is_ctl].mean(axis=0, dtype=np.float64)
    std = np.maximum(expr.std(axis=0, dtype=np.float64), 1e-6)
    expr -= mean.astype(np.float32)
    expr /= std.astype(np.float32)
    z = expr
    print(f"[gate] centred and z-scored: {z.shape[0]:,} x {z.shape[1]} "
          f"({z.nbytes / 2**20:.0f} MiB)", flush=True)

    # ---- 3. per-line tau_hat, floor and within-line reliability -----------
    cell = df["cell_id"].values
    arm = np.array([f"{p}|{d:.6g}" for p, d in
                    zip(df["pert_id"].values, df["dose_level"].values)], dtype=object)
    per_line, floors, rel = {}, {}, {}
    for L in lines:
        inL = cell == L
        d_rows = np.flatnonzero(inL & is_ctl)
        t_rows = np.flatnonzero(inL & ~is_ctl)
        mu0 = z[d_rows].mean(0)
        uniq, mu, cnt = group_means(z[t_rows], arm[t_rows].astype(str))
        per_line[L] = {"arm": uniq.astype(str), "tau": mu - mu0, "n": cnt,
                       "mu0": mu0, "n_dmso": int(d_rows.size)}
        floors[L] = noise_floor(z[d_rows], np.unique(cnt), args.n_floor_draws, rng_seed)
        rel[L] = split_half_reliability(z[t_rows], arm[t_rows].astype(str), mu0, rng_seed)
        fr = np.linalg.norm(per_line[L]["tau"], axis=1) / floor_for_sizes(floors[L], cnt)
        per_line[L]["floor_ratio"] = fr
        print(f"[gate] {L:5s} {uniq.size:,} arms, {d_rows.size:,} DMSO; median wells/arm "
              f"{int(np.median(cnt))}; median ||tau||/floor {np.median(fr):.2f}; "
              f"responder arms {float((fr > 1.5).mean()):.1%}", flush=True)

    # ---- 4. cross-line vs within-line -------------------------------------
    index = {L: {a: i for i, a in enumerate(per_line[L]["arm"])} for L in lines}
    recs = []
    for L1, L2 in _pairs(lines):
        shared = [a for a in per_line[L1]["arm"] if a in index[L2]]
        for a in shared:
            i1, i2 = index[L1][a], index[L2][a]
            n1, n2 = int(per_line[L1]["n"][i1]), int(per_line[L2]["n"][i2])
            if min(n1, n2) < args.min_wells:
                continue
            t1, t2 = per_line[L1]["tau"][i1], per_line[L2]["tau"][i2]
            f1 = float(floor_for_sizes(floors[L1], np.array([n1]))[0])
            f2 = float(floor_for_sizes(floors[L2], np.array([n2]))[0])
            r1 = rel[L1]["per_arm"].get(a, np.nan)
            r2 = rel[L2]["per_arm"].get(a, np.nan)
            recs.append({
                "arm": a, "L1": L1, "L2": L2, "n1": n1, "n2": n2,
                "cos": _cos(t1, t2),
                # under one shared tau: ||t1 - t2|| ~ sqrt(f1^2 + f2^2)
                "norm_ratio": float(np.linalg.norm(t1 - t2) / np.sqrt(f1 ** 2 + f2 ** 2)),
                "fr1": float(per_line[L1]["floor_ratio"][i1]),
                "fr2": float(per_line[L2]["floor_ratio"][i2]),
                "r_half": float(np.nanmean([r1, r2])),
            })
    R = pd.DataFrame(recs)
    if R.empty:
        raise SystemExit("[gate] no arm is measured in two lines with enough wells")
    R["r_full"] = 2 * R["r_half"] / (1 + R["r_half"])      # Spearman-Brown (E11)
    R["has_mcf7"] = (R["L1"] == SELECTED_LINE) | (R["L2"] == SELECTED_LINE)
    # Only arms that respond somewhere can inform the comparison: a non-responding
    # arm is noise in both lines, so r_cross and r_full are both ~0.
    R["responds"] = (R["fr1"] > 1.5) | (R["fr2"] > 1.5)

    def block(sub, label):
        s = sub[sub["responds"] & np.isfinite(sub["r_full"]) & (sub["r_full"] > 0)]
        if s.empty:
            return {"label": label, "n_pairs": 0}
        rc, rf = float(s["cos"].median()), float(s["r_full"].median())
        return {"label": label, "n_pairs": int(len(s)), "n_arms": int(s["arm"].nunique()),
                "r_cross_median": rc, "r_full_median": rf,
                "ratio_cross_over_full": float(rc / rf) if rf > 0 else None,
                "norm_ratio_median": float(s["norm_ratio"].median()),
                "cos": _q(s["cos"].values), "norm_ratio": _q(s["norm_ratio"].values)}

    head = block(R[~R["has_mcf7"]], "pairs among the 4 unselected lines")
    mcf7 = block(R[R["has_mcf7"]], f"pairs involving {SELECTED_LINE} (selected on)")
    allp = block(R, "all pairs")
    per_pair = {f"{a}|{b}": block(R[(R["L1"] == a) & (R["L2"] == b)], f"{a} vs {b}")
                for a, b in _pairs(lines)}

    # Per compound, over the unselected pairs: where is the modification?
    s = R[(~R["has_mcf7"]) & R["responds"]]
    by_c = (s.assign(c=s["arm"].str.split("|").str[0])
             .groupby("c").agg(n_pairs=("cos", "size"), cos=("cos", "median"),
                               norm_ratio=("norm_ratio", "median")).sort_values("cos"))
    iname = {d["pert_id"]: d["pert_iname"] for d in rj["detail"]}

    ratio = head.get("ratio_cross_over_full") if head.get("n_pairs") else None
    if ratio is None:
        verdict = "UNDECIDED: no arm pair among the unselected lines cleared the filters"
    elif ratio < args.max_ratio_alike:
        verdict = (f"effect modification: cross-line agreement is {ratio:.2f} of the "
                   f"within-line reliability (< {args.max_ratio_alike}), so cell line "
                   f"changes the response and step A has something to measure")
    else:
        verdict = (f"lines respond alike: cross-line agreement is {ratio:.2f} of the "
                   f"within-line reliability (>= {args.max_ratio_alike}), so step A "
                   f"would be another null check")
    doc = {"population": "core5_24h", "lines": lines, "selected_line": SELECTED_LINE,
           "responders": {"path": os.path.abspath(args.responders),
                          "n_compounds": len(resp), "max_compounds": args.max_compounds or None},
           "n_wells": int(len(df)), "n_plates": len(plate_lv),
           "min_wells": args.min_wells, "seed": rng_seed,
           "max_ratio_alike": args.max_ratio_alike,
           "outcome": {"plate_center": "dmso_median_all_wells (no split exists yet)",
                       "normalize_mean": "dmso_all", "normalize_std": "all_rows_pooled_over_lines",
                       "plate_qc_max_spread_ratio": cfg.outcome.plate_qc_max_spread_ratio},
           "plate_qc": qc,
           "per_line": {L: {"n_arms": int(per_line[L]["arm"].size),
                            "n_dmso": per_line[L]["n_dmso"],
                            "median_wells_per_arm": int(np.median(per_line[L]["n"])),
                            "floor_median_n3": (floors[L].get("3") or {}).get("median"),
                            "tau_norm": _q(np.linalg.norm(per_line[L]["tau"], axis=1)),
                            "floor_ratio": _q(per_line[L]["floor_ratio"]),
                            "responder_arm_frac": float((per_line[L]["floor_ratio"] > 1.5).mean()),
                            "split_half_overall": rel[L]["overall"]} for L in lines},
           "headline": head, "mcf7_pairs": mcf7, "all_pairs": allp, "per_pair": per_pair,
           "most_line_specific_compounds": [
               {"pert_id": c, "pert_iname": iname.get(c), "n_pairs": int(r.n_pairs),
                "cos_median": float(r.cos), "norm_ratio_median": float(r.norm_ratio)}
               for c, r in by_c.head(15).iterrows()],
           "least_line_specific_compounds": [
               {"pert_id": c, "pert_iname": iname.get(c), "n_pairs": int(r.n_pairs),
                "cos_median": float(r.cos), "norm_ratio_median": float(r.norm_ratio)}
               for c, r in by_c.tail(10).iterrows()],
           "verdict": verdict,
           "statistic": ("per arm and line pair: cos(tau_L1, tau_L2) against the full-sample "
                         "reliability r_full = 2 r_half / (1 + r_half) from the within-line "
                         "split-half; and ||tau_L1 - tau_L2|| / sqrt(floor1^2 + floor2^2), "
                         "which is ~1 under one shared tau. Restricted to arms responding in "
                         "at least one of the two lines."),
           "elapsed_sec": round(time.time() - t0, 1)}
    out = args.out if os.path.isabs(args.out) else str(PROJECT_ROOT / args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_json(out, doc)

    print(f"\n=== cross-line agreement (arms responding in >= 1 line) ===")
    for b in (head, mcf7, allp):
        if b.get("n_pairs"):
            print(f"  {b['label']:42s} pairs {b['n_pairs']:6,}  r_cross {b['r_cross_median']:.3f}  "
                  f"r_full {b['r_full_median']:.3f}  ratio {b['ratio_cross_over_full']:.3f}  "
                  f"||dtau||/floor {b['norm_ratio_median']:.2f}")
    print("  per pair:")
    for k, b in per_pair.items():
        if b.get("n_pairs"):
            print(f"    {k:14s} r_cross {b['r_cross_median']:.3f}  r_full {b['r_full_median']:.3f}"
                  f"  ratio {b['ratio_cross_over_full']:.3f}  ||dtau||/floor {b['norm_ratio_median']:.2f}")
    print(f"\n[gate] VERDICT: {verdict}")
    print(f"[gate] -> {out}  ({doc['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
