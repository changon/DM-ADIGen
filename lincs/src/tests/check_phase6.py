"""Read-back checks of the step-A data layer (`STEP_A.md` §5, A4-A6). No PyTorch.

Re-derives with pandas, independently of the builders:

  oracle    the line groups recorded = spec.LINE_GROUPS; each arm's G2 - G1
            contrast recomputed from expr.npy (group means minus that group's
            vehicle mean), over the pool's wells AND over its holdout wells only
            (`contrast_dir`, the readout's direction); the per-group well counts
  tiers     both instances (gamma = 0 and gamma > 0): line groups, every (arm,
            line) cell of a scored compound keeps >= 1 train well, the holdout is
            the base split's, the kept G2 share moves in OPPOSITE directions in
            the two dose halves at gamma > 0 and not at gamma = 0
  weights   counts weights restore each scored arm's unthinned line mix EXACTLY
            (the property that makes a weighted risk possible at arm level, which
            step C2 lacked); P2's group normalisation leaves that balance intact
            and every unthinned row at exactly 1
  power     step_a_power.json: this population and line groups, and S0's
            ratio is |memoriser DiD| / SE

    python -m src.tests.check_phase6 --data_dir data/core5_24h
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_tiered_split import tier_dir  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.nuisances.weight_norm import P2_CLIP, group_normalize, train_groups  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args, line_groups)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--oracle", default=None)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n_arm_sample", type=int, default=60)
    p.add_argument("--power", default="step_a_power.json", help="Relative to <runs>/eval_artifacts/.")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    lg = line_groups(cfg.population)
    if lg is None:
        raise SystemExit(f"population {cfg.population.name!r} declares no line groups")
    lgj = {k: list(v) for k, v in lg.items()}
    R = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    print(f"[check6] {cfg.paths.data_dir}  population {cfg.population.name}  groups {lgj}")

    df = (load_from_disk(cfg.paths.tabular_dataset_dir)
          .select_columns(["compound_idx", "dose_level", "is_control", "det_plate", "cell_id"])
          .to_pandas())
    N = len(df)
    ctl = df["is_control"].values.astype(bool)
    df["arm"] = np.where(ctl, "0|ctl", df["compound_idx"].astype(str) + "|"
                         + df["dose_level"].map(lambda v: f"{float(v):.6g}"))
    df["g2"] = df["cell_id"].isin(lg["G2"]).values
    check(bool(df["cell_id"].isin(lg["G1"] + lg["G2"]).all()), "every row's line is in G1 or G2")
    # dose half, re-derived with pandas (dense rank of the compound's distinct levels)
    sub = df.loc[~ctl, ["compound_idx", "dose_level"]]
    lvl = sub.groupby("compound_idx")["dose_level"].rank(method="dense") - 1
    nlev = sub.groupby("compound_idx")["dose_level"].transform("nunique")
    half = np.full(N, -1)
    half[~ctl] = (lvl >= nlev // 2).astype(int).values
    df["half"] = half

    # ---- oracle ------------------------------------------------------------
    op = args.oracle or os.path.join(R, f"oracle_{cfg.population.name}_poolall.json")
    doc = json.load(open(op))
    z = np.load(op[:-5] + "_tau.npz", allow_pickle=False)
    check(doc.get("line_groups") == lgj, f"oracle line groups = spec.LINE_GROUPS")
    splits = load_splits(cfg)
    m = load_expr_meta(cfg, doc["plate_center"], splits=splits)
    y = normalize_expr(np.asarray(np.load(cfg.paths.expr_npy, mmap_mode="r")),
                       plate_codes(df["det_plate"].values, m["plates"]), m)
    ak = z["all/arm_key"].astype(str)
    con = z["all/contrast"].astype(np.float64)
    check(con.shape == (ak.size, cfg.outcome.n_genes), f"contrast table {con.shape} for {ak.size:,} arms")
    idx_of = df.groupby(["arm", "g2"]).indices
    v1 = y[idx_of[("0|ctl", False)]].astype(np.float64).mean(0)
    v2 = y[idx_of[("0|ctl", True)]].astype(np.float64).mean(0)
    rng = np.random.default_rng(0)
    pick = rng.choice(ak.size, min(args.n_arm_sample, ak.size), replace=False)
    worst, n_bad = 0.0, 0
    for i in pick:
        a = ak[i]
        r1, r2 = idx_of.get((a, False)), idx_of.get((a, True))
        if r1 is None or r2 is None:
            n_bad += int(np.isfinite(con[i]).any())
            continue
        want = (y[r2].astype(np.float64).mean(0) - v2) - (y[r1].astype(np.float64).mean(0) - v1)
        worst = max(worst, float(np.abs(want - con[i]).max()))
        n_bad += int(z["all/contrast_n1"][i] != len(r1) or z["all/contrast_n2"][i] != len(r2))
    check(worst < 2e-4 and n_bad == 0,
          f"oracle contrast = (G2 arm mean - G2 vehicle mean) - (G1 ...) recomputed with pandas "
          f"for {len(pick)} arms (max diff {worst:.1e}); group well counts match")
    # the readout's direction: the same contrast over the HOLDOUT wells only
    cdir = z["all/contrast_dir"].astype(np.float64)
    hold = np.zeros(N, bool); hold[splits["holdout_idx"]] = True
    dh = df.loc[hold]
    idx_h = dh.groupby(["arm", "g2"]).indices
    rows_h = dh.index.values
    h1 = y[rows_h[idx_h[("0|ctl", False)]]].astype(np.float64).mean(0)
    h2 = y[rows_h[idx_h[("0|ctl", True)]]].astype(np.float64).mean(0)
    worst_d, n_bad_d = 0.0, 0
    for i in pick:
        r1, r2 = idx_h.get((ak[i], False)), idx_h.get((ak[i], True))
        if r1 is None or r2 is None:
            n_bad_d += int(np.isfinite(cdir[i]).any())
            continue
        want = (y[rows_h[r2]].astype(np.float64).mean(0) - h2) - (y[rows_h[r1]].astype(np.float64).mean(0) - h1)
        worst_d = max(worst_d, float(np.abs(want - cdir[i]).max()))
        n_bad_d += int(z["all/contrast_dir_n1"][i] != len(r1) or z["all/contrast_dir_n2"][i] != len(r2))
    check(worst_d < 2e-4 and n_bad_d == 0,
          f"oracle contrast_dir = the same contrast over HOLDOUT wells only, recomputed with "
          f"pandas for {len(pick)} arms (max diff {worst_d:.1e})")
    hd = np.isfinite(cdir).all(1)
    nh = pd.Series(1, index=df.index)[hold & ~ctl].groupby(df.loc[hold & ~ctl, "arm"]).sum()
    tot = z["all/contrast_dir_n1"] + z["all/contrast_dir_n2"]
    check(bool((tot[hd] == nh.reindex(ak[hd]).values).all()),
          f"direction wells = the arm's holdout wells ({int(hd.sum()):,} arms have one; "
          f"median {int(np.median(tot[hd]))} wells, none of them a training well)")
    has = np.isfinite(con).all(1)
    n12 = z["all/contrast_n1"] + z["all/contrast_n2"]
    check(bool((n12[has] == z["all/n_wells"][has]).all()), "G1 + G2 wells = the arm's wells")
    cn = np.linalg.norm(con[has], axis=1)
    print(f"      {int(has.sum()):,}/{ak.size:,} arms have a contrast; ||c|| median {np.median(cn):.2f} "
          f"(responder arms {np.median(cn[z['all/responder'][has]]):.2f})"
          if z["all/responder"][has].any() else f"      {int(has.sum()):,} arms have a contrast")

    # ---- tiers -------------------------------------------------------------
    seed = int(splits["seed"])
    dp_rec, kept_frac = {}, {}
    for g in (0.0, float(args.gamma)):
        T = tier_dir(cfg, "cell_id", g, seed)
        nm = os.path.basename(T)
        if not os.path.isdir(T):
            check(False, f"{nm}: the tier dir exists"); continue
        ts = json.load(open(os.path.join(T, "splits.json")))
        tier = ts["tier"]
        check(tier.get("confounder") == "cell_id" and tier.get("line_groups") == lgj
              and abs(float(tier["gamma"]) - g) < 1e-12 and ts["params"].get("line_groups") == lgj,
              f"{nm}: confounder cell_id, gamma {g:g}, line groups recorded in tier and params")
        check(ts["holdout_idx"] == splits["holdout_idx"].tolist(), f"{nm}: holdout = the base split's")
        tr = np.asarray(ts["train_idx"])
        nu = np.load(os.path.join(T, "nu_rows.npy"))
        scored = np.array(sorted(int(v) for v in tier["scored_compounds"].values()))
        in_tr = np.zeros(N, bool); in_tr[tr] = True
        in_nu = np.zeros(N, bool); in_nu[nu] = True
        sc_row = df["compound_idx"].isin(scored).values & ~ctl
        cell_nu = df.loc[in_nu & sc_row].groupby(["arm", "cell_id"]).size()
        cell_tr = df.loc[in_tr & sc_row].groupby(["arm", "cell_id"]).size().reindex(cell_nu.index, fill_value=0)
        check(bool((cell_tr >= 1).all()),
              f"{nm}: all {len(cell_nu):,} scored (arm, line) cells keep >= 1 train well "
              f"(min {int(cell_tr.min())}; unthinned {int(cell_nu.sum()):,} -> kept {int(cell_tr.sum()):,})")
        n_lines = df.loc[in_nu & sc_row].groupby("arm")["cell_id"].nunique()
        check(bool((n_lines == len(cfg.population.cell_ids)).all()),
              f"{nm}: every scored arm has train wells in all {len(cfg.population.cell_ids)} lines")
        # the lever: kept G2 share minus the unthinned train pool's, per scored arm
        a_nu = df.loc[in_nu & sc_row].groupby("arm")["g2"].mean()
        a_tr = df.loc[in_tr & sc_row].groupby("arm")["g2"].mean().reindex(a_nu.index)
        a_half = df.loc[in_nu & sc_row].groupby("arm")["half"].first()
        dp = (a_tr - a_nu)
        lo, hi = float(dp[a_half == 0].mean()), float(dp[a_half == 1].mean())
        if g == 0:
            check(abs(hi - lo) < 0.02, f"{nm}: gamma = 0 does not move the line mix (dp low {lo:+.4f}, high {hi:+.4f})")
        else:
            check(lo * hi < 0 and abs(hi - lo) > 0.05,
                  f"{nm}: gamma = {g:g} moves the G2 share in OPPOSITE directions by dose half "
                  f"(low {lo:+.4f}, high {hi:+.4f}) -- the dose_half x line-group lever")
        # counts weights: the weighted G2 share of each scored arm = the unthinned pool's, exactly
        wz = np.load(os.path.join(T, "dr_weights_counts.npz"))
        w = pd.Series(wz["w"].astype(np.float64), index=wz["row_id"])
        d_tr = df.loc[in_tr & sc_row].copy()
        d_tr["w"] = w.reindex(d_tr.index).values
        wmix = (d_tr["w"] * d_tr["g2"]).groupby(d_tr["arm"]).sum() / d_tr["w"].groupby(d_tr["arm"]).sum()
        err = float((wmix.reindex(a_nu.index) - a_nu).abs().max())
        check(err < 1e-6, f"{nm}: counts weights restore every scored arm's unthinned line mix "
                          f"(max |weighted G2 share - unthinned| {err:.1e})")
        # P2: the group here is the arm; balance kept, unthinned rows exactly 1
        grp = train_groups(cfg, "cell_id", tr)
        wn, st = group_normalize(wz["w"], grp, cap=P2_CLIP)
        d_tr["wn"] = pd.Series(wn, index=tr).reindex(d_tr.index).values
        wmix2 = (d_tr["wn"] * d_tr["g2"]).groupby(d_tr["arm"]).sum() / d_tr["wn"].groupby(d_tr["arm"]).sum()
        err2 = float((wmix2.reindex(a_nu.index) - a_nu).abs().max())
        un = ~pd.Series(sc_row, index=df.index).reindex(tr).values
        check(st["max_group_sum_rel_err"] < 1e-9 and bool((wn[un] == 1.0).all())
              and (err2 < 1e-6 or st["n_capped"] > 0),
              f"{nm}: P2 (group = arm, cap {P2_CLIP:g}) keeps every arm's row count, leaves "
              f"{int(un.sum()):,} unthinned rows at exactly 1, and the line balance "
              f"(max err {err2:.1e}; {st['n_capped']} capped rows, max weight {st['max']:.2f})")
        dp_rec[f"{g:g}"] = (lo, hi)
        kept_frac[g] = float(cell_tr.sum() / cell_nu.sum())

    if len(kept_frac) == 2:
        a_, b_ = kept_frac[0.0], kept_frac[float(args.gamma)]
        check(abs(a_ - b_) < 0.02,
              f"the gamma = 0 control is size-matched: it keeps {a_:.3f} of the scored train "
              f"wells, the gamma = {args.gamma:g} instance {b_:.3f}")

    # ---- power -------------------------------------------------------------
    pp = args.power if os.path.sep in args.power else os.path.join(R, args.power)
    if os.path.isfile(pp):
        pw = json.load(open(pp))
        check(pw["line_groups"] == lgj and pw["population"] == cfg.population.name,
              "step_a_power.json: this population and these line groups")
        mem = pw["memoriser"]
        check(mem["se"] is None or abs(abs(mem["did"]) / mem["se"] - pw["S0"]["ratio"]) < 1e-9,
              f"S0 ratio = |memoriser DiD| / SE = {pw['S0']['ratio']} (DiD {mem['did']:+.3f}; "
              f"pass={pw['S0']['pass']} at >= {pw['S0']['min_ratio']})")
        print(f"      power: memoriser {mem['did']:+.3f} +/- {mem['se']}, planned "
              f"{pw['planned']['did']:+.3f}, counts-weighted {pw['weighted_memorisers']['counts']['did']:+.3f}, "
              f"design-weighted {pw['weighted_memorisers']['design']['did']:+.3f}")
    else:
        print(f"skip  no {pp}; run src.eval.step_a_power first")

    if FAILS:
        print(f"\n[check6] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS))
        sys.exit(1)
    print("\n[check6] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
