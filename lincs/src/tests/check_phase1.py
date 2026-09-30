"""Read-back checks of the Phase 1 artifacts (IMPLEMENT.md §3.10). No PyTorch.

Re-derives what it can independently of the builders (pandas groupby rather
than the split / stats code paths) and checks the v1 known answers:

  split      strata counts (arm; DMSO by plate), disjointness, coverage,
             population + table key, determinism, realised holdout (P3)
  nu         nu_rows.npy = population - holdout - reserve (v1: = train)
  expr_meta  centres / mean / std / cap fraction recomputed on the unthinned
             train pool (nu); recorded stats give DMSO z-mean 0, z-std 1
  gate       centring gate on held-out DMSO recomputed and passing (P11)
  folds      stratified cross-fit folds of fit_urr's run (P5)   [after fit_urr]
  urr        gate usable; v1: alpha ~ 1/(1-pi0) treated, ~0 DMSO
  weights    net weights scored out-of-fold (per-fold means = fit_urr's);
             v1: counts w = 1 exactly; net w ~ 1/(1-pi0) treated, 1 vehicle,
             ~1.004 / ~0.942 after mean-1 normalisation   [after export]
  tier       thinning instance, recomputed from the table: rows, z, pi,
             positivity, design weights P_c / pi; unscored holdout = v1's
Sections run when their artifacts exist. Exits nonzero on any failure.

Run from lincs/ (a CPU job for the full population):
    python -m src.tests.check_phase1                                    # data/mcf7_24h, base split
    python -m src.tests.check_phase1 --data_dir data/mcf7_24h_limit1500
    python -m src.tests.check_phase1 --nuisance_dir data/mcf7_24h/nuisances_tier_...
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.expr_stats import GATE_MAX_OMEGA2, GATE_MAX_OMEGA2_RATIO, GATE_MAX_PLATE_MEAN_Z  # noqa: E402
from src.data.splits import load_splits, split_layer, split_strata  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout_frac_tol", type=float, default=1e-9)
    p.add_argument("--known_answer_rtol", type=float, default=0.02, help="v1: |alpha_treated / (1/(1-pi0)) - 1|")
    p.add_argument("--no_gate", action="store_true", help="skip the centring-gate thresholds (e.g. a tiny --limit build)")
    p.add_argument("--prefix", default="alpha_urr", help="fit_urr --out_prefix to check")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    P, nz = cfg.paths, cfg.paths.nuisance_dir
    base_nz = os.path.join(P.data_dir, "nuisances")
    print(f"[check1] data {P.data_dir}\n[check1] split dir {nz}")

    from datasets import load_from_disk
    df = load_from_disk(P.tabular_dataset_dir).select_columns(
        ["pert_id", "cell_id", "compound_idx", "is_control", "dose_level", "det_plate", "syn_c",
         "cov_vec"]).to_pandas()
    N = len(df)
    ctl = df["is_control"].values.astype(bool)
    qc = json.load(open(P.population_qc_json))

    # === split ================================================================
    s = load_splits(cfg)                              # validates key + fingerprint
    tr, ho, rs = s["train_idx"], s["holdout_idx"], s["reserve_idx"]
    tier = s["tier"]
    check(s["population"]["table_fingerprint"] == qc["table_fingerprint"] and s["n_total"] == N,
          f"split keyed on this table ({qc['table_fingerprint']}, {N:,} rows) and population {cfg.population.key()}")
    check(np.all(np.diff(tr) > 0) and np.all(np.diff(ho) > 0), "train / holdout sorted, unique")
    check(not (set(tr) & set(ho)) and not (set(rs) & (set(tr) | set(ho))), "train, holdout, reserve disjoint")
    pop = np.ones(N, dtype=bool)
    if cfg.population.compounds:
        pop = df["pert_id"].isin(cfg.population.compounds).values | ctl
    nu = np.load(os.path.join(nz, "nu_rows.npy"))
    want_nu = np.flatnonzero(pop & ~np.isin(np.arange(N), ho) & ~np.isin(np.arange(N), rs))
    check(np.array_equal(nu, want_nu), f"nu_rows.npy = population - holdout - reserve ({nu.size:,} rows)")
    check(np.isin(tr, nu).all(), "train is a subset of nu")
    unthinned = nu                                         # the split layer's train side
    if not tier.get("active"):
        check(np.array_equal(tr, nu) and rs.size == 0, "v1: train = nu, no reserve")
    check(np.array_equal(np.sort(np.concatenate([unthinned, ho, rs])), np.flatnonzero(pop)),
          "unthinned train + holdout + reserve = population")

    # strata recomputed with pandas: treated arm (cell, compound, dose_level), DMSO by plate
    key = np.where(ctl, "dmso|" + df["det_plate"],
                   "arm|" + df["cell_id"] + "|" + df["compound_idx"].astype(str) + "|"
                   + df["dose_level"].map(lambda v: f"{v:.6g}"))
    role = np.full(N, "", dtype=object)
    role[unthinned], role[ho] = "train", "holdout"
    g = pd.DataFrame({"key": key, "role": role})[role != ""].groupby("key")["role"]
    n_pool = g.size()
    n_ho = g.apply(lambda r: int((r == "holdout").sum()))
    hf = s["holdout_frac"]
    expect = n_pool.map(lambda n: int(round(hf * n)) if n >= 2 else 0)
    bad = (n_ho != expect)
    check(not bad.any(), f"every stratum holds out round({hf} n) (n >= 2) of {len(n_pool):,} strata "
                         f"({int(bad.sum())} wrong)")
    is_dmso = n_pool.index.str.startswith("dmso|")
    dmso_train = (n_pool - n_ho)[is_dmso]
    n_arm_train = (n_pool - n_ho)[~is_dmso]
    realised = ho.size / (unthinned.size + ho.size)
    check(abs(realised - s["report"]["holdout_frac_realised"]) < args.holdout_frac_tol,
          f"realised holdout {realised:.4f} (target {hf}; treated {s['report']['holdout_frac_treated']:.4f}, "
          f"DMSO {s['report']['holdout_frac_dmso']:.4f}); {int((n_ho[~is_dmso] == 0).sum()):,}/"
          f"{int((~is_dmso).sum()):,} arms have no holdout well")
    check((n_arm_train >= 1).all(), f"every arm keeps a train well (min {int(n_arm_train.min())})")
    check(dmso_train.min() >= 3 and set(dmso_train.index.str[5:]) == set(df["det_plate"]),
          f"every plate keeps train DMSO wells (min {int(dmso_train.min())}, median {dmso_train.median():g})")
    strata = split_strata(df["cell_id"].values, df["compound_idx"].values, df["dose_level"].values, ctl,
                          df["det_plate"].values)
    pool = pop.copy()
    pool[rs] = False
    tr2, ho2, _ = split_layer(strata, np.flatnonzero(pool), hf, s["seed"])
    check(np.array_equal(ho2, ho) and np.array_equal(tr2, unthinned), f"split reproducible from seed {s['seed']}")

    # === split dir files ======================================================
    vocab = json.load(open(os.path.join(nz, "compound_vocab.json")))
    nm = json.load(open(os.path.join(nz, "nuisance_meta.json")))
    cv = np.stack(df["cov_vec"].values)
    check(nm["n_compounds"] == len(vocab) == int(df["compound_idx"].max()) + 1 and nm["cov_dim"] == cv.shape[1],
          f"nuisance_meta.json in the split dir: n_compounds {nm['n_compounds']}, cov_dim {nm['cov_dim']} "
          f"(trainer needs no fit_urr)")
    if os.path.normpath(nz) != os.path.normpath(base_nz):
        same = all(filecmp.cmp(os.path.join(nz, f), os.path.join(base_nz, f), shallow=False)
                   for f in ("nuisance_meta.json", "compound_vocab.json", "covariate_encoder.json"))
        check(same, "tiered dir copies of nuisance_meta / vocab / covariate encoder = base")

    # === expr_meta + centring gate (stats fit on the unthinned train pool, nu) ===
    em_path = os.path.join(nz, "expr_meta.json")
    if os.path.isfile(em_path):
        em = json.load(open(em_path))
        go = json.load(open(P.gene_order_json))
        check(em["split_fingerprint"] == s["split_fingerprint"] and em["table_fingerprint"] == qc["table_fingerprint"]
              and em["pr_gene_id"] == go["pr_gene_id"], "expr_meta keyed on this split, table and gene order")
        expr = np.load(P.expr_npy)
        x_tr = pd.DataFrame(expr[nu].astype(np.float64))
        plates_tr = df["det_plate"].values[nu]
        dm = ctl[nu]
        centre = x_tr[dm].groupby(plates_tr[dm]).median()
        centre = centre.reindex(em["plates"])
        c_em = np.asarray(em["centre"])
        check(em["plate_center"] == "dmso_median_train" and np.abs(centre.values - c_em).max() < 1e-9,
              f"plate centres = per-plate median of train DMSO ({len(em['plates'])} plates)")
        xc = x_tr.values - centre.loc[plates_tr].values
        mean, std = xc[dm].mean(0), xc.std(0)
        em_mean, em_std = np.asarray(em["mean"]), np.asarray(em["std"])
        check(np.abs(mean - em_mean).max() < 1e-9 and np.abs(std - em_std).max() < 1e-9,
              f"mean = centred train DMSO mean; std = std of all {nu.size:,} centred unthinned-train rows")
        z = (xc - em_mean) / em_std
        check(np.abs(z[dm].mean(0)).max() < 1e-9 and np.abs(z.std(0) - 1).max() < 1e-9,
              f"recorded stats give z with DMSO mean 0, all-row std 1 (range [{z.min():.1f}, {z.max():.1f}], unclamped)")
        cap = (expr[nu] >= 15.0).mean(0)
        sym = np.asarray(go["pr_gene_symbol"])
        gi = int(np.flatnonzero(sym == "GAPDH")[0])
        check(np.abs(cap - np.asarray(em["cap_frac"])).max() < 1e-12,
              f"cap fraction at 15.0: GAPDH {cap[gi]:.3f}; {int((cap > 0.01).sum())} genes > 1%")

        # gate, recomputed on held-out DMSO
        H = ho[ctl[ho]]
        ph = df["det_plate"].values[H]
        xh = pd.DataFrame(expr[H].astype(np.float64))
        ch = xh - centre.loc[ph].values
        def share(y):                                             # one-way ANOVA per gene
            gm = y.groupby(ph).transform("mean")
            sst = ((y - y.mean()) ** 2).sum()
            ssb = ((gm - y.mean()) ** 2).sum()
            k, n = len(set(ph)), len(y)
            msw = (sst - ssb) / (n - k)
            return float(np.median(ssb / sst)), float(np.median((ssb - (k - 1) * msw) / (sst + msw)))
        (r2b, omb), (r2a, oma) = share(xh), share(ch)
        gate = em["gate"]
        check(abs(r2b - gate["plate_share_r2"]["before"]) < 1e-9 and abs(r2a - gate["plate_share_r2"]["after"]) < 1e-9
              and abs(omb - gate["plate_share_omega2"]["before"]) < 1e-9
              and abs(oma - gate["plate_share_omega2"]["after"]) < 1e-9,
              f"gate plate shares recomputed: R2 {r2b:.3f} -> {r2a:.3f} (chance "
              f"{gate['plate_share_r2']['chance']:.3f}), omega2 {omb:.3f} -> {oma:.3f}")
        cnt = pd.Series(plates_tr[dm]).value_counts()
        sd = np.sqrt(((x_tr[dm] - x_tr[dm].groupby(plates_tr[dm]).transform("mean")) ** 2).sum().values
                     / (dm.sum() - cnt.size))
        mh = ch.groupby(ph).mean()
        nh = pd.Series(ph).value_counts().reindex(mh.index)
        se = np.sqrt(1.0 / nh.values[:, None] + (math.pi / 2) / cnt.reindex(mh.index).values[:, None]) * sd[None, :]
        zmed = float(np.median(np.abs(mh.values) / se))
        check(abs(zmed - gate["plate_mean_abs_z"]["median_centred"]) < 1e-9,
              f"gate plate-mean |z| recomputed: median {zmed:.3f} (expected ~0.674 when centred; "
              f"uncentred {gate['plate_mean_abs_z']['median_uncentred']:.1f})")
        if not args.no_gate:
            check(zmed <= GATE_MAX_PLATE_MEAN_Z, f"GATE per-plate held-out means ~ 0 (median |z| {zmed:.3f} <= {GATE_MAX_PLATE_MEAN_Z})")
            check(oma <= GATE_MAX_OMEGA2 and oma <= GATE_MAX_OMEGA2_RATIO * omb,
                  f"GATE chance-corrected plate share of held-out DMSO variance omega2 {omb:.3f} -> {oma:.3f} "
                  f"(<= {GATE_MAX_OMEGA2} and <= {GATE_MAX_OMEGA2_RATIO} x before; raw R2 {r2b:.3f} -> {r2a:.3f}, "
                  f"chance {gate['plate_share_r2']['chance']:.3f})")
    else:
        print("skip  expr_meta.json absent (run src.data.expr_stats)")

    # === folds + URR ============================================================
    arm = np.where(ctl, "0|ctl", df["compound_idx"].astype(str) + "|" + df["dose_level"].map(lambda v: f"{v:.6g}"))
    v1 = not tier.get("active") and not cfg.adjustment_set
    pi0 = float(ctl[tr].mean())
    r_v1 = 1.0 / (1.0 - pi0)
    umeta_path = os.path.join(nz, f"{args.prefix}_meta.json")
    um = json.load(open(umeta_path)) if os.path.isfile(umeta_path) else None
    fpath = os.path.join(nz, f"{args.prefix}_folds.npy")
    folds = None
    if um is not None and os.path.isfile(fpath):
        folds = np.load(fpath)
        pool = nu if um["fit"]["nu_rows"] else tr
        outside = np.ones(N, dtype=bool)
        outside[pool] = False
        check(hashlib.sha1(folds.tobytes()).hexdigest() == um["fit"]["folds_sha1"]
              and np.isin(folds[pool], (0, 1)).all() and (folds[outside] == -1).all(),
              f"folds = fit_urr run {um['fit']['run_id']}'s: 0/1 on the {pool.size:,}-row pool, -1 elsewhere")
        xs = np.array(["".join(map(str, r)) for r in (cv[:, um["fit"]["cov_idx"]] > 0.5).astype(np.int8)])
        in_tr = np.isin(np.arange(N), tr)
        key = pd.Series(arm[pool]) + "|X=" + xs[pool] + np.where(in_tr[pool], "|train", "|nu")
        c = pd.DataFrame({"k": key.values, "f": folds[pool]}).groupby("k")["f"].agg(["size", "sum"])
        multi = c[c["size"] >= 2]
        bal = ((2 * multi["sum"] - multi["size"]).abs() <= 1).all()
        tr_arm = pd.Series(arm[tr])
        n_arm = tr_arm.map(tr_arm.value_counts())
        check(bal, f"folds stratified on (arm, X, train) (P5): all {len(multi):,} groups of >= 2 rows split evenly; "
                   f"{(n_arm >= 2).mean():.1%} of train rows see their arm in the other fold; "
                   f"train rows per fold {np.bincount(folds[tr]).tolist()}")
    if um is not None:
        check(um["fit"]["split_fingerprint"] == s["split_fingerprint"], f"{args.prefix} fit on this split")
        check(um["gate"]["usable"], f"URR gate usable: {um['gate']['per_fold']}")
        for f in ("fold0", "fold1"):
            v = um[f]
            print(f"      {f}: beats const by {v['beats_constant_by']:+.4f}, alpha mean {v['alpha_mean']:.4f} "
                  f"std {v['alpha_std']:.4f}, ESS/n {v['ess_frac']:.3f}, tail k {v['tail_khat']:.3f}, "
                  f"nu gap cells {v['nu_gap_cells']}")
        if v1 and um["fit"]["X"] == []:
            for f in ("fold0", "fold1"):
                v = um[f]
                want = 1.0 / (1.0 - v["pi0_fit"])
                check(abs(v["alpha_mean_treated"] / want - 1) < args.known_answer_rtol and v["alpha_mean_dmso"] < 0.05,
                      f"v1 known answer {f}: alpha treated {v['alpha_mean_treated']:.4f} (std {v['alpha_std_treated']:.4f}) "
                      f"vs 1/(1-pi0) = {want:.4f}; DMSO {v['alpha_mean_dmso']:.4f} (max {v['alpha_max_dmso']:.3f}) vs 0")
    for name in ("dr_weights_counts.npz", "dr_weights_urr.npz"):
        wp = os.path.join(nz, name)
        if not os.path.isfile(wp):
            continue
        wz = np.load(wp)
        w = wz["w"].astype(np.float64)
        trt = ~ctl[tr]
        check(np.array_equal(wz["row_id"], tr) and str(wz["split_fingerprint"]) == s["split_fingerprint"],
              f"{name}: row_id = train_idx")
        check(np.all(w[~trt] == 1.0), f"{name}: vehicle rows w = 1 ({int((~trt).sum()):,})")
        wn = w / w.mean()
        print(f"      {name}: treated mean {w[trt].mean():.4f} std {w[trt].std():.4f} [{w[trt].min():.3f}, "
              f"{w[trt].max():.3f}]; normalised treated {wn[trt].mean():.4f}, vehicle {wn[~trt].mean():.4f}")
        if name == "dr_weights_urr.npz" and um is not None and folds is not None:
            d = [abs(w[trt & (folds[tr] == f)].mean() - um[f"fold{f}"]["alpha_mean_treated"]) for f in (0, 1)]
            check(max(d) < 1e-4, f"{name}: fold f rows scored out-of-fold (per-fold treated means = fit_urr's, "
                                 f"|diff| {max(d):.1e})")
        if v1 and name == "dr_weights_counts.npz":
            check(np.all(w == 1.0), "v1 known answer: counts weights are exactly 1 (nu = the train pool)")
        if v1 and name == "dr_weights_urr.npz":
            check(abs(w[trt].mean() / r_v1 - 1) < args.known_answer_rtol,
                  f"v1 known answer: net weights treated {w[trt].mean():.4f} vs 1/(1-pi0) = {r_v1:.4f}")
            check(abs(wn[trt].mean() - r_v1 / (1 + pi0)) < 0.01 and abs(wn[~trt].mean() - 1 / (1 + pi0)) < 0.01,
                  f"v1 known answer: normalised treated {wn[trt].mean():.4f} / vehicle {wn[~trt].mean():.4f} vs "
                  f"{r_v1 / (1 + pi0):.4f} / {1 / (1 + pi0):.4f} (plan: ~1.004 / ~0.942)")

    # === thinning instance: recomputed from the table ===========================
    base_split = os.path.join(base_nz, "splits.json")
    if tier.get("active"):
        tm = json.load(open(os.path.join(nz, "tier_meta.json")))
        dz = np.load(os.path.join(nz, "dr_weights_design.npz"))
        check(np.array_equal(dz["row_id"], tr) and str(dz["split_fingerprint"]) == s["split_fingerprint"],
              "dr_weights_design.npz row_id = train_idx, same split")
        comp, dl, syn = df["compound_idx"].values, df["dose_level"].values, df["syn_c"].values
        w_want = np.ones(tr.size)
        pos = {int(r): i for i, r in enumerate(tr)}
        dropped, ok = set(), {"rows": True, "z": True, "pi": True, "kept": True, "cells": True}
        g, kf, pmin = tm["gamma"], tm["keep_frac"], tm["pmin"]
        for c in tm["compounds"]:
            ci = c["compound_idx"]
            rows = np.array(c["rows"])
            ok["rows"] &= np.array_equal(np.sort(rows), nu[(comp[nu] == ci) & ~ctl[nu]])
            lv = sorted(set(dl[pop & ~ctl & (comp == ci)].tolist()))
            high = np.array([lv.index(v) >= len(lv) // 2 for v in dl[rows]])
            raw_z = np.where(high, 1.0, -1.0) * np.where(syn[rows] == 1, 1.0, -1.0)
            z = (raw_z - raw_z.mean()) / raw_z.std() if raw_z.std() > 1e-12 else np.zeros(rows.size)
            ok["z"] &= np.allclose(z, c["z"])
            pi = np.array(c["pi"])
            free = (pi > pmin + 1e-9) & (pi < 1 - 1e-9)
            ratio = pi[free] / np.exp(-g * z[free])
            ok["pi"] &= (ratio.size == 0 or np.ptp(ratio) <= 1e-6 * ratio.mean()) \
                and abs(pi.sum() - kf * rows.size) < 1e-4 * rows.size and pi.min() >= pmin - 1e-9
            kept = np.array(c["kept"]).astype(bool)
            ok["kept"] &= set(rows[kept].tolist()) == set(rows.tolist()) & set(tr.tolist())
            dropped |= set(rows[~kept].tolist())
            cells = np.array([f"{int(h)}|{int(sc)}" for h, sc in zip(high, syn[rows])])
            ok["cells"] &= len(set(cells)) == 4 and all(kept[cells == k].any() for k in set(cells))
            for i in np.flatnonzero(kept):
                p_c = 1.0 - np.prod(1.0 - pi[cells == cells[i]])
                w_want[pos[int(rows[i])]] = p_c / pi[i]
        check(ok["rows"], "tier rows = each scored compound's unthinned treated train rows (table)")
        check(ok["z"], "selection z = standardize(z_dose * z_syn_c) recomputed from dose_level / syn_c")
        check(ok["pi"], "pi = calibrate(exp(-gamma z)): proportional where unclipped, sum = keep_frac * n, >= pmin")
        check(ok["kept"] and set(nu.tolist()) - set(tr.tolist()) == dropped,
              f"thinned rows ({len(dropped):,}) = scored train rows not kept; kept rows stay in train")
        check(ok["cells"], "every positivity cell (compound, dose half, syn_c) keeps >= 1 train well (table)")
        check(np.allclose(dz["w"], w_want, rtol=1e-6),
              "design weights = P_c / pi on kept scored rows (inverse realised inclusion), 1 elsewhere")
    if os.path.normpath(nz) != os.path.normpath(base_nz) and os.path.isfile(base_split):
        bs = json.load(open(base_split))
        if bs["seed"] == s["seed"] and bs["population"] == s["population"] and bs["holdout_frac"] == s["holdout_frac"]:
            scored = np.zeros(N, dtype=bool)
            for ci in tier.get("scored_compounds", {}).values():
                scored |= (df["compound_idx"].values == ci) & ~ctl
            bho = np.asarray(bs["holdout_idx"])
            check(np.array_equal(bho[~scored[bho]], ho[~scored[ho]]),
                  f"holdout of every unscored stratum (DMSO included) = v1's at seed {s['seed']} "
                  f"(k_reserve {tier.get('k_reserve', 0)})")
        else:
            print("skip  base split has another seed / population / holdout_frac: no shared-holdout check")

    print(f"[check1] train {tr.size:,}  holdout {ho.size:,}  reserve {rs.size:,}  nu {nu.size:,}  pi0 {pi0:.4f}")
    if FAILS:
        print(f"[check1] {len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("[check1] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
