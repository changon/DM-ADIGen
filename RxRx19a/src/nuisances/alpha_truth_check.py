"""KNOWN-TRUTH check for the fitted Riesz alpha under an imposed ablation.

`fit_urr.py` minimises

    L(a) = E_{(X,A) ~ P_fit}[ a(X,A)^2 ] - 2 E_{X ~ P_fit, At ~ nu}[ a(X,At) ]

whose pointwise minimiser is

    alpha*(x,a) = nu(a) / f_fit(a|x)

Note there are two truths.
  truth_train  nu drawn from the ABLATED pool  (what the nets on disk were fit to)
  truth_full   nu drawn from the FULL pool     (what --nu_source full targets)

we can compare to these here.

    python -m src.nuisances.alpha_truth_check \
        --rare-compound-frac 1.0 --rare-keep-frac 0.4 --rare-seed 42 \
        --rare-confound-pos-gamma 1.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.rarity import (  # noqa: E402
    add_rarity_cli_args,
    rarity_from_args,
    tagged_nuisance_dir,
)
from src.data.splits import load_splits  # noqa: E402
from src.nuisances.alpha_net import AlphaNet  # noqa: E402
from src.spec import default_config  # noqa: E402


def _wcorr(x, y, w):
    """Weighted Pearson correlation (weights = rows behind each cell)."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    vx = (w * (x - mx) ** 2).sum()
    vy = (w * (y - my) ** 2).sum()
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt(vx * vy))


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return _wcorr(rx, ry, np.ones_like(rx))


def _ols_slope(x, y, w):
    """Slope of y on x. 1.0 = perfectly calibrated in log space."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    vx = (w * (x - mx) ** 2).sum()
    if vx <= 0:
        return float("nan")
    return float((w * (x - mx) * (y - my)).sum() / vx)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--prefix", default="alpha_urr",
                   help="Basename of the nets to check (alpha_urr -> "
                        "alpha_urr_fold{0,1}.pt).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--cov_blocks", default="cell_type,experiment")
    p.add_argument("--target_support", default="common", choices=("common", "all"))
    p.add_argument("--min_cell_rows", type=int, default=5,
                   help="Skip (stratum, action) cells with fewer surviving rows: "
                        "their empirical f(a|x) is too noisy to be a 'truth'.")
    p.add_argument("--report_compounds", default="GS-441524,Camostat,Chloroquine",
                   help="Comma-separated names for the per-dose table.")
    p.add_argument("--out", default=None)
    add_rarity_cli_args(p)
    args = p.parse_args()

    cfg = default_config()
    rcfg = rarity_from_args(args)
    dev = torch.device(args.device)
    base_nz = cfg.paths.nuisance_dir
    if rcfg.active:
        cfg.paths.nuisance_dir = tagged_nuisance_dir(cfg, rcfg)
    nz = cfg.paths.nuisance_dir
    print(f"[truth] arm = {rcfg.tag or 'FULL DATA'} -> {nz}")

    meta = load_from_disk(cfg.paths.tabular_dataset_dir)
    cov_all = np.asarray(meta["cov_vec"], dtype=np.float32)
    comp_all = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx_all = np.asarray(meta["log10_conc"], dtype=np.float32)
    ic_all = np.asarray(meta["is_control"], dtype=np.float32)
    dis = np.array([str(x) for x in meta["disease_condition"]])
    inf_all = (dis == cfg.population.disease_condition).astype(np.float32)
    cov_dim = cov_all.shape[1]
    with open(os.path.join(nz, "nuisance_meta.json")) as f:
        n_compounds = int(json.load(f)["n_compounds"])

    # ---- the same population fit_urr fit on -------------------------------
    train_idx = np.asarray(load_splits(cfg)["train_idx"], dtype=np.int64)
    folds = np.load(os.path.join(nz, "fold_assignment.npy"))
    _inf_tr = inf_all[train_idx] == 1
    if len(folds) == len(train_idx):
        folds = folds[_inf_tr]
    train_idx = train_idx[_inf_tr]
    fold_tr = (folds if len(folds) == len(train_idx) else folds[train_idx]).astype(np.int64)

    # the full training set
    full_train = np.asarray(
        json.load(open(os.path.join(base_nz, "splits.json")))["train_idx"],
        dtype=np.int64)
    full_train = full_train[inf_all[full_train] == 1]
    print(f"[truth] ablated train (infected) = {len(train_idx):,}   "
          f"full train (infected) = {len(full_train):,}   "
          f"survival = {len(train_idx) / max(len(full_train), 1):.3f}")

    # ---- strata and nu support ------------------
    with open(os.path.join(base_nz, "covariate_encoder.json")) as f:
        enc = json.load(f)
    blocks, off = {}, 0
    for c in enc["categorical_cols"]:
        w = len(enc["levels"][c])
        blocks[c] = list(range(off, off + w))
        off += w
    keep = [b.strip() for b in args.cov_blocks.split(",") if b.strip()]
    cov_idx = sorted(i for b in keep for i in blocks[b])

    strata_key = [tuple(r) for r in (cov_all[:, cov_idx] > 0.5).astype(np.int8)]
    per_stratum: dict[tuple, set] = {}
    for s, c in zip(strata_key, comp_all):
        per_stratum.setdefault(s, set()).add(int(c))
    common = set.intersection(*per_stratum.values()) if per_stratum else set()
    nu_compounds = common if args.target_support == "common" else set(comp_all.tolist())
    nu_mask_all = np.isin(comp_all, np.fromiter(nu_compounds, dtype=np.int64))
    print(f"[truth] X = {keep} -> {len(per_stratum)} strata; "
          f"nu support = {len(nu_compounds)} compounds")

    # dose is a fixed grid -> exact discrete action key
    dose_key = np.round(lx_all.astype(np.float64), 4)
    act = list(zip(comp_all.tolist(), dose_key.tolist(), ic_all.tolist()))
    strat_id = {s: i for i, s in enumerate(sorted(per_stratum))}
    sid = np.array([strat_id[s] for s in strata_key], dtype=np.int64)

    vocab_path = os.path.join(base_nz, "compound_vocab.json")
    name_of = {}
    if os.path.exists(vocab_path):
        v = json.load(open(vocab_path))
        v = v.get("compound_to_idx", v)
        if isinstance(v, dict):
            for k, i in v.items():
                try:
                    name_of[int(i)] = str(k)
                except (TypeError, ValueError):
                    pass
        elif isinstance(v, list):
            name_of = {i: str(k) for i, k in enumerate(v)}

    want_names = {n.strip().lower() for n in args.report_compounds.split(",") if n.strip()}
    want_ids = {i for i, n in name_of.items() if n.lower() in want_names}

    out = {"arm": rcfg.tag or "full", "nuisance_dir": nz, "folds": {},
           "rarity": rcfg.to_dict()}
    rows_report = []

    for f_ in (0, 1):
        net_path = os.path.join(nz, f"{args.prefix}_fold{f_}.pt")
        if not os.path.exists(net_path):
            print(f"[truth] fold {f_}: {net_path} missing -- skipped")
            continue

        # reconstruct
        fit_rows = train_idx[fold_tr != f_]
        rng = np.random.default_rng(args.seed + f_)
        perm = rng.permutation(len(fit_rows))
        nv = max(512, int(args.val_frac * len(fit_rows)))
        ti = fit_rows[perm[nv:]]
        ti_nu = ti[nu_mask_all[ti]]

        # counts: f(a|x) over the fit rows, nu(a) over the two candidate pools
        n_x = defaultdict(int)
        n_xa = defaultdict(int)
        for i in ti:
            n_x[sid[i]] += 1
            n_xa[(sid[i], act[i])] += 1
        nu_abl = defaultdict(int)
        for i in ti_nu:
            nu_abl[act[i]] += 1
        n_nu_abl = len(ti_nu)

        full_nu_rows = full_train[nu_mask_all[full_train]]
        nu_full = defaultdict(int)
        for i in full_nu_rows:
            nu_full[act[i]] += 1
        n_nu_full = len(full_nu_rows)

        # full-pool denominator counts, for the per-dose survival column
        n_xa_full = defaultdict(int)
        for i in full_train:
            n_xa_full[(sid[i], act[i])] += 1

        net = AlphaNet.load(net_path, map_location=dev)
        net.to(dev).eval()

        # one representative row per cell: every row in a cell feeds the net same inp
        rep = {}
        for i in ti:
            rep.setdefault((sid[i], act[i]), i)

        cells = [k for k, n in n_xa.items()
                 if n >= args.min_cell_rows and nu_abl.get(k[1], 0) > 0]
        if not cells:
            print(f"[truth] fold {f_}: no cells pass the filters")
            continue
        ridx = np.array([rep[k] for k in cells], dtype=np.int64)
        with torch.no_grad():
            fitted = net(
                torch.tensor(cov_all[ridx], device=dev),
                torch.tensor(comp_all[ridx], device=dev),
                torch.tensor(lx_all[ridx], device=dev),
                torch.tensor(ic_all[ridx], device=dev),
                torch.tensor(inf_all[ridx], device=dev),
            ).cpu().numpy().astype(np.float64)

        w = np.array([n_xa[k] for k in cells], dtype=np.float64)
        f_cond = np.array([n_xa[k] / n_x[k[0]] for k in cells], dtype=np.float64)
        t_train = np.array([(nu_abl[k[1]] / n_nu_abl) for k in cells]) / f_cond
        t_full = np.array([(nu_full.get(k[1], 0) / n_nu_full) for k in cells]) / f_cond

        ok = (t_train > 0) & (t_full > 0) & (fitted > 0)
        lf, lt, lu = np.log(fitted[ok]), np.log(t_train[ok]), np.log(t_full[ok])
        ww = w[ok]

        res = {
            "n_cells": int(len(cells)),
            "n_cells_scored": int(ok.sum()),
            "n_rows_behind": int(w.sum()),
            "corr_log_fitted_vs_truth_train": _wcorr(lt, lf, ww),
            "corr_log_fitted_vs_truth_full": _wcorr(lu, lf, ww),
            "spearman_fitted_vs_truth_train": _spearman(t_train[ok], fitted[ok]),
            "spearman_fitted_vs_truth_full": _spearman(t_full[ok], fitted[ok]),
            "slope_log_fitted_on_truth_train": _ols_slope(lt, lf, ww),
            "slope_log_fitted_on_truth_full": _ols_slope(lu, lf, ww),
            "geo_mean_ratio_fitted_over_truth_train": float(
                np.exp(np.average(lf - lt, weights=ww))),
            "geo_mean_ratio_fitted_over_truth_full": float(
                np.exp(np.average(lf - lu, weights=ww))),
            "median_abs_log2_err_vs_truth_train": float(
                np.median(np.abs((lf - lt) / np.log(2)))),
            "median_abs_log2_err_vs_truth_full": float(
                np.median(np.abs((lf - lu) / np.log(2)))),
            "row_mean_fitted": float(np.average(fitted, weights=w)),
            "row_mean_truth_train": float(np.average(t_train, weights=w)),
            "row_mean_truth_full": float(np.average(t_full, weights=w)),
            "truth_full_over_truth_train_p10_p50_p90": [
                float(np.percentile(t_full[ok] / t_train[ok], q)) for q in (10, 50, 90)],
        }
        out["folds"][f"fold{f_}"] = res

        print(f"\n[truth] ===== fold {f_} =====")
        print(f"  cells={res['n_cells_scored']:,} over {res['n_rows_behind']:,} rows")
        print(f"  vs truth_train (nu = ABLATED pool, what this net was fit to):")
        print(f"     corr(log) = {res['corr_log_fitted_vs_truth_train']:+.3f}   "
              f"spearman = {res['spearman_fitted_vs_truth_train']:+.3f}   "
              f"slope = {res['slope_log_fitted_on_truth_train']:+.3f}   "
              f"median |err| = {res['median_abs_log2_err_vs_truth_train']:.2f} log2")
        print(f"  vs truth_full  (nu = FULL pool, the --nu_source full target):")
        print(f"     corr(log) = {res['corr_log_fitted_vs_truth_full']:+.3f}   "
              f"spearman = {res['spearman_fitted_vs_truth_full']:+.3f}   "
              f"slope = {res['slope_log_fitted_on_truth_full']:+.3f}   "
              f"median |err| = {res['median_abs_log2_err_vs_truth_full']:.2f} log2")
        print(f"  E_P[alpha]: fitted {res['row_mean_fitted']:.3f}  "
              f"truth_train {res['row_mean_truth_train']:.3f}  "
              f"truth_full {res['row_mean_truth_full']:.3f}   (should be ~1)")
        print(f"  truth_full / truth_train  p10/p50/p90 = "
              + "/".join(f"{v:.2f}" for v in res["truth_full_over_truth_train_p10_p50_p90"]))

        # ---- THE HEADLINE: the WITHIN-(stratum, compound) DOSE SLOPE --------
        by_cd = defaultdict(list)
        for k, fit_v, tt, tf in zip(cells, fitted, t_train, t_full):
            if tt > 0 and tf > 0 and fit_v > 0:
                by_cd[(k[0], k[1][0])].append((k[1][1], fit_v, tt, tf))
        sl_fit, sl_tr, sl_fu = [], [], []
        for grp in by_cd.values():
            if len(grp) < 3:
                continue
            d = np.array([g[0] for g in grp], dtype=np.float64)
            if d.max() - d.min() < 0.5:
                continue
            one = np.ones_like(d)
            sl_fit.append(_ols_slope(d, np.log(np.array([g[1] for g in grp])), one))
            sl_tr.append(_ols_slope(d, np.log(np.array([g[2] for g in grp])), one))
            sl_fu.append(_ols_slope(d, np.log(np.array([g[3] for g in grp])), one))
        if sl_fit:
            sl_fit, sl_tr, sl_fu = map(np.array, (sl_fit, sl_tr, sl_fu))
            res["dose_slope"] = {
                "n_groups": int(len(sl_fit)),
                "median_slope_fitted": float(np.median(sl_fit)),
                "median_slope_truth_train": float(np.median(sl_tr)),
                "median_slope_truth_full": float(np.median(sl_fu)),
                "corr_slope_fitted_vs_truth_train": _spearman(sl_tr, sl_fit),
                "corr_slope_fitted_vs_truth_full": _spearman(sl_fu, sl_fit),
                "frac_sign_agree_truth_full": float(
                    np.mean(np.sign(sl_fit) == np.sign(sl_fu))),
            }
            ds = res["dose_slope"]
            print(f"  WITHIN-compound dose slope of log(alpha) "
                  f"({ds['n_groups']} stratum x compound groups):")
            print(f"     fitted {ds['median_slope_fitted']:+.3f}   "
                  f"truth_train {ds['median_slope_truth_train']:+.3f}   "
                  f"truth_full {ds['median_slope_truth_full']:+.3f}  (per log10 dose)")
            print(f"     rank-corr of slopes: vs truth_train "
                  f"{ds['corr_slope_fitted_vs_truth_train']:+.3f}   vs truth_full "
                  f"{ds['corr_slope_fitted_vs_truth_full']:+.3f}   "
                  f"sign agreement vs truth_full {ds['frac_sign_agree_truth_full']:.2f}")

        if f_ == 0 and want_ids:
            for k, fit_v, tt, tf in zip(cells, fitted, t_train, t_full):
                cid = k[1][0]
                if cid in want_ids:
                    nf = n_xa_full.get(k, 0)
                    rows_report.append({
                        "compound": name_of.get(cid, str(cid)),
                        "stratum": int(k[0]),
                        "log10_conc": k[1][1],
                        "n_full": int(nf),
                        "n_kept": int(n_xa[k]),
                        "survival": float(n_xa[k] / nf) if nf else float("nan"),
                        "truth_train": float(tt),
                        "truth_full": float(tf),
                        "fitted": float(fit_v),
                    })

    if rows_report:
        rows_report.sort(key=lambda r: (r["compound"], r["stratum"], r["log10_conc"]))
        print("\n[truth] ===== per-dose detail (fold 0) =====")
        print(f"  {'compound':<14}{'str':>4}{'dose':>7}{'n_full':>8}{'n_kept':>8}"
              f"{'surv':>7}{'truth_tr':>10}{'truth_fu':>10}{'fitted':>9}")
        for r in rows_report:
            print(f"  {r['compound'][:13]:<14}{r['stratum']:>4}{r['log10_conc']:>7.2f}"
                  f"{r['n_full']:>8}{r['n_kept']:>8}{r['survival']:>7.3f}"
                  f"{r['truth_train']:>10.3f}{r['truth_full']:>10.3f}{r['fitted']:>9.3f}")
    out["per_dose_detail"] = rows_report

    dest = args.out or os.path.join(nz, f"{args.prefix}_truth_check.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[truth] wrote {dest}")


if __name__ == "__main__":
    main()
