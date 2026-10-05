"""Step A's power gate S0 (`STEP_A.md` §3): is the confounding big enough to see? No PyTorch.

Before any GPU job, this reads the bias the realised thinning puts into the
TRAINING DATA, with the same statistic `step_a_report` will apply to the
generators:

    memoriser   tau_mem(a) = mean of arm a's kept train wells - mu_hat(0)
    b_a         = <tau_mem(a) - tau_oracle(a), u_a>,  u_a = the G2 - G1 line-group
                  contrast of arm a over its HOLDOUT wells, unit norm
    DiD         = [mean b | high half - mean b | low half] at gamma > 0
                  minus the same at gamma = 0, on scored arms

The direction comes from holdout wells only (the oracle's `contrast_dir`), so the
noise the memoriser copies from its kept wells is independent of it. With a
direction from the pool's own wells the memoriser would show dp * ||c_a|| with
||c_a|| inflated by replicate noise, and S0 could pass with no line effect at
all (review, IMPLEMENT.md §5; on the --limit build that inflated it ~2.4x).

A generator blind to the line can show at most what its data holds, and in steps
C / C2 it showed 65% and 18% of it, so S0 asks for a wide margin:

    S0 passes when |DiD| >= `--min_ratio` (5) x its compound-clustered SE.

It also reports, as diagnostics and not as gates:
  * `planned`: the same DiD from the line mix alone, dp_a * <c_nu(a), u_a>, where
    dp_a is the shift of the G2 share among the arm's kept wells and c_nu the
    contrast over the UNTHINNED train wells -- disjoint from the holdout wells
    behind u_a, so it is unbiased for dp_a * <c_true, u_a>. It should agree with
    the memoriser, which is the cross-check.
  * the WEIGHTED memorisers (counts and design weights). In step A the target
    group is the arm, so weights that balance the lines within an arm should take
    the memoriser's DiD to ~0 -- the data-level statement that a weighted risk
    can work here, which it could not in step C2 (IMPLEMENT.md §5).

    python -m src.eval.step_a_power --data_dir data/core5_24h
    python -m src.eval.step_a_power --data_dir D --tier0 DIR --tier1 DIR --oracle O.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import _atomic_write  # noqa: E402
from src.data.build_tiered_split import tier_dir  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import SPLITS_FILENAME, arm_keys, dose_half, load_splits  # noqa: E402
from src.eval import dist_metrics as dm  # noqa: E402
from src.eval.contrast_stats import did  # noqa: E402
from src.eval.dr_target import hajek_delta  # noqa: E402
from src.eval.evaluate import group_contrast  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args,
    line_group_sign, line_groups)

WEIGHT_SETS = ("ones", "counts", "design")


def find_tier(cfg, gamma: float, seed: int) -> str:
    """The step-A tier instance of this build at `gamma`."""
    d = tier_dir(cfg, "cell_id", gamma, seed)
    if not os.path.isfile(os.path.join(d, SPLITS_FILENAME)):
        raise SystemExit(f"[power] no tier instance at {d}; pass --tier0 / --tier1")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tier0", default=None, help="The gamma = 0 tier dir (default: discovered).")
    p.add_argument("--tier1", default=None, help="The gamma > 0 tier dir (default: discovered at --gamma).")
    p.add_argument("--gamma", type=float, default=1.0, help="The confounded instance's gamma, for discovery.")
    p.add_argument("--oracle", default=None, help="Default: <runs>/eval_artifacts/oracle_<population>_poolall.json")
    p.add_argument("--pool", default="all", choices=("all",))
    p.add_argument("--min_ratio", type=float, default=5.0, help="S0: |DiD| / clustered SE must reach this.")
    p.add_argument("--out", default="step_a_power.json", help="Relative to <runs>/eval_artifacts/, or a path.")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    t0 = time.time()
    print(f"[blas] numpy matrix products verified (rel err {dm.assert_blas_ok():.1e})", flush=True)
    cfg = apply_paths_args(config_from_args(args), args)
    lg = line_groups(cfg.population)
    if lg is None:
        raise SystemExit(f"[power] population {cfg.population.name!r} declares no line groups")
    splits = load_splits(cfg)                     # the BASE split: the oracle's frame
    oracle = args.oracle or os.path.join(cfg.paths.train_output_dir, "eval_artifacts",
                                         f"oracle_{cfg.population.name}_pool{args.pool}.json")
    with open(oracle) as fh:
        odoc = json.load(fh)
    if odoc.get("source") != "real" or odoc.get("population") != cfg.population.name:
        raise SystemExit(f"[power] {oracle}: not this population's --source real oracle")
    if odoc.get("split_fingerprint") != splits["split_fingerprint"]:
        raise SystemExit(f"[power] {oracle} was built on split {odoc.get('split_fingerprint')}, "
                         f"not the base split {splits['split_fingerprint']}")
    if float(odoc.get("syn_effect") or 0) != 0:
        raise SystemExit(f"[power] {oracle} carries a synthetic injection; step A uses the plain oracle")
    oz = np.load(oracle[:-5] + "_tau.npz", allow_pickle=False)
    pre = f"{args.pool}/"
    if pre + "contrast_dir" not in oz:
        raise SystemExit(f"[power] {oracle} has no holdout-well contrast direction; rebuild it "
                         f"with the current evaluate.py")
    ak = oz[pre + "arm_key"].astype(str)
    tau_o = oz[pre + "tau"].astype(np.float64)
    mu0 = oz[pre + "mu0"].astype(np.float64)
    con = oz[pre + "contrast_dir"].astype(np.float64)      # holdout wells only
    cn = np.linalg.norm(con, axis=1)
    has_c = np.isfinite(con).all(axis=1) & (cn > 0)
    u = np.zeros_like(con)
    u[has_c] = con[has_c] / cn[has_c, None]

    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "det_plate", "cell_id"])
            .to_pandas())
    comp_t = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ic = meta["is_control"].values.astype(np.int64)
    keys = arm_keys(comp_t, dl, ic).astype(str)
    half_of = dict(zip(keys, dose_half(comp_t, dl, ic)))   # over the WHOLE table
    g2 = line_group_sign(meta["cell_id"].values, cfg.population) > 0
    m = load_expr_meta(cfg, odoc["plate_center"], splits=splits)
    y = normalize_expr(np.asarray(np.load(cfg.paths.expr_npy, mmap_mode="r")),
                       plate_codes(meta["det_plate"].values, m["plates"]), m)

    comp = np.array([int(k.split("|", 1)[0]) for k in ak], dtype=np.int64)
    half = np.array([int(half_of[k]) for k in ak])
    pos = {k: i for i, k in enumerate(ak)}
    arm_of_row = np.array([pos.get(k, -1) for k in keys], dtype=np.int64)
    trt = ic == 0
    # the pool's G2 share per arm (what the oracle's line mix is)
    in_arm = trt & (arm_of_row >= 0)
    n_pool = np.bincount(arm_of_row[in_arm], minlength=ak.size).astype(np.float64)
    p_pool = np.bincount(arm_of_row[in_arm], weights=g2[in_arm], minlength=ak.size) / np.maximum(n_pool, 1)

    # the contrast over the unthinned train wells, for `planned`
    nu_base = np.asarray(splits["train_idx"], dtype=np.int64)
    sign_all = np.where(g2, 1.0, -1.0)
    c_nu = group_contrast(y[nu_base], keys[nu_base], sign_all[nu_base], ak)["contrast"]
    c_nu_u = np.einsum("ag,ag->a", np.nan_to_num(c_nu), u)
    c_nu_u = np.where(np.isfinite(c_nu).all(axis=1) & has_c, c_nu_u, np.nan)

    seed = int(splits["seed"])
    tiers = {0.0: args.tier0 or find_tier(cfg, 0.0, seed),
             float(args.gamma): args.tier1 or find_tier(cfg, float(args.gamma), seed)}
    scored_sets, B, tier_rec, DP = [], {}, {}, {}
    for g, tdir in tiers.items():
        with open(os.path.join(tdir, SPLITS_FILENAME)) as fh:
            ts = json.load(fh)
        tier = ts["tier"]
        if tier.get("confounder") != "cell_id" or abs(float(tier["gamma"]) - g) > 1e-12:
            raise SystemExit(f"[power] {tdir}: tier {tier.get('confounder')!r} gamma "
                             f"{tier.get('gamma')}, expected cell_id / {g:g}")
        if tier.get("line_groups") != {k: list(v) for k, v in lg.items()}:
            raise SystemExit(f"[power] {tdir}: its line groups {tier.get('line_groups')} are not "
                             f"spec.LINE_GROUPS {lg}")
        if ts["holdout_idx"] != splits["holdout_idx"].tolist():
            raise SystemExit(f"[power] {tdir}: its holdout is not the base split's, so its arms "
                             f"are not scored against this oracle's frame")
        tr = np.asarray(ts["train_idx"], dtype=np.int64)
        scored_sets.append(sorted(int(v) for v in tier["scored_compounds"].values()))
        rows = tr[trt[tr] & (arm_of_row[tr] >= 0)]
        gi = arm_of_row[rows]
        w_of = {"ones": np.ones(tr.size)}
        for name in ("counts", "design"):
            wz = np.load(os.path.join(tdir, f"dr_weights_{name}.npz"))
            if not np.array_equal(wz["row_id"], tr):
                raise SystemExit(f"[power] {tdir}/dr_weights_{name}.npz: row_id != train_idx")
            w_of[name] = wz["w"].astype(np.float64)
        sel_tr = trt[tr] & (arm_of_row[tr] >= 0)
        n_kept = np.bincount(gi, minlength=ak.size).astype(np.float64)
        p_kept = np.bincount(gi, weights=g2[rows], minlength=ak.size) / np.maximum(n_kept, 1)
        dp = np.where(n_kept > 0, p_kept - p_pool, np.nan)
        DP[g] = dp
        B[(g, "planned")] = dp * c_nu_u
        for name, w_all in w_of.items():
            mu, sw, _, _ = hajek_delta(y[rows], w_all[sel_tr], gi, ak.size)
            b = np.einsum("ag,ag->a", (mu - mu0) - tau_o, u)
            B[(g, name)] = np.where(has_c & (sw > 0), b, np.nan)
        tier_rec[f"{g:g}"] = {"dir": tdir, "n_train": int(tr.size),
                              "split_fingerprint": ts["split_fingerprint"],
                              "n_arms_without_kept_train": int((n_kept[n_pool > 0] == 0).sum())}
    if scored_sets[0] != scored_sets[1]:
        raise SystemExit("[power] the two tier instances score different compounds")
    scored_c = np.array(scored_sets[0], dtype=np.int64)
    sc = np.isin(comp, scored_c)
    g1 = float(args.gamma)
    # The lever itself: the shift of the G2 share among kept wells, by dose half.
    for g, dp in DP.items():
        tier_rec[f"{g:g}"]["dp_G2_share_scored"] = {
            hn: (float(np.nanmean(dp[sc & (half == h)])) if (sc & (half == h)).any() else None)
            for hn, h in (("low", 0), ("high", 1))}

    res = {}
    for name in ("ones", "planned", "counts", "design"):
        d = did(B[(g1, name)], B[(0.0, name)], comp, half, sc, scored_c)
        un = did(B[(g1, name)], B[(0.0, name)], comp, half, ~sc)
        res[name] = {"did": d["value"], "se": d["se"],
                     "ratio": (abs(d["value"]) / d["se"] if d["se"] else None),
                     "contrast_g1": d["contrast_g1"], "contrast_g0": d["contrast_g0"],
                     "n_arms": d["n_arms"], "n_compounds": d["n_clusters"],
                     "did_unscored": un["value"], "se_unscored": un["se"]}
    mem = res["ones"]
    s0 = bool(mem["ratio"] is not None and mem["ratio"] >= args.min_ratio)
    doc = {
        "population": cfg.population.name, "pool": args.pool, "oracle": os.path.abspath(oracle),
        "line_groups": {k: list(v) for k, v in lg.items()}, "gamma": g1,
        "table_fingerprint": odoc.get("table_fingerprint"),
        "split_fingerprint": splits["split_fingerprint"],
        "tiers": tier_rec, "n_scored_compounds": int(scored_c.size),
        "n_scored_arms": int(sc.sum()), "n_scored_arms_with_contrast": int((sc & has_c).sum()),
        "direction": "G2 - G1 contrast over the holdout wells (oracle contrast_dir)",
        "contrast_dir_norm_scored_median": float(np.median(cn[sc & has_c])) if (sc & has_c).any() else None,
        "train_contrast_on_direction_scored_mean": (float(np.nanmean(c_nu_u[sc]))
                                                    if np.isfinite(c_nu_u[sc]).any() else None),
        "statistic": "DiD (gamma > 0 minus gamma = 0) of the mean high - low contrast of "
                     "<tau - tau_oracle, u_a> on scored arms; SE by delete-one-compound jackknife",
        "memoriser": mem, "planned": res["planned"],
        "weighted_memorisers": {k: res[k] for k in ("counts", "design")},
        "S0": {"pass": s0, "min_ratio": args.min_ratio, "ratio": mem["ratio"],
               "rule": f"|memoriser DiD| >= {args.min_ratio:g} x its compound-clustered SE "
                       f"(STEP_A.md §3)"},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = args.out if os.path.sep in args.out else os.path.join(
        cfg.paths.train_output_dir, "eval_artifacts", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _atomic_write(out, lambda f: json.dump(doc, f, indent=2))

    f = lambda r: (f"{r['did']:+.3f} +/- {r['se']:.3f}" if r["se"] is not None else f"{r['did']:+.3f}")
    print(f"[power] {cfg.population.name}: {scored_c.size:,} scored compounds, {int((sc & has_c).sum()):,} "
          f"scored arms with a holdout-well direction; <c_train, u> mean "
          f"{doc['train_contrast_on_direction_scored_mean']:.3f} (the line effect per unit dp)")
    print(f"[power] memoriser (unweighted)   DiD {f(mem)}   ratio {mem['ratio'] and round(mem['ratio'], 1)}   "
          f"[g1 {mem['contrast_g1']:+.3f}, g0 {mem['contrast_g0']:+.3f}]   unscored {mem['did_unscored']:+.3f}")
    print(f"[power] planned (dp x <c_nu, u>) DiD {f(res['planned'])}")
    for k in ("counts", "design"):
        print(f"[power] memoriser, {k:6s} weights DiD {f(res[k])}")
    print(f"[power] S0 {'PASS' if s0 else 'FAIL'}: |DiD| / SE = "
          f"{mem['ratio'] and round(mem['ratio'], 1)} (needs >= {args.min_ratio:g})")
    print(f"[power] -> {out}")
    if not s0:
        sys.exit(3)


if __name__ == "__main__":
    main()
