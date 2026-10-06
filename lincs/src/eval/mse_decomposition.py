"""Where a step-A arm's squared error comes from (`IMPLEMENT.md` §5, "MSE decomposition"). No PyTorch.

`evaluate` reports ||tau_gen(a) - tau_oracle(a)||^2 (the pooled gene MSE x 978).
The oracle is itself an arm mean over ~15 wells, so most of that number is the
oracle's noise. This script splits it, per arm and summed over genes, into terms
that add up, each estimated without a noise model:

    total  =  oracle noise  -  2 x overlap  +  model error
    model error  =  generation noise + run-to-run variance + systematic error
    systematic error (gamma > 0)  =  the same at gamma = 0  +  what the selective thinning added

Two references with INDEPENDENT noise make this possible: tau_h from the arm's
holdout wells and tau_tr from its (unthinned) train wells, both standardised to
the arm's own line mix over all wells, so both are unbiased for the truth the
all-wells oracle estimates. No generator trains on a holdout well. Then

    model error   = ||tau_gen||^2 - 2 <tau_gen, tau_h> + <tau_h, tau_tr>
    oracle noise  = ||tau_o||^2 - <tau_h, tau_tr>
    overlap       = <tau_gen - truth, oracle noise> = (model error + oracle noise - total) / 2
                    (positive when the generator copies noise from training wells
                    that are also in the oracle)
    generation noise  = sum over the arm's rows of row_var / (m - 1) / n_rows^2
                        (m samples per row; `_gen.npz`)
    run-to-run        = 1/2 ||tau_gen(seed 0) - tau_gen(seed 1)||^2 - generation noise
    systematic        = model error - generation noise - run-to-run

and, for reference (these do not add to the rows above), the size of the shift the
selective thinning causes, D = tau_gen(gamma > 0) - tau_gen(gamma = 0):

    ||D||^2 over all directions, and <D, u_a>^2 along the line-contrast direction
    (both as cross-seed products, so run noise cancels), and the square of the
    mean shift per dose half -- the quantity the step-A DiD measures.

Every mean carries a delete-one-compound jackknife SE. Runs are taken from a
`step_a_report` verdict, so the admission rules are that report's.

    python -m src.eval.mse_decomposition --data_dir data/core5_24h
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

from src.data.build_dataset import _write_json  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import arm_keys, dose_half, load_splits  # noqa: E402
from src.eval import dist_metrics as dm  # noqa: E402
from src.eval.contrast_stats import jackknife_se  # noqa: E402
from src.spec import add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args  # noqa: E402

GEN_ARMS = ("naive", "conditional", "dr", "dr_p2")
ARM_SETS = ("scored", "unscored", "all")


def group_sum(y: np.ndarray, g: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_groups, G) sums of y's rows by group id, and the group sizes."""
    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    out = np.zeros((n_groups, y.shape[1]), dtype=np.float64)
    out[gs[starts]] = np.add.reduceat(y[order].astype(np.float64), starts, axis=0)
    return out, np.bincount(g, minlength=n_groups).astype(np.float64)


def cmean(x: np.ndarray, comp_pos: np.ndarray, n_comp: int, sel: np.ndarray) -> dict:
    """Mean of x over the selected arms, with a delete-one-compound jackknife SE."""
    ok = sel & np.isfinite(x)
    if not ok.any():
        return {"mean": None, "se": None, "n_arms": 0}
    s = np.bincount(comp_pos[ok], weights=x[ok], minlength=n_comp)
    n = np.bincount(comp_pos[ok], minlength=n_comp).astype(np.float64)
    used = n > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (s.sum() - s[used]) / (n.sum() - n[used])
    return {"mean": float(s.sum() / n.sum()), "se": jackknife_se(loo), "n_arms": int(ok.sum())}


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ag,ag->a", a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verdict", default=None,
                   help="A step_a_report verdict on --pool all (default <runs>/eval_artifacts/step_a_verdict.json).")
    p.add_argument("--out", default="mse_decomposition.json", help="Relative to <runs>/eval_artifacts/, or a path.")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    t0 = time.time()
    print(f"[blas] numpy matrix products verified (rel err {dm.assert_blas_ok():.1e})", flush=True)
    cfg = apply_paths_args(config_from_args(args), args)
    ea = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    vpath = args.verdict or os.path.join(ea, "step_a_verdict.json")
    with open(vpath) as fh:
        verdict = json.load(fh)
    if verdict.get("pool") != "all" or verdict.get("population") != cfg.population.name:
        raise SystemExit(f"[mse] {vpath}: need this population's --pool all verdict")
    with open(verdict["truth"]) as fh:
        odoc = json.load(fh)
    splits = load_splits(cfg)                      # the BASE split: the oracle's frame
    if odoc.get("split_fingerprint") != splits["split_fingerprint"]:
        raise SystemExit(f"[mse] the oracle was built on split {odoc.get('split_fingerprint')}, "
                         f"not the base split {splits['split_fingerprint']}")
    oz = np.load(verdict["truth"][:-5] + "_tau.npz", allow_pickle=False)
    ak = oz["all/arm_key"].astype(str)
    tau_o = oz["all/tau"].astype(np.float64)
    con = oz["all/contrast_dir"].astype(np.float64)        # holdout wells only
    cn = np.linalg.norm(con, axis=1)
    has_u = np.isfinite(con).all(axis=1) & (cn > 0)
    u = np.zeros_like(con)
    u[has_u] = con[has_u] / cn[has_u, None]
    K, G = tau_o.shape

    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "det_plate", "cell_id"])
            .to_pandas())
    comp_t = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ic = meta["is_control"].values.astype(np.int64)
    keys = arm_keys(comp_t, dl, ic).astype(str)
    half_of = dict(zip(keys, dose_half(comp_t, dl, ic)))
    lines = sorted(set(meta["cell_id"].values))
    line_t = np.array([lines.index(c) for c in meta["cell_id"].values], dtype=np.int64)
    L = len(lines)
    m = load_expr_meta(cfg, odoc["plate_center"], splits=splits)
    y = normalize_expr(np.asarray(np.load(cfg.paths.expr_npy, mmap_mode="r")),
                       plate_codes(meta["det_plate"].values, m["plates"]), m)
    pos = {k: i for i, k in enumerate(ak)}
    arm_of_row = np.array([pos.get(k, -1) for k in keys], dtype=np.int64)
    trt = (ic == 0) & (arm_of_row >= 0)
    in_h = np.zeros(len(meta), dtype=bool)
    in_h[np.asarray(splits["holdout_idx"], dtype=np.int64)] = True
    in_tr = np.zeros(len(meta), dtype=bool)
    in_tr[np.asarray(splits["train_idx"], dtype=np.int64)] = True
    if (in_h & in_tr).any() or not (in_h | in_tr).all():
        raise SystemExit("[mse] the base split's train and holdout rows do not partition the table")

    # ---- the three references, per (arm, line) cell --------------------------------
    cell = arm_of_row * L + line_t
    ref, cnt = {}, {}
    for name, mask in (("all", trt), ("h", trt & in_h), ("tr", trt & in_tr)):
        s, n = group_sum(y[mask], cell[mask], K * L)
        cnt[name] = n.reshape(K, L)
        with np.errstate(invalid="ignore", divide="ignore"):
            ref[name] = (s / n[:, None]).reshape(K, L, G)          # nan where the cell is empty
    n_all = cnt["all"].sum(axis=1)
    P = cnt["all"] / np.maximum(n_all, 1)[:, None]                 # the arm's line mix over all wells
    complete = (n_all > 0) & ((cnt["all"] == 0) | ((cnt["h"] > 0) & (cnt["tr"] > 0))).all(axis=1)
    mu0 = {name: y[(ic == 1) & mask].astype(np.float64).mean(axis=0)
           for name, mask in (("all", np.ones(len(meta), bool)), ("h", in_h), ("tr", in_tr))}
    tau = {name: np.einsum("al,alg->ag", P, np.nan_to_num(ref[name])) - mu0[name] for name in ref}
    err = float(np.abs(tau["all"] - tau_o)[n_all > 0].max())
    print(f"[mse] the all-wells reference recomputed from rows = the stored oracle "
          f"(max |diff| {err:.1e}); {int(complete.sum()):,}/{K:,} arms have a holdout and a "
          f"train well in every one of their lines", flush=True)
    if err > 1e-3:
        raise SystemExit("[mse] the recomputed oracle differs from the stored one: wrong frame")
    tau_h, tau_tr = tau["h"], tau["tr"]
    truth_sq = dot(tau_h, tau_tr)                                  # unbiased for ||truth||^2
    oracle_noise = dot(tau_o, tau_o) - truth_sq

    comp = np.array([int(k.split("|", 1)[0]) for k in ak], dtype=np.int64)
    half = np.array([int(half_of[k]) for k in ak])
    cl = np.unique(comp)
    comp_pos = np.searchsorted(cl, comp)
    sc_c = sorted({int(v) for v in json.load(open(os.path.join(
        verdict["cells"][next(iter(verdict["cells"]))]["nuisance_dir"], "splits.json")))
        ["tier"]["scored_compounds"].values()})
    sc = np.isin(comp, np.array(sc_c, dtype=np.int64))
    sets = {"scored": complete & sc, "unscored": complete & ~sc, "all": complete}

    def agg(x, sel):
        return cmean(x, comp_pos, cl.size, sel)

    # ---- the runs ----------------------------------------------------------------------
    runs: dict[tuple, dict] = {}
    for ck, c in verdict["cells"].items():
        arm, g, s = ck.split("|")
        if arm not in GEN_ARMS:
            continue
        z = np.load(c["npz"], allow_pickle=False)
        if not np.array_equal(z["all/arm_key"].astype(str), ak):
            raise SystemExit(f"[mse] {c['npz']}: its arm table differs from the oracle's")
        tg = z["all/tau_gen"].astype(np.float64)
        gp = c["npz"][:-len("_tau.npz")] + "_gen.npz"
        gz = np.load(gp, allow_pickle=False)
        rid = gz["row_id"].astype(np.int64)
        mrep = int(gz["n_per_row"])
        rv = gz["row_var"].sum(axis=1, dtype=np.float64) / (mrep - 1)   # var of each row's mean
        a = arm_of_row[rid]
        ok = trt[rid]
        vgen = np.bincount(a[ok], weights=rv[ok], minlength=K) / np.maximum(n_all, 1) ** 2
        runs[(arm, float(g[1:]), int(s[1:]))] = {"tau": tg, "vgen": vgen}
        print(f"[mse] loaded {ck} (m = {mrep})", flush=True)
    gammas = sorted({k[1] for k in runs})
    g1 = [g for g in gammas if g > 0][0]

    out = {"population": cfg.population.name, "verdict": os.path.abspath(vpath),
           "oracle": verdict["truth"], "gamma": g1, "n_genes": G,
           "n_arms": {k: int(v.sum()) for k, v in sets.items()},
           "units": "squared error per arm, summed over genes; divide by n_genes for the pooled gene MSE",
           "oracle_noise": {k: agg(oracle_noise, v) for k, v in sets.items()},
           "truth_sq": {k: agg(truth_sq, v) for k, v in sets.items()}, "arms": {}}
    PER = {}
    for arm in GEN_ARMS:
        seeds = sorted({k[2] for k in runs if k[0] == arm})
        if not seeds:
            continue
        rec = {"seeds": seeds, "by_gamma": {}, "shift": {}}
        per = {}                                                   # per-arm vectors by gamma
        for g in gammas:
            taus = [runs[(arm, g, s)]["tau"] for s in seeds]
            total = np.mean([((t - tau_o) ** 2).sum(axis=1) for t in taus], axis=0)
            model = np.mean([dot(t, t) - 2 * dot(t, tau_h) for t in taus], axis=0) + truth_sq
            vgen = np.mean([runs[(arm, g, s)]["vgen"] for s in seeds], axis=0)
            overlap = (model + oracle_noise - total) / 2
            d = {"total": total, "model_error": model, "overlap": overlap, "generation_noise": vgen}
            if len(seeds) >= 2:
                between = 0.5 * ((taus[0] - taus[1]) ** 2).sum(axis=1)
                d["run_to_run"] = between - vgen
                d["systematic"] = model - between
            per[g] = d
            rec["by_gamma"][f"{g:g}"] = {name: {k: agg(x, v) for k, v in sets.items()}
                                          for name, x in d.items()}
        # what the selective thinning added, arm by arm
        added = {"model_error_added": per[g1]["model_error"] - per[0.0]["model_error"]}
        if len(seeds) >= 2:
            added["systematic_added"] = per[g1]["systematic"] - per[0.0]["systematic"]
        D = [runs[(arm, g1, s)]["tau"] - runs[(arm, 0.0, s)]["tau"] for s in seeds]
        if len(seeds) >= 2:
            added["shift_sq"] = dot(D[0], D[1])                    # cross-seed: run noise cancels
            added["shift_sq_along_u"] = np.where(has_u, dot(D[0], u) * dot(D[1], u), np.nan)
        else:
            added["shift_sq_raw_one_seed"] = dot(D[0], D[0])       # includes run noise: an upper bound
            added["shift_sq_along_u_raw_one_seed"] = np.where(has_u, dot(D[0], u) ** 2, np.nan)
        Du = np.where(has_u, np.mean([dot(x, u) for x in D], axis=0), np.nan)
        for k, v in sets.items():
            r = {name: agg(x, v) for name, x in added.items()}
            lo, hi = agg(Du, v & (half == 0)), agg(Du, v & (half == 1))
            if lo["mean"] is not None and hi["mean"] is not None:
                w = hi["n_arms"] / (hi["n_arms"] + lo["n_arms"])
                r["mean_shift_along_u"] = {"low": lo["mean"], "high": hi["mean"],
                                           "did": hi["mean"] - lo["mean"],
                                           "squared": w * hi["mean"] ** 2 + (1 - w) * lo["mean"] ** 2}
            rec["shift"][k] = r
        out["arms"][arm] = rec
        PER[arm] = per
    # Paired with `conditional`, arm by arm: the references are common, so their
    # noise cancels in the difference.
    for arm, per in PER.items():
        if arm == "conditional" or "conditional" not in PER:
            continue
        out["arms"][arm]["minus_conditional"] = {
            f"{g:g}": {name: {k: agg(per[g][name] - PER["conditional"][g][name], v)
                              for k, v in sets.items()}
                       for name in ("total", "model_error")}
            for g in gammas}

    def f(d, nd=3):
        if d is None or d.get("mean") is None:
            return "n/a"
        return f"{d['mean']:+.{nd}f} ± {d['se']:.{nd}f}" if d.get("se") is not None else f"{d['mean']:+.{nd}f}"

    for k in ARM_SETS:
        arms = list(out["arms"])
        print(f"\n### {k} arms ({out['n_arms'][k]:,}), gamma = {g1:g}: squared error per arm, summed over {G} genes\n")
        print("| term | " + " | ".join(f"`{a}`" for a in arms) + " |")
        print("|---|" + "---|" * len(arms))
        B = {a: out["arms"][a]["by_gamma"][f"{g1:g}"] for a in arms}
        B0 = {a: out["arms"][a]["by_gamma"]["0"] for a in arms}
        S = {a: out["arms"][a]["shift"][k] for a in arms}
        rows = [("total against the all-wells oracle", lambda a: f(B[a]["total"][k])),
                ("oracle noise", lambda a: f(out["oracle_noise"][k])),
                ("overlap (enters as -2x)", lambda a: f(B[a]["overlap"][k])),
                ("model error against the truth", lambda a: f(B[a]["model_error"][k])),
                ("- generation noise", lambda a: f(B[a]["generation_noise"][k], 4)),
                ("- run-to-run variance", lambda a: f(B[a].get("run_to_run", {}).get(k))),
                ("- systematic error", lambda a: f(B[a].get("systematic", {}).get(k))),
                ("-- of it, present at gamma = 0", lambda a: f(B0[a].get("systematic", {}).get(k))),
                ("-- of it, added by the selective thinning", lambda a: f(S[a].get("systematic_added"))),
                ("model error minus `conditional`'s (paired)", lambda a: (
                    f(out["arms"][a]["minus_conditional"][f"{g1:g}"]["model_error"][k])
                    if "minus_conditional" in out["arms"][a] else "0")),
                ("the same at gamma = 0", lambda a: (
                    f(out["arms"][a]["minus_conditional"]["0"]["model_error"][k])
                    if "minus_conditional" in out["arms"][a] else "0")),
                ("model error added by the selective thinning", lambda a: f(S[a]["model_error_added"])),
                ("shift ||D||^2, all directions", lambda a: f(S[a].get("shift_sq") or S[a].get("shift_sq_raw_one_seed"))),
                ("shift along the line contrast, <D,u>^2", lambda a: f(S[a].get("shift_sq_along_u") or S[a].get("shift_sq_along_u_raw_one_seed"), 4)),
                ("(mean shift per dose half)^2: what the DiD reads", lambda a: (
                    f"{S[a]['mean_shift_along_u']['squared']:.5f} (DiD {S[a]['mean_shift_along_u']['did']:+.3f})"
                    if "mean_shift_along_u" in S[a] else "n/a"))]
        for name, fn in rows:
            print(f"| {name} | " + " | ".join(fn(a) for a in arms) + " |")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    op = args.out if os.path.isabs(args.out) or os.sep in args.out else os.path.join(ea, args.out)
    _write_json(op, out)
    print(f"\n[mse] -> {op}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
