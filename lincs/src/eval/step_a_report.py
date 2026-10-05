"""Step-A verdict: criteria S1-S5 of `STEP_A.md` §3 from the scored arm matrix. No PyTorch.

Collects the scored (and P1-targeted) eval JSONs of a `core5_24h` build, pivots
them on (arm, gamma, seed), and judges the criteria on

    b_a   = <tau_est(a) - tau_oracle(a), u_a>       (evaluate's `contrast_proj`)
    DiD   = [mean b | high half - mean b | low half] at gamma > 0
            minus the same at gamma = 0, on scored arms, seeds averaged per arm

with every error a delete-one-compound jackknife (`contrast_stats`): the
thinning is drawn per compound, `naive` has one seed, and the spread of 2-3
seeds that share one thinning draw understates the error (IMPLEMENT.md §5, "P1
results"). The seed spread is reported next to it, not used.

    S1  naive is biased:              |DiD| > 3 SE
    S2  testability:                  |DiD(conditional)| > 2 SE; if not, "DR vs
                                      conditional" is NOT TESTABLE here, not failed
    S3  an arm beats its baseline:    |DiD| smaller, by > 2 SE of the paired
                                      difference. Judged for dr and dr_p2
                                      (ADIGen) and for the P1 arms (baselines)
    S4  unscored compounds unchanged: |DiD_unscored| < 0.25 x naive's scored |DiD|,
                                      or within 2 SE of zero
    S5  weights:                      every weighted or targeted run's weight
                                      hash still matches its file

A JSON is admitted only if it was scored at the §3.14 settings (`SCORING`), on
this population, without a synthetic injection, against the same oracle as every
other admitted JSON, and from a run thinned on `cell_id` with the line groups
spec.py declares. Two JSONs for one (arm, gamma, seed) are an error.

    python -m src.eval.step_a_report --data_dir data/core5_24h
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import _atomic_write  # noqa: E402
from src.data.splits import arm_keys, dose_half  # noqa: E402
from src.eval.contrast_stats import did, paired  # noqa: E402
from src.eval.step_c_report import SCORING, arm_label, targeted_label  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args, line_groups)

CONFOUNDER = "cell_id"
# What each judged arm must beat (S3). ADIGen's arms and P1's are judged by the
# same rule; P1 is a baseline (understand.md §3.3.1), which the verdict records.
BASELINE_OF = {"dr": "conditional", "dr_p2": "conditional",
               "dr_design": "conditional", "dr_design_p2": "conditional",
               "p1_cond_counts": "conditional", "p1_cond_design": "conditional",
               "p1_naive_counts": "naive", "p1_naive_design": "naive"}
ADIGEN_ARMS = ("dr", "dr_p2")
P1_CONTROLS = {"p1_cond_ones": "conditional", "p1_naive_ones": "naive"}
ORDER = ("naive", "conditional", "dr", "dr_p2", "dr_design", "dr_design_p2",
         "p1_cond_counts", "p1_cond_design", "p1_cond_ones",
         "p1_naive_counts", "p1_naive_design", "p1_naive_ones")


def _sha1_w(path: str) -> str:
    w = np.load(path)["w"].astype(np.float32)
    return hashlib.sha1(np.ascontiguousarray(w).tobytes()).hexdigest()


def _mark(v) -> str:
    return {True: "PASS", False: "FAIL", None: "UNDECIDED"}.get(v, str(v))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pool", default="all", choices=("all", "holdout"))
    p.add_argument("--glob", default=None, help="Override the arm-JSON glob.")
    p.add_argument("--out", default=None, help="Relative to <runs>/eval_artifacts/, or a path. "
                                               "Default step_a_verdict[_pool<pool>].json")
    for k, v in SCORING.items():
        p.add_argument(f"--{k}", type=type(v), default=v,
                       help=f"Required scoring setting (default {v}, §3.14); JSONs scored "
                            f"otherwise are skipped.")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    lg = line_groups(cfg.population)
    if lg is None:
        raise SystemExit(f"[report] population {cfg.population.name!r} declares no line groups; "
                         f"step A runs on core5_24h")
    lg_json = {k: list(v) for k, v in lg.items()}
    runs = cfg.paths.train_output_dir
    pat = args.glob or os.path.join(runs, "*", "eval_artifacts", "gen_*.json")
    if args.out is None:
        args.out = "step_a_verdict" + ("" if args.pool == "all" else f"_pool{args.pool}") + ".json"

    cells: dict[tuple, dict] = {}
    claims: dict[tuple, list[str]] = {}
    frames: dict[tuple, list[str]] = {}
    skipped: list[str] = []
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            d = json.load(fh)
        nm = os.path.basename(f)
        g = d.get("generator") or {}
        if d.get("source") != "generated":
            skipped.append(f"{nm}: source {d.get('source')!r}"); continue
        if d.get("population") != cfg.population.name:
            skipped.append(f"{nm}: population {d.get('population')!r}"); continue
        if float(d.get("syn_effect") or 0.0) != 0.0:
            skipped.append(f"{nm}: carries a synthetic injection (syn_effect "
                           f"{d.get('syn_effect')}); step A is scored without one"); continue
        off = [f"{k}={g.get(k)!r}" for k in SCORING if g.get(k) != getattr(args, k)]
        if off:
            skipped.append(f"{nm}: not scored at the step-A settings ({', '.join(off)})"); continue
        if args.pool not in d.get("pools", {}):
            skipped.append(f"{nm}: no '{args.pool}' pool"); continue
        blk = d["pools"][args.pool]
        acc = blk.get("accuracy") or {}
        bc = acc.get("bias_along_contrast")
        if not bc or "scored_high_minus_low_mean" not in bc:
            skipped.append(f"{nm}: no bias_along_contrast (scored against an oracle "
                           f"without line contrasts?)"); continue
        ap = os.path.join(g.get("run_dir", ""), "arch.json")
        if not os.path.isfile(ap):
            skipped.append(f"{nm}: no arch.json at {ap}"); continue
        with open(ap) as fh:
            arch = json.load(fh)
        tier = arch.get("tier") or {}
        if not tier.get("active") or tier.get("confounder") != CONFOUNDER:
            skipped.append(f"{nm}: not thinned on {CONFOUNDER} (tier.active="
                           f"{tier.get('active')}, confounder={tier.get('confounder')!r})"); continue
        if tier.get("line_groups") != lg_json:
            skipped.append(f"{nm}: thinned with line groups {tier.get('line_groups')}, not "
                           f"spec.LINE_GROUPS {lg_json}"); continue
        lbl = targeted_label(d, arm_label(arch, CONFOUNDER))
        if lbl is None:
            skipped.append(f"{nm}: not a step-A arm (C={arch.get('adjustment_set')}, "
                           f"dr_mode={arch.get('dr_mode')}, dr_weight_norm="
                           f"{arch.get('dr_weight_norm')!r}, dr_weight_clip="
                           f"{arch.get('dr_weight_clip')!r}"
                           + (f", targeting={d.get('targeting', {}).get('arm')!r}"
                              if d.get("targeting") else "") + ")"); continue
        # weight provenance (S5)
        tgt = d.get("targeting") or {}
        wcheck = None
        try:
            if tgt and not (tgt.get("weights") or {}).get("is_control"):
                wp = (tgt.get("weights") or {}).get("path")
                wcheck = {"file": wp, "ok": _sha1_w(wp) == (tgt.get("weights") or {}).get("sha1")}
            elif not tgt and arch.get("dr_mode") == "weighted":
                wp = os.path.join(arch.get("nuisance_dir") or "", str(arch.get("dr_weights_file")))
                wcheck = {"file": wp, "ok": _sha1_w(wp) == arch.get("dr_weights_sha1")}
        except (OSError, TypeError, ValueError, KeyError) as e:
            wcheck = {"file": None, "ok": False, "error": str(e)}
        npz = (d.get("artifacts") or {}).get("tau_npz") or (f[:-5] + "_tau.npz")
        if not os.path.isfile(npz):
            skipped.append(f"{nm}: its _tau.npz is missing ({npz})"); continue
        key = (lbl, float(tier["gamma"]), int(arch["seed"]))
        claims.setdefault(key, []).append(f)
        frame = (d.get("table_fingerprint"), d.get("split_fingerprint"),
                 os.path.realpath((d.get("truth") or {}).get("path") or ""),
                 tuple(sorted(int(v) for v in (tier.get("scored_compounds") or {}).values())))
        frames.setdefault(frame, []).append(f)
        ll = blk.get("learned_line_contrast") or {}
        cells[key] = {"file": f, "npz": npz, "run": g.get("run"), "weights": wcheck,
                      "nuisance_dir": arch.get("nuisance_dir"),
                      "contrast_scored": bc.get("scored_high_minus_low_mean"),
                      "pooled_mse": (acc.get("pooled_gene") or {}).get("mse"),
                      "cos_responder": (acc.get("cos_responder") or {}).get("median"),
                      "learned_scored": (ll.get("scored") or {}).get("ratio"),
                      "learned_unscored": (ll.get("unscored") or {}).get("ratio")}

    if not cells:
        print(f"no step-A arm JSONs matched {pat}")
        for s in skipped[:12]:
            print("  skip ", s)
        sys.exit(1)
    dups = {k: fs for k, fs in claims.items() if len(fs) > 1}
    if dups:
        lines = [f"  {k}:\n" + "\n".join(f"      {x}" for x in fs) for k, fs in dups.items()]
        raise SystemExit("[report] more than one scored JSON per (arm, gamma, seed); refusing "
                         "to pick one. Narrow --glob or move the extras:\n" + "\n".join(lines))
    if len(frames) > 1:
        lines = [f"  table {a} split {b} truth {os.path.basename(c)} ({len(sc)} scored compounds): "
                 f"{len(fs)} JSONs, e.g. {os.path.basename(fs[0])}"
                 for (a, b, c, sc), fs in frames.items()]
        raise SystemExit("[report] the matched arms do not share one table / split / oracle / "
                         "scored set; a contrast across them is meaningless:\n" + "\n".join(lines))
    (_, _, truth_path, scored_tuple), = frames
    scored_c = np.array(scored_tuple, dtype=np.int64)

    # ---- per-arm projections, on one arm table ----
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control"]).to_pandas())
    comp_t = meta["compound_idx"].values.astype(np.int64)
    dl_t = meta["dose_level"].values.astype(np.float64)
    ic_t = meta["is_control"].values.astype(np.int64)
    half_of = dict(zip(arm_keys(comp_t, dl_t, ic_t).astype(str), dose_half(comp_t, dl_t, ic_t)))
    ak = None
    for key, c in cells.items():
        z = np.load(c["npz"], allow_pickle=False)
        k_ = z[f"{args.pool}/arm_key"].astype(str)
        if f"{args.pool}/contrast_proj" not in z:
            raise SystemExit(f"[report] {c['npz']} has no contrast_proj; re-score the run")
        if ak is None:
            ak = k_
        elif not np.array_equal(ak, k_):
            raise SystemExit(f"[report] {c['npz']}: its arm table differs from the others'")
        c["proj"] = z[f"{args.pool}/contrast_proj"].astype(np.float64)
    comp = np.array([int(k.split("|", 1)[0]) for k in ak], dtype=np.int64)
    half = np.array([int(half_of[k]) for k in ak])
    sc = np.isin(comp, scored_c)
    un_c = np.unique(comp[~sc])

    gammas = sorted({k[1] for k in cells})
    if 0.0 not in gammas or len(gammas) != 2:
        raise SystemExit(f"[report] need gamma = 0 and one gamma > 0, found {gammas}")
    g1 = [g for g in gammas if g > 0][0]
    arms = [a for a in ORDER if any(k[0] == a for k in cells)]
    print(f"[report] {cfg.population.name} pool={args.pool}: {len(cells)} arm-runs, arms {arms}, "
          f"gammas {gammas}; {scored_c.size:,} scored compounds, {int(sc.sum()):,} scored arms")
    if skipped:
        print(f"[report] skipped {len(skipped)}:")
        for s in skipped[:8]:
            print("    ", s)

    per_arm: dict[str, dict] = {}
    D: dict[str, dict] = {}
    for a in arms:
        seeds = sorted(s for (aa, g, s) in cells if aa == a and g == g1 and (a, 0.0, s) in cells)
        if not seeds:
            per_arm[a] = {"n_seeds": 0, "note": "no seed has both gamma cells"}
            continue
        b1 = np.mean([cells[(a, g1, s)]["proj"] for s in seeds], axis=0)
        b0 = np.mean([cells[(a, 0.0, s)]["proj"] for s in seeds], axis=0)
        d_sc = did(b1, b0, comp, half, sc, scored_c)
        d_un = did(b1, b0, comp, half, ~sc, un_c)
        by_seed = [did(cells[(a, g1, s)]["proj"], cells[(a, 0.0, s)]["proj"],
                       comp, half, sc, scored_c)["value"] for s in seeds]
        D[a] = d_sc
        mse = [cells[(a, g, s)]["pooled_mse"] for g in gammas for s in seeds]
        per_arm[a] = {
            "n_seeds": len(seeds), "seeds": seeds,
            "did": d_sc["value"], "se": d_sc["se"],
            "abs_over_se": (abs(d_sc["value"]) / d_sc["se"] if d_sc["se"] else None),
            "contrast_g1": d_sc["contrast_g1"], "contrast_g0": d_sc["contrast_g0"],
            "did_by_seed": by_seed,
            "seed_sd": (float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else None),
            "did_unscored": d_un["value"], "se_unscored": d_un["se"],
            "n_scored_arms": d_sc["n_arms"], "n_scored_compounds": d_sc["n_clusters"],
            "pooled_mse": (float(np.mean(mse)) if all(m is not None for m in mse) else None),
            "learned_line_contrast_scored": (
                lambda v: float(np.mean(v)) if v and all(x is not None for x in v) else None)(
                [cells[(a, g1, s)]["learned_scored"] for s in seeds]),
            "learned_line_contrast_unscored": (
                lambda v: float(np.mean(v)) if v and all(x is not None for x in v) else None)(
                [cells[(a, g, s)]["learned_unscored"] for g in gammas for s in seeds]),
        }
    mse_c = (per_arm.get("conditional") or {}).get("pooled_mse")
    for a, r in per_arm.items():
        if r.get("pooled_mse") is not None and mse_c:
            r["mse_ratio_vs_conditional"] = r["pooled_mse"] / mse_c

    def _f(v, se):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{v:+.3f}" + (f" +/- {se:.3f}" if se is not None else "")

    # The last three columns are the generator-health diagnostics track B found
    # decisive: a weighted risk can damage the generator on compounds it never
    # thinned (P2 results), which shows as a higher MSE and a lower learned
    # contrast on UNSCORED compounds, not in the bias column.
    print("\n| arm | seeds | DiD scored (+/- clustered SE) | seed sd | DiD unscored | "
          "MSE / conditional | learned line contrast, scored | unscored |")
    print("|---|---|---|---|---|---|---|---|")
    for a in arms:
        r = per_arm[a]
        if not r.get("n_seeds"):
            print(f"| `{a}` | 0 | n/a | | | | | |"); continue
        print(f"| `{a}` | {r['n_seeds']} | {_f(r['did'], r['se'])} "
              f"| {'n/a' if r['seed_sd'] is None else format(r['seed_sd'], '.3f')} "
              f"| {_f(r['did_unscored'], r['se_unscored'])} "
              f"| {'n/a' if r.get('mse_ratio_vs_conditional') is None else format(r['mse_ratio_vs_conditional'], '.3f')} "
              f"| {'n/a' if r['learned_line_contrast_scored'] is None else format(r['learned_line_contrast_scored'], '.3f')} "
              f"| {'n/a' if r['learned_line_contrast_unscored'] is None else format(r['learned_line_contrast_unscored'], '.3f')} |")

    # ---- criteria ----
    crit: dict[str, dict] = {}
    nav, cond = per_arm.get("naive") or {}, per_arm.get("conditional") or {}
    r1 = nav.get("abs_over_se")
    crit["S1_naive_biased"] = {"did": nav.get("did"), "se": nav.get("se"), "abs_over_se": r1,
                               "needs": "> 3", "pass": None if r1 is None else bool(r1 > 3.0)}
    r2 = cond.get("abs_over_se")
    testable = None if r2 is None else bool(r2 > 2.0)
    crit["S2_conditional_biased_testability"] = {
        "did": cond.get("did"), "se": cond.get("se"), "abs_over_se": r2, "needs": "> 2",
        "testable": testable,
        # A testability flag, not a pass/fail: when `conditional` shows no bias,
        # "DR vs conditional" cannot be tested here (as in step C). That is
        # recorded as "not_testable", never as a failed criterion.
        "pass": (None if testable is None else True if testable else "not_testable"),
        "note": "pass = testable. 'not_testable' means `conditional` is unbiased here; it is "
                "not a failure"}
    s3 = {}
    for a, base in BASELINE_OF.items():
        if a not in D or base not in D:
            continue
        pd_ = paired(D[a], D[base], lambda x, y: np.abs(y) - np.abs(x))
        gain_over_se = (pd_["value"] / pd_["se"]) if pd_["se"] else None
        beats = (None if gain_over_se is None else bool(pd_["value"] > 0 and gain_over_se > 2.0))
        status = beats
        if base == "conditional" and testable is False:
            status = "not_testable"
        s3[a] = {"baseline": base, "did": D[a]["value"], "baseline_did": D[base]["value"],
                 "abs_reduction": pd_["value"], "se_paired": pd_["se"],
                 "reduction_over_se": gain_over_se, "needs": "> 2", "pass": status,
                 "kind": ("ADIGen" if a in ADIGEN_ARMS else
                          "P1 baseline (AIPW on the estimand)" if a.startswith("p1_") else "reference")}
    crit["S3_beats_baseline"] = {
        "by_arm": s3,
        "adigen": {a: s3[a]["pass"] for a in ADIGEN_ARMS if a in s3},
        "pass": (None if not any(a in s3 for a in ADIGEN_ARMS) else
                 "not_testable" if testable is False else
                 bool(any(s3[a]["pass"] is True for a in ADIGEN_ARMS if a in s3))),
        "note": "pass = at least one ADIGen arm (dr, dr_p2) beats `conditional`; each arm's "
                "own result is in by_arm"}
    judged = [a for a in ("naive", "conditional") + ADIGEN_ARMS if per_arm.get(a, {}).get("n_seeds")]
    # An arm fails S4 only if its unscored DiD is BOTH above a quarter of naive's
    # scored bias AND more than 2 SE from zero. The second half is the noise
    # allowance the P2 pass rule lacked (IMPLEMENT.md §5, "P2 results", finding
    # 4): if `naive` shows little bias, a quarter of it is smaller than the
    # unscored DiD's own sampling error, and the bound alone would fail on noise.
    bound = None if nav.get("did") is None else 0.25 * abs(nav["did"])
    s4 = {}
    for a in judged:
        v, se = per_arm[a].get("did_unscored"), per_arm[a].get("se_unscored")
        if v is None or math.isnan(v) or bound is None:
            s4[a] = {"did_unscored": v, "se": se, "pass": None}
            continue
        small = bool(abs(v) < bound)
        null = bool(se is not None and abs(v) < 2.0 * se)
        s4[a] = {"did_unscored": v, "se": se, "below_bound": small, "within_2se_of_zero": null,
                 "pass": bool(small or null)}
    ok4 = [r["pass"] for r in s4.values()]
    worst = max((abs(r["did_unscored"]) for r in s4.values() if r["pass"] is not None), default=None)
    crit["S4_unscored_unchanged"] = {
        "max_abs_did_unscored": worst, "arms": judged, "by_arm": s4, "bound": bound,
        "rule": "per arm: |DiD_unscored| < 0.25 x naive's scored |DiD|, or within 2 SE of zero",
        "pass": (None if not ok4 or any(x is None for x in ok4) else bool(all(ok4)))}
    wbad = sorted(f"{a}|g{g:g}|s{s}" for (a, g, s), c in cells.items()
                  if c["weights"] is not None and not c["weights"]["ok"])
    n_w = sum(1 for c in cells.values() if c["weights"] is not None)
    corr = {}
    for nz in sorted({c["nuisance_dir"] for c in cells.values() if c["nuisance_dir"]}):
        try:
            wc = np.load(os.path.join(nz, "dr_weights_counts.npz"))["w"].astype(np.float64)
            wd = np.load(os.path.join(nz, "dr_weights_design.npz"))["w"].astype(np.float64)
            m = wd != 1.0
            corr[os.path.basename(nz)] = (float(np.corrcoef(wc[m], wd[m])[0, 1])
                                          if m.sum() > 2 and wc[m].std() > 0 and wd[m].std() > 0 else None)
        except OSError:
            corr[os.path.basename(nz)] = None
    crit["S5_weights"] = {
        "weighted_or_targeted_runs_checked": n_w, "weights_mismatch": wbad,
        "corr_counts_design_on_kept_scored_rows": corr,
        "pass": bool(n_w and not wbad),
        "note": "pass = every weight hash matches its file. That the counts weights equal "
                "n_unthinned / n_kept per (arm, line) cell is check_phase1's exact recompute; "
                "the correlation with the design weights is reported, not gated"}
    ctrl = {}
    for a, src in P1_CONTROLS.items():
        if a in D and src in D:
            ctrl[a] = {"did": D[a]["value"], "source": src, "source_did": D[src]["value"],
                       "worse_than_source": bool(abs(D[a]["value"]) > abs(D[src]["value"]))}

    print("\n### STEP_A.md §3 criteria\n")
    for k, c in crit.items():
        print(f"- **{_mark(c['pass'])}** {k}"
              + (f" (|DiD|/SE {c['abs_over_se']:.1f})" if c.get("abs_over_se") is not None else ""))
    for a, r in s3.items():
        print(f"    - {a} vs {r['baseline']}: |DiD| {abs(r['did']):.3f} vs {abs(r['baseline_did']):.3f}, "
              f"reduction {r['abs_reduction']:+.3f}"
              + (f" +/- {r['se_paired']:.3f}" if r["se_paired"] is not None else "")
              + f" -> {_mark(r['pass'])} [{r['kind']}]")
    for a, r in ctrl.items():
        print(f"    - control {a}: DiD {r['did']:+.3f} vs {r['source']} {r['source_did']:+.3f} "
              f"-> {'worse, as it must be' if r['worse_than_source'] else 'NOT worse'}")

    strip = lambda d_: {k: v for k, v in d_.items() if k not in ("proj",)}
    doc = {"population": cfg.population.name, "pool": args.pool, "line_groups": lg_json,
           "gammas": gammas, "n_arm_runs": len(cells), "arms": arms,
           "n_scored_compounds": int(scored_c.size), "n_scored_arms": int(sc.sum()),
           "truth": truth_path, "scoring": {k: getattr(args, k) for k in SCORING},
           "statistic": "DiD (gamma > 0 minus gamma = 0) of the mean high - low dose-half "
                        "contrast of <tau_est - tau_oracle, u_a> on scored arms, seeds averaged "
                        "per arm; errors are a delete-one-compound jackknife",
           "per_arm": per_arm, "criteria": crit, "p1_controls": ctrl,
           "cells": {f"{a}|g{g:g}|s{s}": strip(c) for (a, g, s), c in sorted(cells.items())},
           "skipped": skipped}
    out = args.out if os.path.sep in args.out else os.path.join(runs, "eval_artifacts", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _atomic_write(out, lambda f: json.dump(doc, f, indent=2))
    print(f"\n[report] -> {out}")


if __name__ == "__main__":
    main()
