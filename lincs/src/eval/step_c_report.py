"""Step-C verdict: the §3.8.1 success criteria from the scored arm matrix. No PyTorch.

Collects `gen_*_syn<effect>*.json` across the step-C arms, pivots them on
(arm, gamma, seed), and judges the four criteria. Writes `step_c_verdict.json`
and prints a markdown table for IMPLEMENT.md §5.

**The statistic is a difference-in-differences on the MEAN contrast.**

  contrast(arm, gamma, seed) = bias_along_v.scored_high_minus_low_mean
  Delta(arm, seed)           = contrast(gamma=1) - contrast(gamma=0)

Why each half of that matters:

  - `--pool all` only. On `--pool holdout` the P3 split leaves one well per arm,
    so <tau_oracle, v> = beta * (syn_c - 1/2) is bimodal at +/- beta/2 and the
    statistic is not interpretable (§5, corrected 2026-10-01).
  - The MEAN, not the median: the per-arm projection is discrete and bimodal by
    construction, which is the case a median handles badly.
  - Differenced against gamma = 0: the MCAR control at the same `keep_frac`
    carries the same model quality and the same dose-response gradient with no
    thinning, so differencing it out leaves the thinning bias. That is what the
    gamma = 0 control is for, and why all four arms are run at gamma = 0.

Seed noise is the spread of Delta across the training seeds of one arm, which is
what §3.8.1 means by judging "DR beats conditional" against seed noise.

**What is admitted.** A JSON counts only if it was scored at the §3.14 settings
(`SCORING`), at this `--syn_effect`, against the same injection (beta, v) as every
other admitted JSON. Two JSONs for one (arm, gamma, seed) cell are an error, not
a choice: the report refuses rather than keep whichever sorts first.

**Which injection.** `--syn_meta` names the resolved injection the arms must have
been scored against: `syn_meta.json` (step C, the default) or
`syn_meta_compound_r1.json` (step C2, §3.8.4). Its `v_sha1` selects the JSONs, so
step C and step C2 runs can share a runs dir and a glob.

    python -m src.eval.step_c_report
    python -m src.eval.step_c_report --syn_effect 1.0 --pool all --out step_c_verdict.json
    python -m src.eval.step_c_report --syn_meta syn_meta_compound_r1.json   # step C2
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import _atomic_write  # noqa: E402
from src.data.synthetic import SYN_META, load_syn_meta, table_syn_seed  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

ARMS = ("naive", "conditional", "dr", "dr_design")

# The scoring settings every step-C arm must share: §3.14 E1 (checkpoint-0499,
# EMA), E2 (w = 1), E3 (16 samples per real row) and E10 (100 steps). A JSON
# scored any other way is not a step-C result however its arch.json reads: the
# p5smoke arms are tiered, gamma 1, seed 0 and C = syn_c too, but were scored at
# epoch 0 with 1 sample and 4 steps, and one of them once filled the
# (dr, 1.0, 0) cell in place of the real run (§5, step-C results).
SCORING = {"gen_epoch": 499, "which_wgt": "ema", "guidance_scale": 1.0,
           "n_per_row": 16, "num_inference_steps": 100}


def arm_label(arch: dict) -> str | None:
    """The §3.8.1 arm this run is, from its own arch.json.

    Read from `arch.json` rather than the directory name: the name is derived
    from the flags, so trusting it would make the report agree with itself by
    construction rather than with what was trained.
    """
    c = tuple(arch.get("adjustment_set") or ())
    mode = str(arch.get("dr_mode"))
    if mode == "conditional":
        return "naive" if not c else ("conditional" if c == ("syn_c",) else None)
    if mode == "weighted" and c == ("syn_c",):
        f = str(arch.get("dr_weights_file") or "")
        if "counts" in f:
            return "dr"
        if "design" in f:
            return "dr_design"
    return None


def _mean_sd(v: list[float]) -> tuple[float | None, float | None, int]:
    a = np.asarray([x for x in v if x is not None and math.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return None, None, 0
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else None), int(a.size)


def _sep(a: list[float], b: list[float]) -> float | None:
    """(|mean b| - |mean a|) in units of the pooled seed noise: how far arm `a`'s
    bias sits below arm `b`'s, measured against seed-to-seed spread."""
    ma, sa, na = _mean_sd(a)
    mb, sb, nb = _mean_sd(b)
    if ma is None or mb is None or na < 2 or nb < 2 or sa is None or sb is None:
        return None
    se = math.sqrt(sa ** 2 / na + sb ** 2 / nb)
    return None if se <= 0 else float((abs(mb) - abs(ma)) / se)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--syn_effect", type=float, default=1.0)
    p.add_argument("--pool", default="all", choices=("all", "train", "holdout"),
                   help="Only 'all' is interpretable for step C; see the module docstring.")
    p.add_argument("--glob", default=None, help="Override the arm-JSON glob.")
    p.add_argument("--plan_contrast", type=float, default=-8.32,
                   help="`build_tiered_split --plan`'s predicted high-low contrast at gamma=1.")
    p.add_argument("--syn_meta", default=SYN_META,
                   help="The injection the arms were scored against (step C: syn_meta.json; "
                        "step C2: syn_meta_compound_r1.json). JSONs with another v_sha1 are "
                        "skipped.")
    p.add_argument("--out", default=None,
                   help="Relative to <runs>/eval_artifacts/, or a path. Default: "
                        "step_c_verdict.json (step C), step_c_verdict_<variant>.json otherwise.")
    for k, v in SCORING.items():
        p.add_argument(f"--{k}", type=type(v), default=v,
                       help=f"Required scoring setting (default {v}, §3.14); JSONs scored "
                            f"otherwise are skipped.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    runs = cfg.paths.train_output_dir
    # Validated like every other consumer's (table, syn_c draw, syn_effect, and in
    # compound mode the regenerated direction matrix against its hash), so a
    # stale file cannot become the reference the JSONs are filtered on.
    cfg.outcome = replace(cfg.outcome, syn_effect=float(args.syn_effect),
                          syn_seed=table_syn_seed(cfg))
    want_meta = load_syn_meta(cfg, name=args.syn_meta, n_genes=cfg.outcome.n_genes)
    want_sha1 = str(want_meta["v_sha1"])
    base = os.path.basename(args.syn_meta)
    variant = ("" if base == SYN_META else
               os.path.splitext(base)[0].replace("syn_meta_", "", 1))
    if args.out is None:
        args.out = f"step_c_verdict{'_' + variant if variant else ''}.json"
    print(f"[report] injection {args.syn_meta}: mode {want_meta.get('mode', 'global')}, "
          f"beta {float(want_meta['beta']):.4f}, sha1 {want_sha1[:12]}")
    tag = f"syn{args.syn_effect:g}"
    pat = args.glob or os.path.join(runs, "*", "eval_artifacts", f"gen_*_{tag}*.json")

    cells: dict[tuple, dict] = {}
    claims: dict[tuple, list[str]] = {}
    truths: dict[tuple, list[str]] = {}
    skipped: list[str] = []
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            d = json.load(fh)
        g = d.get("generator") or {}
        # The glob's `syn1*` also matches syn1.5 or syn10; the JSON's own value is exact.
        if abs(float(d.get("syn_effect") or 0.0) - args.syn_effect) > 1e-12:
            skipped.append(f"{os.path.basename(f)}: syn_effect {d.get('syn_effect')}, "
                           f"not {args.syn_effect}"); continue
        off = [f"{k}={g.get(k)!r}" for k in SCORING if g.get(k) != getattr(args, k)]
        if off:
            skipped.append(f"{os.path.basename(f)}: not scored at the step-C settings "
                           f"({', '.join(off)})"); continue
        if str(d.get("syn_v_sha1")) != want_sha1:
            skipped.append(f"{os.path.basename(f)}: scored against another injection "
                           f"(v_sha1 {str(d.get('syn_v_sha1'))[:12]}, not {args.syn_meta}'s "
                           f"{want_sha1[:12]})"); continue
        if args.pool not in d.get("pools", {}):
            skipped.append(f"{os.path.basename(f)}: no '{args.pool}' pool"); continue
        acc = (d["pools"][args.pool].get("accuracy") or {})
        ls = d["pools"][args.pool].get("learned_syn_effect") or {}
        bv = acc.get("bias_along_v")
        if not bv:
            skipped.append(f"{os.path.basename(f)}: no bias_along_v (was it scored with --syn_effect?)")
            continue
        ap = os.path.join(g.get("run_dir", ""), "arch.json")
        if not os.path.isfile(ap):
            skipped.append(f"{os.path.basename(f)}: no arch.json at {ap}"); continue
        with open(ap) as fh:
            arch = json.load(fh)
        lbl = arm_label(arch)
        tier = arch.get("tier") or {}
        if lbl is None or not tier.get("active"):
            skipped.append(f"{os.path.basename(f)}: not a step-C arm "
                           f"(C={arch.get('adjustment_set')}, dr_mode={arch.get('dr_mode')}, "
                           f"tier.active={tier.get('active')})")
            continue
        key = (lbl, float(tier["gamma"]), int(arch["seed"]))
        wcheck = None
        if arch.get("dr_mode") == "weighted":
            # The trainer hashes the raw float32 weights it loaded (before the
            # mean-1 normalisation); recompute from the file the run names.
            wp = os.path.join(arch.get("nuisance_dir") or "", str(arch.get("dr_weights_file")))
            try:
                w = np.load(wp)["w"].astype(np.float32)
                now = hashlib.sha1(np.ascontiguousarray(w).tobytes()).hexdigest()
                wcheck = {"file": wp, "ok": now == arch.get("dr_weights_sha1")}
            except OSError as e:
                wcheck = {"file": wp, "ok": False, "error": str(e)}
        # Every accepted JSON is recorded, so a duplicate is refused below rather
        # than resolved by sort order.
        claims.setdefault(key, []).append(f)
        truths.setdefault((d.get("syn_beta"), d.get("syn_v_sha1")), []).append(f)
        cells[key] = {"file": f, "run": g.get("run"),
                      "scored": bv.get("scored_high_minus_low_mean"),
                      "unscored": bv.get("unscored_high_minus_low_mean"),
                      "scored_se": bv.get("scored_high_minus_low_se"),
                      "n_scored_arms": bv.get("n_scored_arms"),
                      "median_is_meaningful": bv.get("median_is_meaningful", True),
                      "cos_responder": (acc.get("cos_responder") or {}).get("median"),
                      "pooled_mse": (acc.get("pooled_gene") or {}).get("mse"),
                      "weights": wcheck,
                      # §3.8.4's lambda-hat (absent from JSONs scored before it existed)
                      "lambda_scored": (ls.get("scored") or {}).get("mean"),
                      "lambda_unscored": (ls.get("unscored") or {}).get("mean")}

    if not cells:
        print(f"no step-C arm JSONs matched {pat}")
        for s in skipped[:10]:
            print("  skip ", s)
        sys.exit(1)
    dups = {k: fs for k, fs in claims.items() if len(fs) > 1}
    if dups:
        lines = [f"  {k}:\n" + "\n".join(f"      {x}" for x in fs) for k, fs in dups.items()]
        raise SystemExit("[report] more than one scored JSON per (arm, gamma, seed); refusing "
                         "to pick one. Narrow --glob or move the extras:\n" + "\n".join(lines))
    if len(truths) > 1:
        lines = [f"  beta={b} v_sha1={str(s)[:12]}: {len(fs)} JSONs, e.g. {fs[0]}"
                 for (b, s), fs in truths.items()]
        raise SystemExit("[report] the matched arms were scored against different injections; "
                         "a contrast across them is meaningless. Narrow --glob:\n"
                         + "\n".join(lines))
    (syn_beta, syn_v_sha1), = truths
    if any(not c["median_is_meaningful"] for c in cells.values()):
        print(f"[warn] pool {args.pool!r} has 1-well arms; only the MEAN contrast is used",
              file=sys.stderr)

    seeds = sorted({k[2] for k in cells})
    gammas = sorted({k[1] for k in cells})
    print(f"[report] pool={args.pool}  {len(cells)} arm-runs  "
          f"arms={sorted({k[0] for k in cells})}  gammas={gammas}  seeds={seeds}")
    if skipped:
        print(f"[report] skipped {len(skipped)}:")
        for s in skipped[:8]:
            print("    ", s)

    # ---- per-arm Delta over seeds ----
    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        d_scored, d_unscored, raw = [], [], {}
        for s in seeds:
            c1, c0 = cells.get((arm, 1.0, s)), cells.get((arm, 0.0, s))
            raw[s] = {"g1": c1 and c1["scored"], "g0": c0 and c0["scored"]}
            if c1 and c0 and c1["scored"] is not None and c0["scored"] is not None:
                d_scored.append(c1["scored"] - c0["scored"])
                if c1["unscored"] is not None and c0["unscored"] is not None:
                    d_unscored.append(c1["unscored"] - c0["unscored"])
        m, sd, n = _mean_sd(d_scored)
        mu_, su_, nu_ = _mean_sd(d_unscored)
        lam_g1, _, _ = _mean_sd([cells[(arm, 1.0, s)]["lambda_scored"]
                                 for s in seeds if (arm, 1.0, s) in cells])
        lam_un, _, _ = _mean_sd([c["lambda_unscored"] for k, c in cells.items() if k[0] == arm])
        per_arm[arm] = {"delta_scored_mean": m, "delta_scored_sd": sd, "n_seeds": n,
                        "delta_scored": d_scored,
                        "delta_unscored_mean": mu_, "delta_unscored_sd": su_,
                        "delta_unscored": d_unscored,
                        "raw_contrast_by_seed": raw,
                        "lambda_scored_g1": lam_g1, "lambda_unscored": lam_un,
                        "abs_over_seed_sd": (abs(m) / sd if m is not None and sd else None)}

    def _fmt(m, sd):
        if m is None:
            return "n/a"
        return f"{m:+.2f}" + (f" +/- {sd:.2f}" if sd is not None else "")

    print("\n| arm | n seeds | Delta scored (mean +/- sd) | Delta unscored | |Delta|/sd "
          "| lambda-hat scored (g1) | lambda-hat unscored |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        a = per_arm[arm]
        ratio = a["abs_over_seed_sd"]
        print(f"| `{arm}` | {a['n_seeds']} "
              f"| {_fmt(a['delta_scored_mean'], a['delta_scored_sd'])} "
              f"| {_fmt(a['delta_unscored_mean'], a['delta_unscored_sd'])} "
              f"| {'n/a' if ratio is None else format(ratio, '.1f')} "
              f"| {'n/a' if a['lambda_scored_g1'] is None else format(a['lambda_scored_g1'], '.3f')} "
              f"| {'n/a' if a['lambda_unscored'] is None else format(a['lambda_unscored'], '.3f')} |")

    # ---- the four criteria ----
    nav = per_arm["naive"]
    crit = {}
    crit["1_naive_biased_at_g1_not_g0"] = {
        "delta_naive": nav["delta_scored_mean"], "seed_sd": nav["delta_scored_sd"],
        "abs_over_seed_sd": nav["abs_over_seed_sd"],
        "plan_prediction": args.plan_contrast,
        "g0_contrast_by_seed": {s: cells.get(("naive", 0.0, s), {}).get("scored")
                                for s in seeds},
        "pass": (nav["abs_over_seed_sd"] is not None and nav["abs_over_seed_sd"] > 3.0),
        "note": "Delta = contrast(gamma=1) - contrast(gamma=0) on scored arms; "
                "the gamma=0 contrast itself should sit near 0",
    }
    crit["2_dr_beats_conditional_and_approaches_design"] = {
        "dr_vs_conditional_sigma": _sep(per_arm["dr"]["delta_scored"],
                                        per_arm["conditional"]["delta_scored"]),
        "dr_vs_design_sigma": _sep(per_arm["dr"]["delta_scored"],
                                   per_arm["dr_design"]["delta_scored"]),
        "delta": {a: per_arm[a]["delta_scored_mean"] for a in ARMS},
        "pass": None,
        "note": "dr_vs_conditional_sigma > 2 means dr's |bias| is below conditional's by "
                "more than seed noise. A null result is an admissible outcome (§3.8.1 "
                "gate): selection depends only on (dose half, syn_c), both of which the "
                "conditional arm conditions on, so it is already unbiased under correct "
                "specification and DR's edge is a finite-sample effect.",
    }
    # §3.8.1 criterion 2 has two halves, and both are judged: dr below conditional
    # by more than seed noise, AND close to the true-weight arm. "Close" is the
    # rule §3.8.4 fixed before any C2 result: |dr - dr_design| < 0.25 x
    # |conditional - dr_design|, i.e. dr sits nearer the truth weights than a
    # quarter of the way to the unweighted arm.
    c2 = crit["2_dr_beats_conditional_and_approaches_design"]
    sep = c2["dr_vs_conditional_sigma"]
    dd = {a: per_arm[a]["delta_scored_mean"] for a in ("dr", "dr_design", "conditional")}
    close = (None if any(v is None for v in dd.values()) else
             bool(abs(dd["dr"] - dd["dr_design"])
                  < 0.25 * abs(dd["conditional"] - dd["dr_design"])))
    c2["beats_conditional"] = None if sep is None else bool(sep > 2.0)
    c2["close_to_design"] = close
    c2["close_rule"] = "|dr - dr_design| < 0.25 x |conditional - dr_design| (§3.8.4)"
    c2["pass"] = (None if sep is None or close is None
                  else bool(c2["beats_conditional"] and close))
    worst = max((abs(per_arm[a]["delta_unscored_mean"])
                 for a in ARMS if per_arm[a]["delta_unscored_mean"] is not None), default=None)
    crit["3_unscored_unchanged"] = {
        "max_abs_delta_unscored": worst,
        "by_arm": {a: per_arm[a]["delta_unscored_mean"] for a in ARMS},
        "pass": (worst is not None and nav["delta_scored_mean"] is not None
                 and worst < 0.25 * abs(nav["delta_scored_mean"])),
        "note": "unscored compounds were never thinned, so their Delta is the internal "
                "negative control; judged against naive's scored Delta",
    }
    # The correlations are the data layer's (check_phase1 on the two seed-42 tier
    # instances, which steps C and C2 share). What IS checked here: every weighted
    # run trained on exactly the weights in its tier instance's file today.
    wbad = sorted(f"{a}|g{g:g}|s{s}" for (a, g, s), c in cells.items()
                  if c.get("weights") is not None and not c["weights"]["ok"])
    n_w = sum(1 for c in cells.values() if c.get("weights") is not None)
    crit["4_weights_track_design"] = {
        "pass": bool(n_w and not wbad), "measured_before_training": True,
        "corr_counts_design": {"gamma=0": 0.73, "gamma=1": 0.88},
        "weighted_runs_checked": n_w, "weights_mismatch": wbad,
        "note": "correlations cited from the data layer (check_phase1 --adjustment_set "
                "syn_c on the shared tier instances); each weighted run's arch.json "
                "dr_weights_sha1 is recomputed from the file it names, and must match",
    }

    # §3.8.4 diagnostics, not gates: conditional's DiD should be the share of the
    # shift it did NOT learn, times naive's.
    lam_c = per_arm["conditional"]["lambda_scored_g1"]
    diag = {"lambda_scored_g1": {a: per_arm[a]["lambda_scored_g1"] for a in ARMS},
            "conditional_delta": per_arm["conditional"]["delta_scored_mean"],
            "conditional_delta_predicted": (
                None if lam_c is None or nav["delta_scored_mean"] is None
                else (1.0 - lam_c) * nav["delta_scored_mean"]),
            "note": "predicted = (1 - lambda-hat of conditional's scored arms at gamma=1) x "
                    "naive's Delta; §3.8.4 expects agreement within a factor of 2"}
    if diag["conditional_delta_predicted"] is not None:
        print(f"\n[diag] conditional Delta {diag['conditional_delta']:+.3f} vs "
              f"(1 - lambda-hat {lam_c:.3f}) x naive's = {diag['conditional_delta_predicted']:+.3f}")

    print("\n### §3.8.1 criteria\n")
    for k, c in crit.items():
        mark = {True: "PASS", False: "FAIL", None: "UNDECIDED"}[c["pass"]]
        print(f"- **{mark}** {k}")
    doc = {"pool": args.pool, "syn_effect": args.syn_effect, "population": cfg.population.name,
           "n_arm_runs": len(cells), "seeds": seeds, "gammas": gammas,
           "plan_contrast": args.plan_contrast, "per_arm": per_arm,
           "criteria": crit, "diagnostics": diag, "skipped": skipped,
           "syn_meta": args.syn_meta, "syn_mode": want_meta.get("mode", "global"),
           "syn_rho": want_meta.get("rho"),
           "scoring": {k: getattr(args, k) for k in SCORING},
           "syn_beta": syn_beta, "syn_v_sha1": syn_v_sha1,
           "cells": {f"{a}|g{g:g}|s{s}": {"run": c["run"], "file": c["file"]}
                     for (a, g, s), c in sorted(cells.items())},
           "statistic": "difference-in-differences of the MEAN scored high-low contrast "
                        "of <tau_gen - tau_oracle, v>, pool=all, vs the gamma=0 MCAR control"}
    out = args.out if os.path.sep in args.out else os.path.join(
        runs, "eval_artifacts", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _atomic_write(out, lambda f: json.dump(doc, f, indent=2))
    print(f"\n[report] -> {out}")


if __name__ == "__main__":
    main()
