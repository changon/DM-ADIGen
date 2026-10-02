"""Read-back checks of the Phase 4 eval artifacts (IMPLEMENT.md §3.10, §3.14). No PyTorch.

Re-derives what it can independently of `src.eval` (pandas groupby straight off
the table and expr.npy, never the evaluate code path):

  json/npz   the JSON and <out>_tau.npz agree on arms, ||tau||, and the
             responder flag; every sub-pool has a table
  tau        tau recomputed from expr.npy for a sample of arms
  arms       arm counts and well counts against population_qc.json's
             summary.arm_size_counts and the sub-pool's min_dose_n
  mu0        ||mu_hat(0)|| sits at its own DMSO sampling scale, i.e. z = 0 is
             the vehicle (§3.2 decision 5 / P10)
  floor      the empirical noise floor (E11) against the analytic
             sqrt(sum_g var_g (1/n + 1/n_ref)) for the same sizes
  ceiling    split-half reliability in [-1, 1] and rising with arm size
  responder  the share of arms clearing 1.5x the floor (~29% per §3.8.1)
  arm        a generated arm's JSON: identity fields equal its run's arch.json,
             the accuracy block is present, and cos <= the ceiling in aggregate

Sections run when their artifacts exist. Exits nonzero on any failure.

Run from lincs/ (a CPU job for the full population):
    python -m src.tests.check_phase4 --oracle runs/mcf7_24h/eval_artifacts/oracle_mcf7_24h_poolall.json
    python -m src.tests.check_phase4 --oracle <oracle.json> --arm <gen_....json> [--arm ...]
    python -m src.tests.check_phase4 --data_dir data/mcf7_24h_limit1500 --oracle ... --no_bands
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

from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import CONTROL_ARM, arm_keys, load_splits  # noqa: E402
from src.data.synthetic import inject_meta, load_syn_meta  # noqa: E402
from src.eval.evaluate import TRUTH_GUARD, _min_n_for  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []

# §3.8.1 quotes "only ~29% of MCF7 arms clear 1.5x the 3-well noise floor".
RESPONDER_BAND = (0.15, 0.45)
# ||mu_hat(0)|| should be at the scale of a mean of n_dmso independent wells.
MU0_BAND = (0.0, 3.0)
# The empirical floor is a median of norms, the analytic value a root-mean
# square, so they agree in scale rather than exactly.
FLOOR_RTOL = 0.25


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def _npz_for(doc: dict, path: str) -> str:
    p = (doc.get("artifacts") or {}).get("tau_npz")
    return p if p and os.path.isfile(p) else path[:-5] + "_tau.npz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True, help="A --source real eval JSON.")
    p.add_argument("--arm", action="append", default=[],
                   help="A --source generated eval JSON (repeatable).")
    p.add_argument("--n_arm_sample", type=int, default=40,
                   help="Arms whose tau is recomputed from expr.npy.")
    p.add_argument("--no_bands", action="store_true",
                   help="Skip the responder / mu0 / floor bands (a tiny --limit build "
                        "has too few wells for them to mean anything).")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)

    with open(args.oracle) as fh:
        doc = json.load(fh)
    z = np.load(_npz_for(doc, args.oracle), allow_pickle=False)
    print(f"[oracle] {args.oracle}\n[oracle] pools {list(doc['pools'])}")

    check(doc["source"] == "real", "oracle source is 'real'")
    check(doc["n_genes"] == cfg.outcome.n_genes,
          f"n_genes {doc['n_genes']} == cfg {cfg.outcome.n_genes}")

    # ---- the table, split and outcome, re-derived here ----
    splits = load_splits(cfg)
    m = load_expr_meta(cfg, doc["plate_center"], splits=splits)
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "det_plate", "syn_c"])
            .to_pandas())
    # An injected oracle (step C) carries y + syn_c * beta * v, so the independent
    # recompute below has to apply the SAME injection or it would disagree by beta.
    syn = None
    if float(doc.get("syn_effect") or 0) != 0:
        from dataclasses import replace as _replace
        cfg.outcome = _replace(cfg.outcome, syn_effect=float(doc["syn_effect"]),
                               syn_seed=int(doc["syn_seed"]))
        # The oracle names the injection file it used (step C2 has its own,
        # §3.8.4); oracles from before that record no name and are step C's.
        syn = load_syn_meta(cfg, name=(doc.get("syn") or {}).get("name"),
                            n_genes=cfg.outcome.n_genes)
        check(syn["v_sha1"] == (doc.get("syn") or {}).get("v_sha1"),
              f"syn_meta v_sha1 {syn['v_sha1'][:12]} == the oracle's recorded direction")
        check(abs(syn["beta"] - float((doc.get("syn") or {})["beta"])) < 1e-9,
              f"syn_meta beta {syn['beta']:.4f} == the oracle's recorded beta")
    keys_all = arm_keys(meta["compound_idx"].values, meta["dose_level"].values,
                        meta["is_control"].values).astype(str)
    expr = np.load(cfg.paths.expr_npy, mmap_mode="r")
    pc = plate_codes(meta["det_plate"].values, m["plates"])
    check(doc["split_fingerprint"] == splits["split_fingerprint"],
          "oracle split_fingerprint == splits.json")
    check(doc["table_fingerprint"] == m["table_fingerprint"],
          "oracle table_fingerprint == expr_meta.json")

    qc_path = cfg.paths.population_qc_json
    arm_size_counts = None
    if os.path.isfile(qc_path):
        with open(qc_path) as fh:
            arm_size_counts = (json.load(fh).get("summary") or {}).get("arm_size_counts")

    rows_of = {"all": np.arange(len(meta), dtype=np.int64),
               "train": np.asarray(splits["train_idx"], dtype=np.int64),
               "holdout": np.asarray(splits["holdout_idx"], dtype=np.int64)}

    for pname, block in doc["pools"].items():
        print(f"\n--- sub-pool {pname}")
        rows = np.sort(rows_of[pname])
        min_n = int(block["min_dose_n"])
        pre = f"{pname}/"
        for k in ("arm_key", "tau", "n_wells", "mu0", "tau_norm", "responder",
                  "floor_ratio", "reliability_cos", "tau_norm_denoised"):
            check(pre + k in z.files, f"{pname}: npz has {k}")
        keys, tau = z[pre + "arm_key"], z[pre + "tau"]
        n_wells, mu0 = z[pre + "n_wells"], z[pre + "mu0"]
        check(keys.size == block["n_arms"],
              f"{pname}: npz {keys.size:,} arms == json n_arms {block['n_arms']:,}")
        check(tau.shape == (keys.size, cfg.outcome.n_genes),
              f"{pname}: tau shape {tau.shape} == ({keys.size}, {cfg.outcome.n_genes})")
        check(np.allclose(np.linalg.norm(tau, axis=1), z[pre + "tau_norm"], rtol=1e-5),
              f"{pname}: npz tau_norm == ||tau||")
        dn = z[pre + "tau_norm_denoised"]
        check(bool((dn <= z[pre + "tau_norm"] + 1e-6).all()) and bool((dn >= 0).all()),
              f"{pname}: the noise-corrected ||tau|| is in [0, ||tau||]")
        check(bool((n_wells >= min_n).all()),
              f"{pname}: every kept arm has >= {min_n} wells")
        check(CONTROL_ARM not in set(keys.tolist()),
              f"{pname}: the vehicle arm is not in the treated table")

        # Independent recompute: group the real rows with pandas, not evaluate.
        df = pd.DataFrame({"k": keys_all[rows], "r": rows})
        grp = df.groupby("k")["r"]
        sizes = grp.size()
        ctl_rows = df.loc[df["k"] == CONTROL_ARM, "r"].values
        want_arms = sizes[(sizes.index != CONTROL_ARM) & (sizes >= min_n)]
        check(len(want_arms) == keys.size,
              f"{pname}: pandas finds {len(want_arms):,} arms with >= {min_n} wells "
              f"== {keys.size:,}")
        got_n = pd.Series(n_wells, index=keys)
        check(bool((got_n.reindex(want_arms.index).values == want_arms.values).all()),
              f"{pname}: per-arm well counts match pandas")

        def _y(idx):
            idx = np.sort(np.asarray(idx, dtype=np.int64))
            z = normalize_expr(np.asarray(expr[idx]), pc[idx], m)
            if syn is not None:
                z = inject_meta(z, meta["syn_c"].values.astype(np.int64)[idx],
                                meta["compound_idx"].values.astype(np.int64)[idx], syn)
            return z

        mu0_re = _y(ctl_rows).mean(0)
        check(np.allclose(mu0_re, mu0, atol=2e-5),
              f"{pname}: mu_hat(0) recomputed from expr.npy (max diff "
              f"{float(np.abs(mu0_re - mu0).max()):.2e})")
        rng = np.random.default_rng(0)
        pick = rng.choice(keys.size, min(int(args.n_arm_sample), keys.size), replace=False)
        worst, pos = 0.0, {k: i for i, k in enumerate(keys)}
        for i in pick:
            k = keys[i]
            tr = _y(grp.get_group(k).values).mean(0) - mu0_re
            worst = max(worst, float(np.abs(tr - tau[pos[k]]).max()))
        check(worst < 2e-4,
              f"{pname}: tau recomputed for {len(pick)} arms (max diff {worst:.2e})")

        if arm_size_counts and pname == "all" and len(rows) == len(meta):
            below = sum(v for kk, v in arm_size_counts.items() if int(kk) < min_n)
            check(block["n_arms_below_min_dose_n"] == below,
                  f"all: {below:,} arms below min_dose_n {min_n} per "
                  f"population_qc.json arm_size_counts")
            check(block["n_arms"] + below == sum(arm_size_counts.values()),
                  f"all: kept + dropped == {sum(arm_size_counts.values()):,} arms in QC")

        # mu_hat(0) against its own sampling scale, after removing the step-C
        # injection's known shift of the vehicle mean (evaluate subtracts it).
        r = block.get("mu0_norm_over_expected")
        if args.no_bands or r is None:
            print(f"      (mu0 band skipped; ratio={r})")
        else:
            check(MU0_BAND[0] <= r <= MU0_BAND[1],
                  f"{pname}: ||mu_hat(0)|| is {r:.2f}x its DMSO sampling scale, "
                  f"in {MU0_BAND}")

        # The empirical floor against the analytic value for the same sizes.
        floor = block["reference"]["floor"]
        y_dmso = _y(ctl_rows).astype(np.float64)
        var_g = y_dmso.var(0, ddof=1)
        bad = []
        for sn, rec in floor.items():
            n, n_ref = int(sn), int(rec["n_ref"])
            analytic = float(np.sqrt((var_g * (1.0 / n + 1.0 / n_ref)).sum()))
            if not abs(rec["median"] / analytic - 1.0) <= FLOOR_RTOL:
                bad.append(f"n={n}: empirical {rec['median']:.2f} vs analytic {analytic:.2f}")
        if args.no_bands:
            print(f"      (floor band skipped; {len(bad)}/{len(floor)} sizes outside)")
        else:
            check(not bad, f"{pname}: empirical floor ~ analytic for all "
                           f"{len(floor)} sizes" + (f" -- off: {bad[:3]}" if bad else ""))

        rel = z[pre + "reliability_cos"]
        fin = rel[np.isfinite(rel)]
        check(fin.size == 0 or bool((fin >= -1.0001).all() and (fin <= 1.0001).all()),
              f"{pname}: split-half reliability in [-1, 1] ({fin.size:,} arms)")
        by_n = block["reference"]["reliability"]["by_n"]
        meds = [(int(k), v["median"]) for k, v in by_n.items() if v["median"] is not None]
        if len(meds) >= 3:
            meds.sort()
            ns = np.array([a for a, _ in meds], float)
            vs = np.array([b for _, b in meds], float)
            rho = np.corrcoef(ns, vs)[0, 1]
            check(rho > 0, f"{pname}: reliability rises with arm size (corr {rho:+.2f})")
        else:
            print(f"      (reliability-vs-n trend needs 3+ sizes, have {len(meds)})")

        frac = block["responder_arm_frac"]
        injected = float(doc.get("syn_effect") or 0) != 0
        if injected:
            # The step-C injection raises every arm's noise floor (it adds a
            # beta * (syn_c imbalance) term), so the responder share drops --
            # measured 17.2% -> 14.4% at beta = 25.5. §3.8.1's ~29% describes the
            # REAL data, so the band does not apply to an injected oracle.
            print(f"      (responder band skipped: syn_effect="
                  f"{doc['syn_effect']} raises the floor; frac={frac:.3f})")
        elif args.no_bands or frac is None:
            print(f"      (responder band skipped; frac={frac})")
        else:
            check(RESPONDER_BAND[0] <= frac <= RESPONDER_BAND[1],
                  f"{pname}: {100 * frac:.1f}% of arms clear "
                  f"{doc['responder_mult']}x the floor, in "
                  f"{tuple(100 * b for b in RESPONDER_BAND)}% (§3.8.1 says ~29%)")

    # ---- generated arms ----
    for arm_path in args.arm:
        print(f"\n--- arm {arm_path}")
        with open(arm_path) as fh:
            a = json.load(fh)
        check(a["source"] == "generated", f"{os.path.basename(arm_path)}: source is 'generated'")
        g = a["generator"]
        arch_path = os.path.join(g["run_dir"], "arch.json")
        with open(arch_path) as fh:
            arch = json.load(fh)
        for k in ("population", "table_fingerprint", "split_fingerprint",
                  "gene_order_sha1", "plate_center", "normalize_mean", "normalize_std",
                  "syn_effect", "syn_seed"):
            if k not in arch:
                continue
            same = (abs(float(arch[k]) - float(a[k])) < 1e-12
                    if isinstance(a[k], float) else str(arch[k]) == str(a[k]))
            check(same, f"{g['run']}: {k} matches arch.json ({arch[k]!r})")
        check(list(arch.get("adjustment_set") or []) == list(a["adjustment_set"]),
              f"{g['run']}: adjustment_set matches arch.json")
        check(g["arch"] == arch["arch"] and g["diffusion_method"] == arch["diffusion_method"],
              f"{g['run']}: arch / diffusion_method match arch.json")
        check(a["truth"] is not None, f"{g['run']}: scored against an oracle (--truth)")
        for pname, block in a["pools"].items():
            acc = block.get("accuracy")
            check(acc is not None, f"{g['run']} [{pname}]: accuracy block present")
            if not acc:
                continue
            check(acc["n_matched"] > 0,
                  f"{g['run']} [{pname}]: {acc['n_matched']:,} arms matched the oracle")
            # Compare the two over the SAME arms: the ceiling only exists where
            # the oracle has >= 2 wells.
            cm = acc.get("cos_responder_with_ceiling", {}).get("median")
            ceil = acc["reliability_ceiling_responder"]["median"]
            if cm is not None and ceil is not None:
                # A generator cannot correlate with a noisy oracle better than
                # the oracle's own reliability allows. `ceil` is already the
                # Spearman-Brown-corrected bound (evaluate.corrected_ceiling), so
                # a large violation means tau leaked -- not that the model is good.
                check(cm <= ceil + 0.10,
                      f"{g['run']} [{pname}]: responder cos {cm:.3f} is within the "
                      f"attainable ceiling {ceil:.3f} (+0.10 slack); "
                      f"{cm / ceil:.3f} of it")
            gz = (a.get("artifacts") or {}).get("gen_npz")
            if gz and os.path.isfile(gz):
                zz = np.load(gz, allow_pickle=False)
                check(zz["row_mean"].shape[1] == cfg.outcome.n_genes,
                      f"{g['run']}: gen_npz row_mean has {cfg.outcome.n_genes} genes")
                check(int(zz["n_per_row"]) == g["n_per_row"],
                      f"{g['run']}: gen_npz n_per_row == {g['n_per_row']}")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURE(S):\n  - " + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
