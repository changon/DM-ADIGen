"""Read-back checks of the Phase 5 step-C artifacts (IMPLEMENT.md §3.8.2). No PyTorch.

  syn_meta   v is a unit vector, matches its sha1, and is reproducible from
             vec_seed; beta = syn_effect x scale; syn_seed = the table's draw.
             Compound mode (step C2, §3.8.4): the direction matrix is
             reproducible, its rows are unit, its geometry matches rho, and the
             rho = 0 matrix injects BIT-IDENTICALLY to the global mode
  inject     recomputed independently: exactly beta*v (beta*v_k on a compound-k
             row) on syn_c=1 rows, identity on syn_c=0, and no other row touched
  oracle     the injection does NOT move tau on the unablated pool, because syn_c
             is balanced within every arm and within every plate's DMSO -- this is
             what makes the step-C bias attributable to the thinning alone
  responders every scored compound is a responder, has >= 2 dose levels, and has a
             train well in all four {low,high} x syn_c positivity cells
  tier       the realised kept fraction matches `planned_bias`, and the planner's
             dp is 0 at gamma = 0 and equal-and-opposite across dose halves at
             gamma > 0 (the dose_half x syn_c lever's signature)

Sections run when their artifacts exist. Exits nonzero on any failure.

Run from lincs/ (a CPU job for the full population):
    python -m src.tests.check_phase5
    python -m src.tests.check_phase5 --data_dir data/mcf7_24h_limit1500
    python -m src.tests.check_phase5 --tier data/mcf7_24h/nuisances_tier_Csyn_c_k0_g1_s42
    python -m src.tests.check_phase5 --syn_meta syn_meta_compound_r1.json      # step C2
"""
from __future__ import annotations

import argparse
import glob
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

from src.data.build_tiered_split import planned_bias  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import CONTROL_ARM, arm_keys, dose_half, load_splits  # noqa: E402
from src.data.synthetic import (  # noqa: E402
    SYN_META, _v_sha1, directions_for, effect_directions, effect_vector, inject,
    inject_meta, load_syn_meta, resolve_syn_meta_path, table_syn_seed)
from src.nuisances.precompute_cmean import group_means  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", action="append", default=[],
                   help="A tiered split dir (repeatable). Default: every "
                        "<nuisance_dir>_tier_* that exists.")
    p.add_argument("--syn_meta", default=SYN_META)
    p.add_argument("--responders", default="responders.json")
    p.add_argument("--tau_tol", type=float, default=0.25,
                   help="Max median |<tau_syn - tau_plain, v>| on the unablated pool, as a "
                        "multiple of beta. The residue is the odd-well imbalance: a 3-well "
                        "arm splits syn_c 2/1, so its mean is 2/3 not 1/2, giving beta/6.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    nz = cfg.paths.nuisance_dir
    n_genes = int(cfg.outcome.n_genes)

    # ---- syn_meta -------------------------------------------------------
    # The injection may live in the base build's dir and be shared by every
    # tiered instance (src/data/synthetic.resolve_syn_meta_path).
    sp = resolve_syn_meta_path(cfg, args.syn_meta)
    if not os.path.isfile(sp):
        print(f"skip  no {sp}; run `python -m src.data.synthetic --syn_effect ...`")
        print("\nNOTHING CHECKED")
        sys.exit(1)
    with open(sp) as fh:
        raw = json.load(fh)
    # load_syn_meta validates against cfg, so give it the file's own syn_effect.
    from dataclasses import replace
    cfg.outcome = replace(cfg.outcome, syn_effect=float(raw["syn_effect"]),
                          syn_seed=int(raw["syn_seed"]))
    m = load_syn_meta(cfg, name=args.syn_meta, n_genes=n_genes)
    v = m["v"]
    check(abs(float(np.linalg.norm(v)) - 1.0) < 1e-9,
          f"v is a unit vector (norm {float(np.linalg.norm(v)):.12f})")
    check(np.allclose(v, effect_vector(m["vec_seed"], n_genes), rtol=0, atol=0),
          f"v is reproducible from vec_seed {m['vec_seed']} (bit-identical)")
    check(abs(m["beta"] - m["syn_effect"] * m["scale"]) < 1e-9,
          f"beta {m['beta']:.4f} == syn_effect {m['syn_effect']} x scale {m['scale']:.4f}")
    check(int(m["syn_seed"]) == table_syn_seed(cfg),
          f"syn_seed {m['syn_seed']} == the table's syn_c draw")
    compound = m["mode"] == "compound"
    if compound:
        V = m["V"]
        rho = float(m["rho"])
        with open(os.path.join(nz, "nuisance_meta.json")) as fh:
            n_comp = int(json.load(fh)["n_compounds"])
        check(V.shape == (n_comp, n_genes),
              f"compound mode: one direction per compound_idx ({V.shape[0]} == the "
              f"vocab's {n_comp}, vehicle included)")
        # Independent of synthetic.py: rebuild sampled rows from the raw streams
        # (v: [vec_seed, 23]; u_k: [vec_seed, 29, k]) and the §3.8.4 formula.
        # load_syn_meta already held V to its recorded hash, so this is the check
        # that the hash describes the intended construction.
        v_ind = np.random.default_rng([int(m["vec_seed"]), 23]).normal(size=n_genes)
        v_ind /= np.linalg.norm(v_ind)
        rows = sorted({0, 1, n_comp // 2, n_comp - 1})
        worst = 0.0
        for k in rows:
            u = np.random.default_rng([int(m["vec_seed"]), 29, k]).normal(size=n_genes)
            d = np.sqrt(1 - rho) * v_ind + np.sqrt(rho) * u / np.linalg.norm(u)
            worst = max(worst, float(np.abs(d / np.linalg.norm(d) - V[k]).max()))
        check(worst < 1e-12 and np.allclose(v, v_ind, rtol=0, atol=1e-15),
              f"directions rebuilt independently for compound_idx {rows} match "
              f"(max |diff| {worst:.1e}); sha1 {m['v_sha1'][:12]} is v_sha1 of this matrix")
        check(m["v_sha1"] == _v_sha1(V) and m["v_shared_sha1"] == _v_sha1(v),
              "v_sha1 hashes the matrix and v_shared_sha1 the shared v")
        nr = np.linalg.norm(V, axis=1)
        check(float(np.abs(nr - 1).max()) < 1e-12,
              f"every direction is a unit vector (max |norm - 1| {float(np.abs(nr - 1).max()):.1e})")
        cv = V @ v
        check(abs(float(np.median(cv)) - np.sqrt(1 - rho)) < 0.05,
              f"median cos(v_k, v) {float(np.median(cv)):.4f} ~ sqrt(1 - rho) "
              f"{np.sqrt(1 - rho):.4f}")
        if rho == 1.0:
            g = V[1:] @ V[1:].T
            g = np.abs(g[np.triu_indices(g.shape[0], 1)])
            check(float(np.median(g)) < 3.0 / np.sqrt(n_genes),
                  f"at rho = 1 compounds' directions are near-orthogonal: median "
                  f"|cos| {float(np.median(g)):.4f}, max {float(g.max()):.4f} "
                  f"(isotropic in {n_genes}-d: ~{0.6745 / np.sqrt(n_genes):.4f})")
    _pop = load_splits(cfg).get("population", {})
    if "table_fingerprint" in _pop:
        check(m["table_fingerprint"] == _pop["table_fingerprint"],
              f"syn_meta table_fingerprint {m['table_fingerprint']} == the split's")
    else:
        print("      (the split records no table_fingerprint; not checked)")

    # ---- the table and the outcome --------------------------------------
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "det_plate", "syn_c"])
            .to_pandas())
    splits = load_splits(cfg)
    em = load_expr_meta(cfg, None, splits=splits)
    syn_c = meta["syn_c"].values.astype(np.int64)
    expr = np.load(cfg.paths.expr_npy, mmap_mode="r")
    pc = plate_codes(meta["det_plate"].values, em["plates"])
    y_plain = normalize_expr(np.asarray(expr), pc, em)
    comp_all = meta["compound_idx"].values.astype(np.int64)
    y_syn = inject_meta(y_plain, syn_c, comp_all, m)

    # rho = 0 must be step C exactly: the compound path with every row's direction
    # equal to v has to reproduce the global path bit for bit (§3.8.4).
    n_dirs = int(comp_all.max()) + 1
    y_g = inject(y_plain, syn_c, m["beta"], effect_vector(m["vec_seed"], n_genes))
    y_c0 = inject(y_plain, syn_c, m["beta"],
                  effect_directions(m["vec_seed"], n_genes, n_dirs, 0.0), comp_all)
    check(np.array_equal(y_g, y_c0),
          "compound mode at rho = 0 injects bit-identically to the global mode "
          "(step C2 at rho = 0 IS step C)")
    del y_g, y_c0

    # ---- inject, recomputed independently -------------------------------
    d = (y_syn.astype(np.float64) - y_plain.astype(np.float64))
    one, zero = syn_c == 1, syn_c == 0
    check(bool(np.abs(d[zero]).max() < 1e-6),
          f"syn_c=0 rows are untouched (max |diff| {float(np.abs(d[zero]).max()):.2e})")
    want = m["beta"] * directions_for(m, comp_all[one])
    err = float(np.abs(d[one] - want).max())
    check(err < 2e-3, f"syn_c=1 rows are shifted by exactly beta*v"
                      f"{'_k (their own compound)' if compound else ''} "
                      f"(max |diff| {err:.2e}, on a shift of norm {m['beta']:.2f})")
    check(np.isin(np.unique(syn_c), (0, 1)).all() and one.sum() > 0 and zero.sum() > 0,
          f"syn_c is 0/1 and both levels are present ({int(one.sum()):,} / {int(zero.sum()):,})")

    # ---- what the injection does to tau on the unablated pool ------------
    keys = arm_keys(meta["compound_idx"].values, meta["dose_level"].values,
                    meta["is_control"].values).astype(str)
    def tau_of(y):
        uniq, mu, cnt = group_means(y, keys)
        uniq = uniq.astype(str)
        i0 = int(np.flatnonzero(uniq == CONTROL_ARM)[0])
        trt = np.ones(uniq.size, bool); trt[i0] = False
        return uniq[trt], mu[trt] - mu[i0], cnt[trt]
    k1, t_plain, _ = tau_of(y_plain)
    k2, t_syn, _ = tau_of(y_syn)
    check(np.array_equal(k1, k2), "the arm set is the same with and without the injection")
    if compound:
        # Step C2: a compound-k arm's syn_c = 1 wells move along v_k and the
        # vehicles' along v_0, so syn_c MODIFIES the treatment effect and tau
        # itself moves, by exactly beta * (mix_a v_k - mix_0 v_0). That shift is
        # part of C2's estimand (the oracle carries it, and a generator that
        # learned the interaction reproduces it), so it is checked exactly here
        # rather than required to vanish as in step C.
        k_comp = np.array([int(k.split("|", 1)[0]) for k in k1])
        uq, mix, _ = group_means(syn_c[:, None].astype(np.float64), keys)
        mix_of = dict(zip(uq.astype(str), mix[:, 0]))
        mix_a = np.array([mix_of[k] for k in k1])
        V = m["V"]
        expect = m["beta"] * (mix_a[:, None] * V[k_comp] - mix_of[CONTROL_ARM] * V[0][None, :])
        err = float(np.abs((t_syn - t_plain) - expect).max())
        check(err < 2e-3,
              f"compound mode: tau moves by exactly beta * (mix_a v_k - mix_0 v_0) "
              f"(max |diff| {err:.2e}); median shift norm "
              f"{float(np.median(np.linalg.norm(expect, axis=1))):.2f} -- syn_c now "
              f"modifies the effect, which is what step C2 needs (§3.8.4)")
    else:
        proj = (t_syn - t_plain) @ v
        # A balanced arm shifts by beta*(mean syn_c in arm - mean syn_c in DMSO) ~ 0;
        # the residue is the 1-well imbalance an odd-sized arm cannot avoid.
        scale = float(np.median(np.abs(proj)))
        check(scale < args.tau_tol * m["beta"],
              f"median |<tau_syn - tau_plain, v>| {scale:.3f} < {args.tau_tol} x beta "
              f"{m['beta']:.2f}; beta/6 = {m['beta'] / 6:.3f} is the 3-well arm's own "
              f"2/1 syn_c imbalance, which is the whole residue")
        check(abs(float(np.median(proj))) < 0.05 * m["beta"],
              f"<tau_syn - tau_plain, v> is centred at {float(np.median(proj)):+.3f}, "
              f"i.e. < 5% of beta {m['beta']:.2f}: syn_c is balanced within arms, so the "
              f"unablated oracle is (nearly) unchanged")
        print(f"      |projection| median {scale:.3f}, p90 "
              f"{float(np.percentile(np.abs(proj), 90)):.3f}, max {float(np.abs(proj).max()):.3f} "
              f"(beta {m['beta']:.2f}); the residue is the odd-well imbalance")

    # ---- responders ------------------------------------------------------
    rp = os.path.join(nz, args.responders)
    if os.path.isfile(rp):
        with open(rp) as fh:
            r = json.load(fh)
        det = {d["pert_id"]: d for d in r["detail"]}
        check(len(r["compounds"]) == len(det) == r["selection"]["n_scored"],
              f"responders.json: {len(r['compounds']):,} compounds == detail == n_scored")
        check(all(d["max_floor_ratio"] is not None
                  and d["max_floor_ratio"] > r["selection"]["mult"] for d in det.values()),
              f"every scored compound clears {r['selection']['mult']}x its noise floor")
        comp = meta["compound_idx"].values.astype(np.int64)
        ctl = meta["is_control"].values.astype(bool)
        half = dose_half(comp, meta["dose_level"].values.astype(np.float64), ctl)
        in_tr = np.zeros(len(meta), bool)
        in_tr[np.asarray(splits["train_idx"], dtype=np.int64)] = True
        want_cells = {f"{h}|syn_c={c}" for h in ("low", "high") for c in (0, 1)}
        bad_cells, few = [], []
        for d in det.values():
            ci = int(d["compound_idx"])
            sel = (comp == ci) & ~ctl
            if np.unique(meta["dose_level"].values[sel]).size < 2:
                few.append(ci); continue
            tr = sel & in_tr
            got = {f"{'high' if h == 1 else 'low'}|syn_c={c}"
                   for h, c in zip(half[tr], syn_c[tr])}
            if want_cells - got:
                bad_cells.append(ci)
        check(not few, f"every scored compound has >= 2 dose levels"
                       + (f" -- offenders {few[:5]}" if few else ""))
        check(not bad_cells, f"every scored compound has a train well in all four "
                             f"positivity cells" + (f" -- offenders {bad_cells[:5]}" if bad_cells else ""))
    else:
        print(f"skip  no {rp}")

    # ---- tier instances --------------------------------------------------
    tiers = args.tier or sorted(glob.glob(f"{nz}_tier_*"))
    if not tiers:
        print("skip  no tiered split dirs")
    for T in tiers:
        name = os.path.basename(T)
        with open(os.path.join(T, "splits.json")) as fh:
            ts = json.load(fh)
        with open(os.path.join(T, "tier_meta.json")) as fh:
            tm = json.load(fh)
        tier = ts["tier"]
        if not tier.get("active"):
            continue
        g, kf, pmin = float(tier["gamma"]), float(tier["keep_frac"]), float(tier["pmin"])
        comp = meta["compound_idx"].values.astype(np.int64)
        ctl = meta["is_control"].values.astype(bool)
        half = dose_half(comp, meta["dose_level"].values.astype(np.float64), ctl)
        nu = np.load(os.path.join(T, "nu_rows.npy"))
        in_nu = np.zeros(len(meta), bool); in_nu[nu] = True
        ci_list = [int(x) for x in tier["scored_compounds"].values()]
        recs, keep_obs = [], []
        kept = np.zeros(len(meta), bool)
        kept[np.asarray(ts["train_idx"], dtype=np.int64)] = True
        for ci in ci_list:
            rows = np.flatnonzero(in_nu & (comp == ci) & ~ctl)
            if rows.size == 0:
                continue
            recs.append(planned_bias(rows, half, meta["syn_c"].values, "syn_c", g, kf, pmin))
            keep_obs.append(kept[rows].mean())
        exp_keep = float(np.median([x["keep_frac_expected"] for x in recs]))
        obs_keep = float(np.median(keep_obs))
        check(abs(obs_keep - exp_keep) < 0.06,
              f"{name}: realised kept fraction {obs_keep:.3f} ~ planned {exp_keep:.3f}")
        dlo = float(np.median([x["dp_low"] for x in recs]))
        dhi = float(np.median([x["dp_high"] for x in recs]))
        if g == 0:
            check(abs(dlo) < 1e-9 and abs(dhi) < 1e-9,
                  f"{name}: gamma=0 selects uniformly, so dp is exactly 0 in both halves")
        else:
            check(dlo * dhi < 0 and abs(abs(dlo) - abs(dhi)) < 0.25 * max(abs(dlo), abs(dhi)),
                  f"{name}: gamma={g} gives equal-and-opposite dp across dose halves "
                  f"(low {dlo:+.4f}, high {dhi:+.4f}) -- the dose_half x syn_c lever")
            print(f"      {name}: expected bias along v = beta x dp = "
                  f"{m['beta'] * dlo:+.2f} (low) / {m['beta'] * dhi:+.2f} (high)")
        check(abs(float(tm.get("realised_kept_frac", obs_keep)) - obs_keep) < 0.06,
              f"{name}: tier_meta realised_kept_frac agrees with the table")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURE(S):\n  - " + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
