"""Read-back checks of P1, the post-hoc DR targeting (`understand.md` §3.3.1). No PyTorch.

  unit       delta_g against a closed form on synthetic arrays; Hajek scale
             invariance; HT == Hajek when sum_w = n; the refinement detector;
             tau_P1 = tau_gen + delta[g] elementwise
  weights    the counts identity sum_{i in g} w_i = n_nu(g) on disk, recomputed
             from splits.json + nu_rows.npy + dr_weights_*.npz with plain numpy
             and WITHOUT importing dr_target -- the module under test must not
             be its own witness
  artifacts  each targeted document: delta re-derived independently from
             expr.npy + the source _gen.npz; the tau identity; every untouched
             npz array identical to the source's; learned_syn_effect, quality
             and reference byte-identical; the baseline-reproduction record
  controls   the scientific checks, each with its expectation:
               - NULL on step C: `conditional` is already unbiased there, so
                 the correction must stay small
               - alpha sensitivity: with w = 1 the Hajek mean runs over the
                 THINNED mix, so the correction ADDS the thinning bias and must
                 come out WORSE than both the counts version and the baseline.
                 This is the evidence that P1 is not merely fitting the
                 statistic with one free parameter per group
               - unscored compounds, never thinned, must not move
               - pooled gene MSE must not regress
  report     `step_c_report` with its defaults must still reproduce the
             committed verdicts bit-identically after the --dr_arm refactor

Sections run when their artifacts exist. Exits nonzero on any failure.

    python -m src.tests.check_p1                       # everything it finds
    python -m src.tests.check_p1 --unit_only
    python -m src.tests.check_p1 --n_rederive 4
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import (  # noqa: E402
    CONTROL_ARM, arm_keys, dose_half, load_splits, positivity_cells, target_groups)
from src.data.synthetic import inject_meta, load_syn_meta  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, add_syn_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def _eq(a, b) -> bool:
    """Array equality that treats NaN as equal (reliability_cos is NaN on 1-well arms)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


# ---------------------------------------------------------------------------
# unit
# ---------------------------------------------------------------------------
def section_unit() -> None:
    from src.eval.dr_target import hajek_delta
    rng = np.random.default_rng(0)
    n, G, K = 400, 7, 5
    gi = rng.integers(0, K, n)
    w = rng.uniform(0.5, 4.0, n)
    s = rng.integers(0, 2, n).astype(np.float64)        # a binary "confounder"
    c_g = rng.normal(size=(K, G))                       # its per-group effect
    resid = (s[:, None] * c_g[gi]).astype(np.float32)
    delta, sw, sw2, nrows = hajek_delta(resid, w, gi, K)
    want = np.array([c_g[g] * (w[gi == g] @ s[gi == g]) / w[gi == g].sum() for g in range(K)])
    # `resid` is float32 (as it is in production), so the float64 closed form can
    # only agree to float32 epsilon on values of order 1.
    check(float(np.abs(delta - want).max()) < 1e-6,
          f"delta_g matches the closed form c_g (sum w s)/(sum w) "
          f"(max |diff| {float(np.abs(delta - want).max()):.1e})")
    d7, *_ = hajek_delta(resid, 7.0 * w, gi, K)
    check(float(np.abs(d7 - delta).max()) < 1e-9,
          "Hajek is scale invariant: w -> 7w leaves delta unchanged")
    w1 = np.ones(n)
    d1, sw1, _, n1 = hajek_delta(resid, w1, gi, K)
    ht = np.zeros_like(d1)
    for g in range(K):
        ht[g] = resid[gi == g].astype(np.float64).sum(0) / n1[g]
    check(float(np.abs(d1 - ht).max()) < 1e-12 and _eq(sw1, n1.astype(float)),
          "with w = 1 the Hajek and Horvitz-Thompson forms coincide (sum_w = n)")
    check(_eq(nrows, np.bincount(gi, minlength=K)), "row counts per group are right")
    g2, _, _, n2 = hajek_delta(resid[:1], w[:1], gi[:1], K)
    check(float(np.abs(g2[gi[0]] - resid[0]).max()) < 1e-6,
          "a one-row group's delta is exactly that row's residual")
    # gene blocking must not change the answer
    db, *_ = hajek_delta(resid, w, gi, K, gene_block=1)
    check(float(np.abs(db - delta).max()) < 1e-12,
          "the gene-block size does not change delta (blocking is pure bookkeeping)")
    # a per-group constant leaves any WITHIN-group contrast of tau untouched
    tau = rng.normal(size=(20, G))
    gof = rng.integers(0, K, 20)
    tau_p1 = tau + delta[gof]
    same = [i for i in range(20) for j in range(20) if i < j and gof[i] == gof[j]]
    ok = all(np.allclose((tau_p1[i] - tau_p1[j]), (tau[i] - tau[j]))
             for i in range(20) for j in range(20) if i < j and gof[i] == gof[j])
    check(ok and bool(same), "a per-group constant leaves within-group tau contrasts invariant")


# ---------------------------------------------------------------------------
# weights (independent of dr_target)
# ---------------------------------------------------------------------------
def section_weights(cfg, meta) -> None:
    comp = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool)
    g = target_groups("syn_c", comp, dl, ctl)
    cells = positivity_cells("syn_c", comp, dl, ctl, meta["syn_c"].values)
    trt = ~ctl
    check(len(set(zip(cells[trt].tolist(), g[trt].tolist()))) == len(set(cells[trt].tolist())),
          "positivity cells refine the target group (each cell inside exactly one group)")
    check(bool((g[ctl] == CONTROL_ARM).all()), "vehicles map to CONTROL_ARM, so they are "
                                               "excluded from every group")
    names = sorted({x for x in g[trt]})
    gidx = {x: i for i, x in enumerate(names)}
    base = os.path.join(cfg.paths.data_dir, "nuisances")
    nu = np.load(os.path.join(base, "nu_rows.npy"))
    nu_t = nu[trt[nu]]
    n_nu = np.bincount([gidx[x] for x in g[nu_t]], minlength=len(names))
    for tier in sorted(glob.glob(os.path.join(cfg.paths.data_dir, "nuisances_tier_Csyn_c_*"))):
        nm = os.path.basename(tier)
        with open(os.path.join(tier, "splits.json")) as fh:
            tr = np.asarray(json.load(fh)["train_idx"], dtype=np.int64)
        for mode in ("counts", "design"):
            wp = os.path.join(tier, f"dr_weights_{mode}.npz")
            if not os.path.isfile(wp):
                continue
            wz = np.load(wp)
            check(_eq(wz["row_id"], tr), f"{nm} {mode}: row_id == train_idx ({tr.size:,})")
            w = wz["w"].astype(np.float64)
            tt = trt[tr]
            sw = np.bincount([gidx[x] for x in g[tr[tt]]], weights=w[tt], minlength=len(names))
            have = (n_nu > 0) & (sw > 0)
            rel = float((np.abs(sw[have] - n_nu[have]) / np.maximum(n_nu[have], 1)).max())
            if mode == "counts":
                check(rel < 1e-6, f"{nm} counts: sum_w(g) == n_nu(g) exactly "
                                  f"(max rel err {rel:.2e}) -- HT and Hajek coincide")
            else:
                print(f"      {nm} design: max rel |sum_w - n_nu| {rel:.3f} "
                      f"(HT != Hajek; the Hajek form is why this is not asserted)")
            check(int(((n_nu > 0) & (sw <= 0)).sum()) == 0,
                  f"{nm} {mode}: every group with pool rows keeps a train row")


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def section_artifacts(cfg, meta, targeted: list[str], n_rederive: int) -> None:
    comp = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool)
    expr_raw = np.load(cfg.paths.expr_npy, mmap_mode="r")
    y_cache: dict = {}
    done: set = set()
    for path in targeted:
        with open(path) as fh:
            t = json.load(fh)
        tg = t["targeting"]
        src = tg["source_json"]
        with open(src) as fh:
            s = json.load(fh)
        nm = os.path.basename(path)
        check(tg["method"] == "P1" and tg["version"] == 1 and tg["arm"].startswith("p1_"),
              f"{nm}: targeting block is P1 v1, arm {tg['arm']}")
        bc = tg["pools"]["all"]["baseline_check"]
        check(bc["max_abs_diff"] <= bc["tol"],
              f"{nm}: the source metrics were reproduced before correcting "
              f"(max |diff| {bc['max_abs_diff']:.1e} <= {bc['tol']})")
        # Step-C documents predate `learned_syn_effect` (it arrived with the C2
        # code), so compare whichever inherited keys the SOURCE actually has --
        # and require that P1 neither dropped nor invented one.
        inherit = ("learned_syn_effect", "quality", "reference", "per_compound")
        same, missing = [], []
        for pl in s["pools"]:
            for k in inherit:
                if k in s["pools"][pl]:
                    if k not in t["pools"][pl]:
                        missing.append(f"{pl}.{k}")
                    else:
                        same.append(json.dumps(s["pools"][pl][k])
                                    == json.dumps(t["pools"][pl][k]))
                elif k in t["pools"][pl]:
                    missing.append(f"{pl}.{k} (invented)")
        check(not missing and all(same),
              f"{nm}: the {len(same)} inherited blocks "
              f"(learned_syn_effect / quality / reference / per_compound, where the "
              f"source has them) are byte-identical"
              + (f" -- missing {missing}" if missing else ""))
        zs = np.load(src[:-5] + "_tau.npz")
        tp = path[:-5] + "_tau.npz"
        if not os.path.isfile(tp):
            print(f"skip  {nm}: no _tau.npz (written with --npz none)")
            continue
        z = np.load(tp)
        changed = ("tau_gen", "tau_norm_gen", "arm_cos", "arm_pearson")
        kept = [k for k in zs.files if not k.endswith(changed) and k in z.files]
        check(all(_eq(z[k], zs[k]) for k in kept),
              f"{nm}: every array P1 does not touch and still carries ({len(kept)}) is "
              f"identical to the source's")
        d = z["targeting/delta"]
        # The arm -> group map, re-derived from the WHOLE TABLE. dose_half ranks
        # a compound's distinct dose levels, so deriving it from a pool's own
        # arm table silently shifts the halves of compounds whose arms the pool
        # dropped, and those arms then get the other half's delta. Caught in
        # review 2026-10-04; this is the check that would have caught it.
        names_l = [str(x) for x in z["targeting/group_name"]]
        gidx_l = {x: i for i, x in enumerate(names_l)}
        g_tab = dict(zip(arm_keys(comp, dl, ctl), target_groups(tg["confounder"], comp, dl, ctl)))
        for p in s["pools"]:
            if f"{p}/group_of_arm" not in z.files:
                continue
            want = np.array([gidx_l.get(str(g_tab.get(str(k))), -1)
                             for k in z[f"{p}/arm_key"]], dtype=np.int64)
            got = z[f"{p}/group_of_arm"]
            bad = int((want != got).sum())
            check(bad == 0, f"{nm} [{p}]: the arm -> group map matches the whole-table "
                            f"derivation ({bad} arms differ)")
        for p in s["pools"]:
            if f"{p}/tau_gen" not in z.files:      # written with --npz delta
                continue
            gof = z[f"{p}/group_of_arm"]
            check(np.allclose(z[f"{p}/tau_gen"], zs[f"{p}/tau_gen"] + d[gof].astype(np.float32),
                              atol=2e-4),
                  f"{nm} [{p}]: tau_P1 == tau_gen + delta[g] elementwise")
        # ---- re-derive delta from the raw inputs, independently of dr_target ----
        key = (tg["source_arm"], tg["weights"]["mode"], tg["nuisance_dir"])
        if len(done) >= n_rederive or key in done:
            continue
        done.add(key)
        nz = tg["nuisance_dir"]
        import copy as _c
        a = _c.copy(ARGS)
        a.nuisance_dir = None
        a.syn_effect = float(s.get("syn_effect") or 0.0)
        a.syn_seed = int(s.get("syn_seed") or 0)
        a.syn_meta = (s.get("syn") or {}).get("name")
        c2 = apply_paths_args(config_from_args(a), a)
        sp = load_splits(c2)
        m = load_expr_meta(c2, s["plate_center"], splits=sp)
        ck = (m["split_fingerprint"], a.syn_meta)
        if ck not in y_cache:
            y = normalize_expr(np.asarray(expr_raw),
                               plate_codes(meta["det_plate"].values, m["plates"]), m)
            if c2.outcome.syn_effect != 0:
                y = inject_meta(y, meta["syn_c"].values.astype(np.int64), comp,
                                load_syn_meta(c2, n_genes=c2.outcome.n_genes))
            y_cache[ck] = y
        y_all = y_cache[ck]
        with open(os.path.join(nz, "splits.json")) as fh:
            tr = np.asarray(json.load(fh)["train_idx"], dtype=np.int64)
        w = (np.ones(tr.size, dtype=np.float64) if tg["weights"]["is_control"]
             else np.load(tg["weights"]["path"])["w"].astype(np.float64))
        gz = np.load(tg["source_gen_npz"])
        pos = np.full(len(meta), -1, dtype=np.int64); pos[gz["row_id"]] = np.arange(gz["row_id"].size)
        g = target_groups(tg["confounder"], comp, dl, ctl)
        names = list(z["targeting/group_name"])
        gidx = {str(x): i for i, x in enumerate(names)}
        tt = ~ctl[tr]
        rows, ww = tr[tt], w[tt]
        r = y_all[rows].astype(np.float64) - gz["row_mean"][pos[rows]].astype(np.float64)
        gi = np.array([gidx[str(x)] for x in g[rows]], dtype=np.int64)
        num = np.zeros((len(names), r.shape[1]), dtype=np.float64)
        np.add.at(num, gi, r * ww[:, None])
        sw = np.bincount(gi, weights=ww, minlength=len(names))
        mine = np.zeros_like(num); ok = sw > 0
        mine[ok] = num[ok] / sw[ok, None]
        err = float(np.abs(mine - d).max() / max(float(np.abs(d).max()), 1e-12))
        check(err < 1e-5, f"{nm}: delta re-derived from expr.npy + _gen.npz matches "
                          f"(max rel diff {err:.1e})")
        del r, num


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
def _did(files: list[str], pool: str = "all") -> dict:
    """{arm: (mean DiD scored, mean DiD unscored, n_seeds)} from eval JSONs."""
    cells: dict = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        tg = d.get("targeting")
        arm = tg["arm"] if tg else None
        if arm is None:
            continue
        run = d["generator"]["run"]
        gam = float(json.load(open(os.path.join(d["generator"]["run_dir"],
                                                "arch.json")))["tier"]["gamma"])
        seed = int(run.rsplit("_s", 1)[1])
        bv = ((d["pools"].get(pool) or {}).get("accuracy") or {}).get("bias_along_v")
        if bv:
            cells[(arm, gam, seed)] = (bv["scored_high_minus_low_mean"],
                                       bv["unscored_high_minus_low_mean"])
    out: dict = {}
    for arm in sorted({k[0] for k in cells}):
        sc, un = [], []
        for seed in sorted({k[2] for k in cells if k[0] == arm}):
            a, b = cells.get((arm, 1.0, seed)), cells.get((arm, 0.0, seed))
            if a and b:
                sc.append(a[0] - b[0]); un.append(a[1] - b[1])
        if sc:
            out[arm] = (float(np.mean(sc)), float(np.mean(un)), len(sc))
    return out


def _base_did(pattern: str, labels: dict) -> dict:
    """The same DiD for the UNTARGETED source arms, keyed by arm label."""
    cells: dict = {}
    for f in sorted(glob.glob(pattern)):
        if "_p1" in os.path.basename(f):
            continue
        with open(f) as fh:
            d = json.load(fh)
        run = d["generator"]["run"]
        arch = json.load(open(os.path.join(d["generator"]["run_dir"], "arch.json")))
        arm = labels.get((tuple(arch.get("adjustment_set") or ()), str(arch.get("dr_mode")),
                          str(arch.get("dr_weights_file"))))
        if arm is None:
            continue
        bv = ((d["pools"]["all"].get("accuracy") or {})).get("bias_along_v")
        if bv:
            cells[(arm, float(arch["tier"]["gamma"]), int(run.rsplit("_s", 1)[1]))] = (
                bv["scored_high_minus_low_mean"], bv["unscored_high_minus_low_mean"])
    out: dict = {}
    for arm in sorted({k[0] for k in cells}):
        sc = [cells[(arm, 1.0, s)][0] - cells[(arm, 0.0, s)][0]
              for s in sorted({k[2] for k in cells if k[0] == arm})
              if (arm, 1.0, s) in cells and (arm, 0.0, s) in cells]
        if sc:
            out[arm] = float(np.mean(sc))
    return out


def section_controls(runs_dir: str) -> None:
    labels = {((), "conditional", "None"): "naive",
              (("syn_c",), "conditional", "None"): "conditional"}
    for fam, tag in (("stepc2", "syn1-cmp1"), ("stepc", "syn1")):
        pat = os.path.join(runs_dir, f"{fam}_*", "eval_artifacts", f"gen_*_{tag}*.json")
        files = [f for f in sorted(glob.glob(pat)) if "_p1" in os.path.basename(f)]
        if not files:
            print(f"skip  no targeted documents for {fam}")
            continue
        did = _did(files)
        base = _base_did(pat, labels)
        print(f"\n      [{fam}] DiD of the scored high-low contrast (mean over seeds)")
        for arm, (sc, un, n) in sorted(did.items()):
            print(f"        {arm:18s} scored {sc:+8.3f}  unscored {un:+7.3f}  ({n} seeds)")
        for src, b in sorted(base.items()):
            print(f"        {src + ' (source)':18s} scored {b:+8.3f}")
        for short, src in (("cond", "conditional"), ("naive", "naive")):
            cnt, one = did.get(f"p1_{short}_counts"), did.get(f"p1_{short}_ones")
            b = base.get(src)
            if cnt and b is not None:
                if fam == "stepc" and src == "conditional":
                    # step C's `conditional` is already unbiased, so this is a NULL
                    # check: the correction must not introduce a bias.
                    ref = abs(base.get("naive", 1.0))
                    check(abs(cnt[0]) < 0.25 * ref,
                          f"[{fam}] NULL: p1_cond_counts stays small ({cnt[0]:+.3f}, "
                          f"< 0.25 x |naive| {0.25 * ref:.3f})")
                else:
                    check(abs(cnt[0]) < abs(b),
                          f"[{fam}] p1_{short}_counts reduces the bias "
                          f"({b:+.3f} -> {cnt[0]:+.3f})")
                check(abs(cnt[1]) < 0.05,
                      f"[{fam}] p1_{short}_counts leaves unscored compounds alone "
                      f"({cnt[1]:+.4f})")
            if one and cnt and b is not None:
                check(abs(one[0]) > 3 * abs(cnt[0]) and abs(one[0]) > abs(b),
                      f"[{fam}] ALPHA SENSITIVITY: w=1 is far worse than counts "
                      f"({one[0]:+.3f} vs {cnt[0]:+.3f}; source {b:+.3f}) -- the weights "
                      f"are load-bearing, so P1 is not fitting the statistic")
    # ---- what targeting costs in MSE ----
    # P1 trades variance for bias: it adds one estimated constant per group. Where
    # there IS bias to remove (C2, whose outcome model learns only ~half the
    # interaction) the trade pays and MSE falls. Where there is NOT (step C, where
    # `conditional` already learns the shared shift) delta_g is an estimate of ~0
    # built from a handful of rows, so it is pure added variance. That is a
    # finding, not a fault, so it is asserted only where the correction has
    # something to correct.
    by_fam: dict = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, "*", "eval_artifacts", "gen_*_p1*.json"))):
        with open(f) as fh:
            t = json.load(fh)
        if (t["targeting"]["weights"] or {}).get("is_control"):
            continue
        with open(t["targeting"]["source_json"]) as fh:
            src = json.load(fh)
        a, b = (t["pools"]["all"]["accuracy"]["pooled_gene"]["mse"],
                src["pools"]["all"]["accuracy"]["pooled_gene"]["mse"])
        fam = "stepc2" if "stepc2_" in f else "stepc"
        by_fam.setdefault((fam, t["targeting"]["arm"]), []).append(a / b - 1.0)
    print("\n      pooled gene MSE, targeted vs source (mean over runs)")
    for (fam, arm), v in sorted(by_fam.items()):
        print(f"        {fam:7s} {arm:18s} {100 * float(np.mean(v)):+6.1f}%  ({len(v)} runs)")
    for (fam, arm), v in sorted(by_fam.items()):
        if fam == "stepc2":
            check(float(np.mean(v)) <= 0.02,
                  f"[{fam}] {arm}: pooled gene MSE does not regress "
                  f"({100 * float(np.mean(v)):+.1f}%)")
        else:
            print(f"      [{fam}] {arm}: MSE {100 * float(np.mean(v)):+.1f}% -- expected: "
                  f"step C has no bias to remove, so delta_g is added variance")


# ---------------------------------------------------------------------------
# report regression
# ---------------------------------------------------------------------------
def section_report(runs_dir: str) -> None:
    for variant, syn in (("", "syn_meta.json"), ("_compound_r1", "syn_meta_compound_r1.json")):
        committed = os.path.join(runs_dir, "eval_artifacts", f"step_c_verdict{variant}.json")
        if not os.path.isfile(committed):
            print(f"skip  no committed {os.path.basename(committed)}")
            continue
        with open(committed) as fh:
            old = json.load(fh)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            tmp = fh.name
        argv = sys.argv
        try:
            sys.argv = ["step_c_report", "--syn_meta", syn, "--out", tmp]
            runpy.run_module("src.eval.step_c_report", run_name="__main__")
        except SystemExit as e:
            if e.code:
                check(False, f"step_c_report --syn_meta {syn} exited {e.code}")
                continue
        finally:
            sys.argv = argv
        with open(tmp) as fh:
            new = json.load(fh)
        os.unlink(tmp)
        # `skipped` and `n_arm_runs` grow with the targeted documents the report
        # now sees and declines, and `per_arm` legitimately gains any targeted
        # arm that has runs on disk. What must NOT move is the verdict itself:
        # every criterion, and the numbers of the arms it judges.
        drop = ("skipped", "n_arm_runs", "cells", "out", "per_arm")
        judged = set(((old.get("criteria") or {})
                      .get("3_unscored_unchanged") or {}).get("by_arm") or {})
        a = {k: v for k, v in old.items() if k not in drop}
        b = {k: v for k, v in new.items() if k not in drop}
        # criterion 2 gained an informational `arms` key; it names the defaults.
        for d in (a, b):
            c2 = (d.get("criteria") or {}).get("2_dr_beats_conditional_and_approaches_design")
            if c2:
                c2.pop("arms", None)
            c4 = (d.get("criteria") or {}).get("4_weights_track_design")
            if c4:
                c4.pop("targeted_runs_checked", None)
                c4.pop("targeting_bad", None)
                c4["note"] = ""
            dg = d.get("diagnostics")
            if dg:
                dg.pop("baseline_arm", None)      # new, informational
                dg["note"] = ""
        a["per_arm"] = {k: v for k, v in (old.get("per_arm") or {}).items() if k in judged}
        b["per_arm"] = {k: v for k, v in (new.get("per_arm") or {}).items() if k in judged}
        same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        check(same, f"step_c_report defaults still reproduce step_c_verdict{variant}.json "
                    f"(every criterion and every judged arm) after the --dr_arm refactor")
        if not same:
            for k in sorted(set(a) | set(b)):
                if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True):
                    print(f"      differs: {k}")
                    if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
                        for kk in sorted(set(a[k]) | set(b[k])):
                            if json.dumps(a[k].get(kk), sort_keys=True) != \
                                    json.dumps(b[k].get(kk), sort_keys=True):
                                print(f"        - {kk}")


ARGS = None


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--unit_only", action="store_true")
    p.add_argument("--n_rederive", type=int, default=2,
                   help="How many (source arm, weight mode) combinations to re-derive "
                        "delta for from the raw inputs; each costs a y_all build.")
    p.add_argument("--targeted", action="append", default=[])
    p.add_argument("--skip_report", action="store_true")
    add_syn_cli(p)
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    ARGS = args

    print("== unit")
    section_unit()
    if args.unit_only:
        print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILURE(S)"))
        sys.exit(1 if FAILS else 0)

    cfg = apply_paths_args(config_from_args(args), args, require_splits=False)
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "syn_c", "det_plate"])
            .to_pandas())
    runs_dir = cfg.paths.train_output_dir

    print("\n== weights")
    section_weights(cfg, meta)

    targeted = args.targeted or sorted(glob.glob(
        os.path.join(runs_dir, "*", "eval_artifacts", "gen_*_p1*.json")))
    print(f"\n== artifacts ({len(targeted)} targeted document(s))")
    if targeted:
        section_artifacts(cfg, meta, targeted, args.n_rederive)
    else:
        print("skip  none found")

    print("\n== controls")
    if targeted:
        section_controls(runs_dir)
    else:
        print("skip  none found")

    if not args.skip_report:
        print("\n== report regression")
        section_report(runs_dir)

    print("\n" + ("ALL CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURE(S):\n  - " + "\n  - ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
