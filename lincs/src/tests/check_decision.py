"""Checks of the decision task (`DECISION.md` §8, N3). No PyTorch.

Synthetic (always; a few seconds, no data): a small world with a planted best
dose per compound, a planted line effect and a line-dependent thinning.

  effects   `effects_from_rows` against plain loops; a per-line vehicle offset
            cancels; `pooled` is the arm's mean at its own line mix
  design    eligible doses, their ranks, the >= min_doses rule
  policies  with no noise, the stratified mean recovers every planted best dose
            from the thinned wells and the pooled mean does not; its dose moves
            in the direction `g2_share_shift` predicts
  invariance  a constant added per line to a generator's rows changes no effect
            and no decision
  errors    the jackknife of a mean; `topk_value` on bootstrap multiplicities
            against an explicit resample
  criteria  the net difference splits exactly into D4's and D3's quantities;
            the reference captures 1 and `random` gains 0

Real data (with --data_dir; a CPU job, see scripts/decision_cpu.sub). It reads
NO decision number of any generator:

  frame     the all-wells effect at each arm's own line mix = the stored oracle;
            the holdout line contrast = the oracle's `contrast_dir`
  join      one run's generated rows, grouped the same way = its stored `tau_gen`
  tiers     both instances thin the same compounds, keep a train well in every
            (arm, line) cell, and leave the unthinned compounds' rows alone
  axis      task B's genes are all landmarks and the axis is a unit vector

    python -m src.tests.check_decision                        # synthetic only
    python -m src.tests.check_decision --data_dir data/core5_24h
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import decision_task as dt  # noqa: E402
from src.spec import (  # noqa: E402
    DECISION_K_CONTROL, DECISION_MIN_DOSES, PROLIFERATION_GENES, add_adjustment_set_cli, add_paths_cli,
    apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


# ---------------------------------------------------------------------------
# the synthetic world
# ---------------------------------------------------------------------------
G2 = np.array([False, False, False, True, True])           # three G1 lines, two G2 lines
# The planted dose curve. Compounds whose G2 lines respond MORE have their best dose
# at index 3 (the first of the high half), the others the mirror image (index 2), so
# the thinning below pulls every thinned compound's pooled mean across the halves.
CURVE = np.array([0.2, 0.5, 0.9, 1.0, 0.8, 0.6])


def world(rng, n_comp=80, n_genes=24, noise=0.0, line_offset=0.0):
    """Rows of a table with n_comp compounds x 6 doses x 5 lines, 3 train wells and
    1 holdout well per cell (none for the dropped cells), and vehicles per line.

    truth of a cell = curve_c[d] * amp_c * dir_c * (1 + h_c * z_line), z = +1 on G2
    lines and -1 on G1 lines, curve_c = CURVE (h_c > 0) or its mirror image (h_c < 0);
    each line's vehicles sit at their own offset.
    Returns the rows and the planted quantities."""
    C, D, L = n_comp, CURVE.size, G2.size
    z = np.where(G2, 1.0, -1.0)
    dirs = rng.normal(size=(C, n_genes))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    amp = rng.uniform(2.0, 4.0, size=C)
    h = np.where(np.arange(C) % 2 == 0, 0.8, -0.8)          # G2 responds more / less
    offs = line_offset * rng.normal(size=(L, n_genes))
    curve = np.where(h[:, None] > 0, CURVE[None], CURVE[None, ::-1])
    cell = (curve[:, :, None, None] * amp[:, None, None, None] * dirs[:, None, None, :]
            * (1 + h[:, None, None, None] * z[None, None, :, None]))        # (C, D, L, G)
    # holdout: compound c loses its holdout well in line 0 at dose (c % 7) when that is < D,
    # and at two more doses for every 5th compound (which then has 3 eligible doses and drops out)
    drop = np.zeros((C, D), dtype=bool)
    for c in range(C):
        if c % 7 < D:
            drop[c, c % 7] = True
        if c % 5 == 4:
            drop[c, [(c + 1) % D, (c + 3) % D, (c + 4) % D]] = True
    rows = []                                               # (arm, line, ctl, train, comp, dose index)
    for c in range(C):
        for d in range(D):
            for l in range(L):
                rows += [(c * D + d, l, 0, 1, c, d)] * 3
                if not (drop[c, d] and l == 0):
                    rows += [(c * D + d, l, 0, 0, c, d)]
    for l in range(L):
        rows += [(-1, l, 1, 1, -1, -1)] * 30 + [(-1, l, 1, 0, -1, -1)] * 10
    r = np.array(rows, dtype=np.int64)
    arm, line, ctl, train = r[:, 0], r[:, 1], r[:, 2].astype(bool), r[:, 3].astype(bool)
    y = offs[line].copy()
    t = ~ctl
    y[t] += cell[r[t, 4], r[t, 5], line[t]]
    y += noise * rng.normal(size=y.shape)
    # the thinning at "gamma = 1": the low dose half keeps 1 of its 3 train wells in the G1
    # lines, the high half 1 of 3 in the G2 lines, for the first 60% of the compounds
    thinned = np.arange(C) < int(0.6 * C)
    kept1 = train.copy()
    seen: dict[tuple, int] = {}
    for i in np.flatnonzero(t & train):
        c, d, l = int(r[i, 4]), int(r[i, 5]), int(line[i])
        k = seen[(c, d, l)] = seen.get((c, d, l), 0) + 1
        if thinned[c] and k > 1 and ((d < D // 2) != bool(G2[l])):
            kept1[i] = False
    return {"y": y.astype(np.float32), "arm": arm, "line": line, "ctl": ctl, "train": train, "kept1": kept1,
            "comp_of_arm": np.repeat(np.arange(C), D) + 1, "dose_of_arm": np.tile(0.1 * 3.0 ** np.arange(D), C),
            "half": np.tile((np.arange(D) >= D // 2).astype(np.int64), C), "thinned": thinned,
            "cell": cell, "offs": offs, "h": h, "dirs": dirs, "amp": amp, "drop": drop, "K": C * D, "L": L}


def run_synthetic() -> None:
    rng = np.random.default_rng(0)
    # ---- effects ---------------------------------------------------------------------------
    w = world(rng, noise=0.3, line_offset=1.0)
    K, L = w["K"], w["L"]
    e = dt.effects_from_rows(w["y"], w["arm"], w["line"], w["ctl"], K, L, g2_lines=G2)
    y64 = w["y"].astype(np.float64)
    veh = np.stack([y64[w["ctl"] & (w["line"] == l)].mean(0) for l in range(L)])
    worst_eq = worst_pool = worst_con = 0.0
    for a in rng.choice(K, size=40, replace=False):
        cm = np.stack([y64[(w["arm"] == a) & ~w["ctl"] & (w["line"] == l)].mean(0) for l in range(L)])
        dev = cm - veh
        worst_eq = max(worst_eq, np.abs(e["eq"][a] - dev.mean(0)).max())
        worst_con = max(worst_con, np.abs(e["contrast"][a] - (dev[G2].mean(0) - dev[~G2].mean(0))).max())
        pooled = y64[(w["arm"] == a) & ~w["ctl"]].mean(0) - y64[w["ctl"]].mean(0)
        worst_pool = max(worst_pool, np.abs(e["pooled"][a] - pooled).max())
    check(max(worst_eq, worst_con, worst_pool) < 1e-9,
          f"effects_from_rows = plain loops on 40 arms: eq {worst_eq:.1e}, contrast {worst_con:.1e}, pooled {worst_pool:.1e}")
    check(e["cnt"].shape == (K, L) and (e["cnt"] == 4 - (w["drop"].reshape(-1)[:, None] & (np.arange(L) == 0))).all(),
          "cell counts: 4 rows per cell, 3 where the holdout well was dropped")
    # exact effects with no noise: the per-line vehicle offsets cancel
    w0 = world(np.random.default_rng(1), noise=0.0, line_offset=1.0)
    K, L = w0["K"], w0["L"]
    truth_eq = w0["cell"].mean(axis=2).reshape(K, -1)
    e0 = dt.effects_from_rows(w0["y"], w0["arm"], w0["line"], w0["ctl"], K, L)
    check(np.abs(e0["eq"] - truth_eq).max() < 1e-5,
          f"no noise: the equal-weight effect is the planted one although each line's vehicle sits at "
          f"its own offset (max |diff| {np.abs(e0['eq'] - truth_eq).max():.1e})")
    h_ = ~w0["train"]
    eh = dt.effects_from_rows(w0["y"][h_], w0["arm"][h_], w0["line"][h_], w0["ctl"][h_], K, L, g2_lines=G2)
    check(np.isnan(eh["eq"]).all(axis=1).sum() == int(w0["drop"].sum()) and
          np.isnan(eh["eq"][w0["drop"].reshape(-1)]).all(),
          f"an arm without a holdout well in some line has no holdout effect ({int(w0['drop'].sum())} arms)")

    # ---- design ----------------------------------------------------------------------------
    eligible = (eh["cnt"] > 0).all(axis=1)
    thinned_ids = (np.flatnonzero(w0["thinned"]) + 1).tolist()
    des = dt.build_design(w0["comp_of_arm"], w0["dose_of_arm"], eligible, thinned_ids)
    n_el = (~w0["drop"]).sum(axis=1)
    enter = n_el >= DECISION_MIN_DOSES
    check(np.array_equal(des.comp, np.flatnonzero(enter) + 1) and 0 < (~enter).sum() < enter.size,
          f"design: {des.comp.size} of {enter.size} compounds enter (>= {DECISION_MIN_DOSES} eligible doses); "
          f"{int((~enter).sum())} dropped")
    ok_rank = True
    for i, c in enumerate(des.comp - 1):
        keep = np.flatnonzero(~w0["drop"][c])
        ok_rank &= np.array_equal(des.rank[i][des.valid[i]], keep)
        ok_rank &= np.array_equal(des.arm[i][des.valid[i]], c * CURVE.size + keep)
    check(ok_rank, "design: each compound's eligible arms in ascending dose, with their rank among all its doses")
    check(np.array_equal(des.thinned, w0["thinned"][des.comp - 1]), "design: the thinned flag follows the compound")

    # ---- policies on the noise-free world ----------------------------------------------------
    S = dt.compound_axes(eh["eq"], des)
    cidx = des.comp - 1
    check(np.abs(np.abs(dt.dot(S[des.arm[:, 0]], w0["dirs"][cidx])) - 1).max() < 1e-6,
          "task A's axis is the compound's planted direction (up to sign)")
    truth = dt.dot(truth_eq, S)
    best_planted = np.nanargmax(des.take(truth), axis=1)        # NaN on the padding
    kept = {0.0: w0["train"], 1.0: w0["kept1"]}
    real = {}
    for name, msk in (("train", w0["train"]), ("kept|g0", kept[0.0]), ("kept|g1", kept[1.0])):
        real[name] = dt.effects_from_rows(w0["y"][msk], w0["arm"][msk], w0["line"][msk], w0["ctl"][msk], K, L)
    check(all((r["cnt"] > 0).all() for r in real.values()), "the thinning keeps a train well in every (arm, line) cell")
    strat = des.choose(dt.dot(real["kept|g1"]["eq"], S))
    pool = des.choose(dt.dot(real["kept|g1"]["pooled"], S))
    pool0 = des.choose(dt.dot(real["kept|g0"]["pooled"], S))
    th = des.thinned
    check(np.array_equal(strat, best_planted),
          "no noise: the stratified mean of the THINNED wells picks every planted best dose")
    check(np.array_equal(pool[~th], best_planted[~th]) and np.array_equal(pool0, best_planted),
          "no noise: the pooled mean picks the planted dose where nothing was thinned")
    moved = float((pool[th] != best_planted[th]).mean())
    check(moved > 0.8, f"the pooled mean of the thinned wells picks another dose for {100 * moved:.0f}% of the "
                       f"thinned compounds")
    wsyn = {"g2_lines": G2, "half": w0["half"], "kept": kept}
    shift = dt.g2_share_shift(wsyn, real, des, 1.0)
    check(abs(shift["g0"]["low"]) < 1e-12 and abs(shift["g0"]["high"]) < 1e-12
          and shift["g1"]["low"] > 0.2 and shift["g1"]["high"] < -0.2 and shift["tilt_sign"] == -1.0,
          f"g2_share_shift: none at gamma = 0; low {shift['g1']['low']:+.3f}, high {shift['g1']['high']:+.3f} at gamma = 1")
    con = dt.dot(eh["contrast"], S)
    kappa = np.nanmean(des.take(con), axis=1)
    check(np.array_equal(np.sign(kappa), np.sign(w0["h"][cidx] * dt.dot(S[des.arm[:, 0]], w0["dirs"][cidx]))),
          "kappa (the holdout G2 - G1 contrast on the axis) has the sign of the planted line effect")
    mv = (des.at(des.rank, pool) - des.at(des.rank, pool0))[th]
    pred = (shift["tilt_sign"] * np.sign(kappa))[th]
    check((mv[mv != 0] * pred[mv != 0] > 0).all(),
          "every dose the pooled mean moves, it moves in the predicted direction")

    # ---- a per-line constant on a generator's rows --------------------------------------------
    gen = w0["cell"].reshape(K, L, -1)[np.clip(w0["arm"], 0, None), w0["line"]].copy()
    gen[w0["ctl"]] = 0.0
    gen += 0.05 * np.random.default_rng(2).normal(size=gen.shape)
    bump = gen + 3.0 * np.random.default_rng(3).normal(size=(L, gen.shape[1]))[w0["line"]]
    ea = dt.effects_from_rows(gen, w0["arm"], w0["line"], w0["ctl"], K, L)
    eb = dt.effects_from_rows(bump, w0["arm"], w0["line"], w0["ctl"], K, L)
    check(np.abs(ea["eq"] - eb["eq"]).max() < 1e-9 and
          np.array_equal(des.choose(dt.dot(ea["eq"], S)), des.choose(dt.dot(eb["eq"], S))),
          f"a constant per line on the generated rows changes no effect (max |diff| "
          f"{np.abs(ea['eq'] - eb['eq']).max():.1e}) and no decision")

    # ---- errors -----------------------------------------------------------------------------
    x = rng.normal(size=57)
    check(abs(dt.Est.mean(x).se - x.std(ddof=1) / np.sqrt(x.size)) < 1e-12,
          "the jackknife SE of a mean is sd / sqrt(n)")
    a, b = dt.Est.mean(x), dt.Est.mean(x + 0.1 * rng.normal(size=x.size))
    check((a - b).se < 0.5 * a.se, "a paired difference of two correlated means has a much smaller SE")
    n, k = 37, 9
    score, val = rng.normal(size=n), rng.normal(size=n)
    counts = dt.bootstrap_counts(n, 200, 0, "check")
    fast = dt.topk_value(score, val, k, counts)
    slow = []
    for cnt in counts:
        idx = np.repeat(np.arange(n), cnt)
        slow.append(val[idx[np.argsort(-score[idx], kind="stable")[:k]]].mean())
    check(counts.shape == (200, n) and (counts.sum(1) == n).all() and np.abs(fast - np.array(slow)).max() < 1e-12
          and abs(dt.topk_value(score, val, k) - val[np.argsort(-score)[:k]].mean()) < 1e-12,
          "topk_value on bootstrap multiplicities = the top-k of each explicit resample")
    check(np.array_equal(counts, dt.bootstrap_counts(n, 200, 0, "check"))
          and not np.array_equal(counts, dt.bootstrap_counts(n, 200, 0, "other")),
          "bootstrap_counts is reproducible per (seed, name)")

    # ---- the whole analysis on a noisy world ---------------------------------------------------
    wn = world(np.random.default_rng(4), n_comp=400, noise=1.0, line_offset=0.5)
    K, L = wn["K"], wn["L"]
    hm = ~wn["train"]
    eh = dt.effects_from_rows(wn["y"][hm], wn["arm"][hm], wn["line"][hm], wn["ctl"][hm], K, L, g2_lines=G2)
    des = dt.build_design(wn["comp_of_arm"], wn["dose_of_arm"], (eh["cnt"] > 0).all(axis=1),
                          (np.flatnonzero(wn["thinned"]) + 1).tolist())
    S = dt.compound_axes(eh["eq"], des)
    kept = {0.0: wn["train"], 1.0: wn["kept1"]}
    real = {"train": dt.effects_from_rows(wn["y"][wn["train"]], wn["arm"][wn["train"]], wn["line"][wn["train"]],
                                          wn["ctl"][wn["train"]], K, L)}
    for g, msk in kept.items():
        real[f"kept|g{g:g}"] = dt.effects_from_rows(wn["y"][msk], wn["arm"][msk], wn["line"][msk], wn["ctl"][msk], K, L)
    s_b = np.zeros(wn["y"].shape[1]); s_b[:6] = -1 / np.sqrt(6)
    shift = dt.g2_share_shift({"g2_lines": G2, "half": wn["half"], "kept": kept}, real, des, 1.0)
    gr = np.random.default_rng(5)

    def noisy(t, sd):
        return t + sd * gr.normal(size=t.shape)
    scores = {"A": {}, "B": {}}

    def add(name, tau):
        scores["A"].setdefault(name, []).append(dt.dot(tau, S))
        scores["B"].setdefault(name, []).append(np.einsum("ag,g->a", tau, s_b))
    add(dt.REFERENCE, real["train"]["eq"])
    for g in (0.0, 1.0):
        r = real[f"kept|g{g:g}"]
        add(dt.pkey("pooled_real", g), r["pooled"])
        add(dt.pkey("stratified_real", g), r["eq"])
        add(dt.pkey("naive", g), noisy(r["pooled"], 0.05))
        for arm, sd in (("conditional", 0.05), ("dr", 0.10), ("dr_p2", 0.08)):
            for _ in range(2):
                add(dt.pkey(arm, g), noisy(r["eq"], sd))
    con = {"A": dt.dot(eh["contrast"], S), "B": np.einsum("ag,g->a", eh["contrast"], s_b)}
    for task in ("A", "B"):
        res = dt.analyse_task(scores[task], dict(A=dt.dot(eh["eq"], S), B=np.einsum("ag,g->a", eh["eq"], s_b))[task],
                              des, screens=(task == "B"), kappa=np.nanmean(des.take(con[task]), axis=1),
                              tilt_sign=shift["tilt_sign"], g1=1.0, n_boot=300, seed=0)
        pc = res.pop("_per_compound")
        json.dumps(res)                                    # serialisable as written
        crit = res["criteria"]
        resid = max(crit[f"net_{a}"]["split_residual"] for a in dt.ADIGEN_ARMS)
        add_up = max(abs(crit[f"net_{a}"]["value"] - crit[f"D4_no_cost_{a}"]["value"] - crit[f"D3_gain_{a}"]["value"])
                     for a in dt.ADIGEN_ARMS)
        check(resid < 1e-12 and add_up < 1e-12,
              f"task {task}: net = D4's quantity + D3's quantity (residual {max(resid, add_up):.1e})")
        blk = res["dose"]["thinned"] if task == "A" else res["screens"]["all"]["k"][str(dt.DECISION_K)]
        ref = blk["policies"][dt.REFERENCE]["captured_share"]
        check(abs(ref["value"] - 1) < 1e-12 and (ref["se"] or 0) < 1e-12,
              f"task {task}: the reference's captured share is exactly 1")
        if task == "A":
            check(crit["D0_power_gate"]["pass"] and crit["D1_naive_loses"]["pass"],
                  f"task A: the planted thinning costs the pooled real mean value (z = "
                  f"{dt.fmt_z(crit['D0_power_gate']['z'])}) and the line-blind generator too "
                  f"(z = {dt.fmt_z(crit['D1_naive_loses']['z'])})")
            s = res["dose"]["thinned"]["policies"]
            check(s[dt.pkey("stratified_real", 1.0)]["gain_over_random"]["value"]
                  > s[dt.pkey("pooled_real", 1.0)]["gain_over_random"]["value"],
                  "task A: at gamma = 1 the stratified real mean is worth more than the pooled one")
            d = res["dose"]["thinned"]["descriptive"]
            check(d[dt.pkey("pooled_real", 1.0)]["dose_rank_shift_as_predicted"] > 0.4
                  and abs(d[dt.pkey("stratified_real", 1.0)]["dose_rank_shift_as_predicted"]) < 0.3,
                  f"task A: the pooled mean's dose moves as predicted "
                  f"({d[dt.pkey('pooled_real', 1.0)]['dose_rank_shift_as_predicted']:+.2f} ranks), the stratified mean's "
                  f"does not ({d[dt.pkey('stratified_real', 1.0)]['dose_rank_shift_as_predicted']:+.2f})")
            u = res["dose"]["unthinned"]["policies"]
            check(abs(u[dt.pkey("pooled_real", 0.0)]["gain_over_random"]["value"]
                      - u[dt.pkey("pooled_real", 1.0)]["gain_over_random"]["value"]) < 1e-12,
                  "task A: on the unthinned compounds the real-data policies are the same at both gammas")
            check(set(pc["value"]) == set(scores["A"]) | {dt.RANDOM} and
                  all(v.shape == (des.comp.size,) for v in pc["value"].values()),
                  "the per-compound values cover every policy and compound")
        else:
            check(set(res["screens"]) == {"all", "thinned", "unthinned"} and
                  res["criteria"]["judged_on"] == {"screen": "all", "k": dt.DECISION_K, "control_screen": "unthinned",
                                                   "control_k": DECISION_K_CONTROL},
                  "task B: three screens; judged on the declared screen and k")
    # D2 failing turns D3 into a reported quantity
    # (identical arrays at both gammas give an exactly zero loss, hence no z)
    lost = x + 0.5 + 0.05 * rng.normal(size=x.size)
    V = {dt.pkey(a, g): dt.Est.mean(lost if a == "naive" and g == 0 else x)
         for a in dt.GEN_ARMS + ("pooled_real",) for g in (0.0, 1.0)}
    j = dt.judge(V, None, 1.0)
    check(j["task_testable"] is False and j["D1_naive_loses"]["pass"] is None
          and j["D1_naive_loses"]["would_pass"] is True and j["D3_gain_dr"]["pass"] is None
          and j["D3_testable"] is False and isinstance(j["D4_no_cost_dr"]["pass"], bool),
          "judge: when D0 does not pass, D1-D3 are descriptive (D4 is still judged)")
    V[dt.pkey("pooled_real", 0.0)] = dt.Est.mean(lost)
    j = dt.judge(V, None, 1.0)
    check(j["task_testable"] is True and j["D1_naive_loses"]["pass"] is True
          and j["D2_conditional_loses_testability"]["pass"] is False and j["D3_testable"] is False
          and j["D3_gain_dr"]["pass"] is None and "would_pass" in j["D3_gain_dr"],
          "judge: when D2 does not pass, D3 is reported and not judged")
    r = rng.normal(size=40)
    V = {dt.REFERENCE: dt.Est.mean(r + 0.01), dt.RANDOM: dt.Est.mean(r + 0.3 * rng.normal(size=40)),
         "x|g0": dt.Est.mean(r + 0.2)}
    t = dt.policy_table(V, {k: [v.value] for k, v in V.items()})
    check(t["captured_share_has_error"] is False and t["policies"]["x|g0"]["captured_share"]["se"] is None
          and t["policies"]["x|g0"]["captured_share"].get("unstable_denominator") is True,
          "policy_table: no error on a captured share whose denominator is within 3 SE of zero")


# ---------------------------------------------------------------------------
# the real data
# ---------------------------------------------------------------------------
def run_real(args) -> None:
    cfg = apply_paths_args(config_from_args(args), args)
    verdict, vpath = dt.read_verdict(cfg, args.verdict)       # refuses another pool or population
    print(f"[check_decision] verdict {vpath}")
    w = dt.load_world(cfg, verdict)
    K, L = w["ak"].size, len(w["lines"])
    oz = w["oz"]
    frame = dt.build_frame(w)                                # the same frame the job builds
    real, eligible, des = frame["real"], frame["eligible"], frame["design"]
    err = frame["oracle_err"]
    check(err < 1e-3, f"frame: the all-wells effect at each arm's own line mix = the stored oracle (max |diff| {err:.1e})")
    if len(w["kept"]) != 2 or min(w["kept"]) != 0.0:
        check(False, f"tiers: need gamma = 0 and one gamma > 0, found {sorted(w['kept'])}")
        return
    one = (real["holdout"]["cnt"] == 1).all(axis=1)
    cd = oz["all/contrast_dir"].astype(np.float64)
    d = np.abs(real["holdout"]["contrast"][one] - cd[one])
    check(one.sum() > 1000 and np.isfinite(d).all() and d.max() < 1e-2,
          f"frame: the holdout G2 - G1 contrast = the oracle's contrast_dir on the {int(one.sum()):,} arms with one "
          f"holdout well per line (max |diff| {d.max():.1e}; the two differ only in how the vehicles are pooled)")
    full = (real["all_wells"]["cnt"] > 0).all(axis=1)
    for g in sorted(w["kept"]):
        k = real[f"kept|g{g:g}"]["cnt"]
        check(bool(((k > 0) == (real["train"]["cnt"] > 0)).all()),
              f"tiers: gamma = {g:g} keeps a train well in every (arm, line) cell that had one")
        un = ~np.isin(w["comp_t"], np.asarray(w["scored"])) | w["ctl"]
        check(bool((w["kept"][g][un] == w["in_tr"][un]).all()),
              f"tiers: gamma = {g:g} leaves the vehicles and the unthinned compounds' train rows alone")
        check(not (w["kept"][g] & w["in_h"]).any(), f"tiers: gamma = {g:g} trains on no holdout row")
    n_th, n_un = int(des.thinned.sum()), int((~des.thinned).sum())
    check(min(n_th, n_un) >= 2 * DECISION_K_CONTROL and set(des.comp[des.thinned].tolist()) <= set(w["scored"]),
          f"design: {int(full.sum()):,} arms have a well in every line, {int(eligible.sum()):,} are eligible; "
          f"{n_th} thinned and {n_un} unthinned compounds enter")
    check(bool(np.isfinite(real["holdout"]["eq"][eligible]).all()), "design: every eligible arm has a holdout effect")
    s_b = dt.gene_axis(w["symbols"])
    check(abs(float((s_b ** 2).sum()) - 1) < 1e-12 and int((s_b != 0).sum()) == len(PROLIFERATION_GENES),
          f"axis: task B's {len(PROLIFERATION_GENES)} genes are landmarks and the axis is a unit vector")
    S = dt.compound_axes(real["holdout"]["eq"], des)
    arms = des.arm[des.valid]
    check(np.abs((S[arms] ** 2).sum(axis=1) - 1).max() < 1e-9, "axis: task A's axis is a unit vector on every design arm")
    shift = dt.g2_share_shift(w, real, des, sorted(w["kept"])[-1])
    g0, g1 = (shift[f"g{g:g}"] for g in sorted(w["kept"]))
    check(abs(g0["low"]) < 0.02 and abs(g0["high"]) < 0.02 and g1["low"] * g1["high"] < 0
          and min(abs(g1["low"]), abs(g1["high"])) > 0.05,
          f"tiers: the kept G2 share moves in opposite directions in the two dose halves at gamma > 0 "
          f"(low {g1['low']:+.3f}, high {g1['high']:+.3f}) and not at gamma = 0 ({g0['low']:+.3f}, {g0['high']:+.3f})")
    # one run's generated rows, grouped by this module = the tau_gen its scoring job stored
    ck = next(k for k in sorted(verdict["cells"]) if k.split("|")[0] in dt.GEN_ARMS)
    npz = verdict["cells"][ck]["npz"]
    rid, row_mean = dt.load_run_rows(npz, w)
    e = dt.effects_from_rows(row_mean, w["arm_of_row"][rid], w["line_t"][rid], w["ctl"][rid], K, L)
    z = np.load(npz, allow_pickle=False)
    with open(npz[:-len("_tau.npz")] + ".json") as fh:
        anchors = bool(json.load(fh)["generator"]["gen_anchors"])
    mu0 = z["all/mu0_gen"].astype(np.float64) - (0.0 if anchors else z["all/mu0"].astype(np.float64))
    stored = z["all/tau_gen"].astype(np.float64) - (0.0 if anchors else mu0)
    err = float(np.nanmax(np.abs(e["pooled"] - stored)))
    check(err < 1e-3 and bool((e["cnt"] == real["all_wells"]["cnt"]).all()),
          f"join: {ck}'s generated rows grouped by arm = its stored tau_gen (max |diff| {err:.1e}); its cells are the table's")
    check(bool(np.isfinite(e["eq"][full]).all()), f"join: {ck} has an equal-weight effect on every arm with all {L} lines")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verdict", default=None)
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    print("[check_decision] synthetic")
    run_synthetic()
    if args.data_dir:
        print("[check_decision] real data: " + args.data_dir)
        run_real(args)
    else:
        print("[check_decision] no --data_dir: the real-data checks were NOT run")
    if FAILS:
        print(f"\n[check_decision] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS))
        sys.exit(1)
    print("\n[check_decision] all checks passed")


if __name__ == "__main__":
    main()
