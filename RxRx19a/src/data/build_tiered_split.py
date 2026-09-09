#!/usr/bin/env python3
"""Build a tiered split dir, drawn in three independent layers:
  1. RESERVE — k wells per scored arm, uniform, before everything: the OOS truth. Never trained, identical across instances of one seed.
  2. SPLIT — well-grouped, stratified 80/20 train/holdout on the rest.
  3. EXPERIMENT (confounding drops) — thin the train side of scored arms (trace=1 well, thin=2, full=all) with retention pi ~ exp(-gamma * z_position) clipped to [pmin, 1]; alpha = 1/pi on retained rows. gamma=0 = uniform.

Scored arms = the 8-compound panel tier (trace = top dose + rotated mid; CQ/Baf thin-only) + --n_tier_compounds random compounds, same pattern.

  python -m src.data.build_tiered_split --plan      # power table, no writes
  python -m src.data.build_tiered_split --gamma 0   # build instance
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spec import default_config  # noqa: E402

FEAT_CACHE = "runs/eval_artifacts/feat_cache"

# The settled 10-compound panel (justified-panel-design). The 8 standard-grid  compounds are the panel test tier; CQ/Baf are thin-only.
D8 = [-2.523, -2.0, -1.523, -1.0, -0.523, 0.0, 0.477, 1.0]
PANEL_TEST = [
    "Remdesivir (GS-5734)", "GS-441524", "Aloxistatin", "Camostat",
    "Oseltamivir carboxylate", "Migalastat", "Tenofovir Disoproxil Fumarate",
    "methylprednisolone-sodium-succinate",
]
PANEL_THIN_ONLY = {
    "Chloroquine": [0.477, -0.523],
    "Bafilomycin A1": [-1.0, -2.523],
}
ROT_MIDS = [0.0, -0.523, -1.0, -1.523]   # interpolation-trace rotation pool


def _round_dose(x):
    return round(float(x), 3)


def _calibrate_pi(q: np.ndarray, m: float, pmin: float) -> np.ndarray:
    """pi_i proportional to q_i with sum(pi) = m, clipped to [pmin, 1]."""
    pi = np.full(q.shape, np.nan)
    free = np.ones(q.shape, dtype=bool)
    for _ in range(50):
        clipped_mass = np.where(~free, pi, 0.0).sum()
        rem = m - clipped_mass
        if rem <= 0 or not free.any():
            break
        scale = rem / q[free].sum()
        cand = q * scale
        newly = free & ((cand < pmin) | (cand > 1.0))
        pi = np.where(free, np.clip(cand, pmin, 1.0), pi)
        if not newly.any():
            break
        free = free & ~newly
    return np.clip(pi, pmin, 1.0)


def rotation(doses: list[float], idx: int) -> dict:
    """Deterministic trace/thin assignment for one compound.
    trace = top dose + one mid (rotated by compound index);
    thin = 2 of the remaining doses (rotated)."""
    top = max(doses)
    mids = [d for d in ROT_MIDS if d in doses and d != top]
    if not mids:   # non-standard grid: interior doses, highest first
        mids = sorted([d for d in doses if d != top], reverse=True)[1:-1] or \
               sorted([d for d in doses if d != top], reverse=True)
    mid = mids[idx % len(mids)]
    rest = sorted([d for d in doses if d not in (top, mid)], reverse=True)
    thin = [rest[(idx + j) % len(rest)] for j in range(2)]
    if thin[0] == thin[1]:
        thin[1] = rest[(idx + 2) % len(rest)]
    return {"trace": [top, mid], "thin": thin}

def main():
    ap = argparse.ArgumentParser(description=__doc__,  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k_reserve", type=int, default=2,  help="Wells reserved per scored arm for the truth (fixed across arms).")
    ap.add_argument("--gamma", type=float, default=0.0,  help="Plate-selection strength in exp(-gamma * z).")
    ap.add_argument("--pmin", type=float, default=0.05,  help="Positivity floor on per-well retention probability.")
    ap.add_argument("--keep_trace", type=int, default=1)
    ap.add_argument("--keep_thin", type=int, default=2)
    ap.add_argument("--n_tier_compounds", type=int, default=50)
    ap.add_argument("--holdout_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=None, help="Defaults to cfg.seed.")
    ap.add_argument("--plan", action="store_true",   help="Print expected induced bias per gamma; write nothing.")
    ap.add_argument("--plan_gammas", type=float, nargs="*",  default=[0.0, 2.0, 4.0, 8.0, 12.0])
    ap.add_argument("--out_dir", default=None,   help="Default: data/nuisances_tier_k<k>_pg<gamma>_s<seed>")
    args = ap.parse_args()

    cfg = default_config()
    seed = cfg.seed if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    # load in basics
    ds = load_from_disk(cfg.paths.tabular_dataset_dir)
    n_total = len(ds)
    ctyp = np.array([str(x) for x in ds["cell_type"]])
    dis = np.array([str(x) for x in ds["disease_condition"]])
    expt = np.array([str(x) for x in ds["experiment"]])
    plate = np.asarray(ds["plate"], dtype=np.int64)
    well_id = np.array([str(x) for x in ds["well_id"]])
    comp = np.asarray(ds["compound_idx"], dtype=np.int64)
    isctl = np.asarray(ds["is_control"], dtype=np.int64) == 1
    lx = np.array([_round_dose(x) for x in ds["log10_conc"]], dtype=np.float64)
    wname = np.array([str(x) for x in ds["well"]])

    # get masks
    pop_mask = (dis == str(cfg.population.disease_condition)) & (ctyp == str(cfg.population.cell_type))
    print(f"[tier] population {cfg.population.disease_condition} ∩ {cfg.population.cell_type}: {int(pop_mask.sum()):,} rows")

    # --- infection-axis Y from the cached domain embeddings. build a quick outcome infection score for incoporating confounding against this ------------------
    meta = json.load(open(os.path.join(FEAT_CACHE, "meta.json")))
    assert meta["n_rows"] == n_total, "feat_cache rows != dataset rows"
    phi = np.load(os.path.join(FEAT_CACHE, "domain.npy"), mmap_mode="r")

    Y = np.full(n_total, np.nan)
    G_by_expt = {}
    for e in ("HRCE-1", "HRCE-2"):
        em = expt == e
        mock = em & (dis == "Mock")
        uinf = em & (dis == str(cfg.population.disease_condition)) & isctl
        mu_m = np.asarray(phi[np.where(mock)[0]]).mean(0)
        mu_u = np.asarray(phi[np.where(uinf)[0]]).mean(0)
        u = mu_u - mu_m
        G = float(np.linalg.norm(u))
        u = u / G
        rows = np.where(em)[0]
        for s in range(0, rows.size, 8192):
            r = rows[s:s + 8192]
            Y[r] = (np.asarray(phi[r]) - mu_m) @ u
        G_by_expt[e] = G

    # wY holds on infection score per well.
    df_rows = np.where(pop_mask | (dis == "Mock"))[0]
    wY = defaultdict(list)
    for r in df_rows:
        wY[well_id[r]].append(Y[r])
    wY = {k: float(np.mean(v)) for k, v in wY.items()}

    # per- expiemrent vehicle baslines. with this, gets the zero point for ATEs based on y's above.
    uinf_mask = pop_mask & isctl
    veh_by_expt = {}
    for e in ("HRCE-1", "HRCE-2"):
        wells = np.unique(well_id[uinf_mask & (expt == e)])
        veh_by_expt[e] = float(np.mean([wY[w] for w in wells]))

    # for every well, where does the well sit on the plate? map it to get an experiment, plate pairing.
    _wr = {}
    for r in df_rows:
        _wr.setdefault(well_id[r], r)
    row_of = {w: wname[r][:-2] for w, r in _wr.items()}
    col_of = {w: wname[r][-2:] for w, r in _wr.items()}
    plate_of = {w: (expt[r], int(plate[r])) for w, r in _wr.items()}

    pos_pred = {}
    for e in ("HRCE-1", "HRCE-2"):
        vw = sorted(set(well_id[uinf_mask & (expt == e)]))
        yv = np.array([wY[w] for w in vw])
        plates_v = np.array([plate_of[w][1] for w in vw])
        rows_v = np.array([row_of[w] for w in vw])
        cols_v = np.array([col_of[w] for w in vw])
        mu = yv.mean()
        p_eff = {p: yv[plates_v == p].mean() - mu for p in set(plates_v)}
        resid = yv - np.array([p_eff[p] for p in plates_v]) - mu
        r_eff = {k: resid[rows_v == k].mean() for k in set(rows_v)}
        resid2 = resid - np.array([r_eff[k] for k in rows_v])
        c_eff = {k: resid2[cols_v == k].mean() for k in set(cols_v)}
        pos_pred[e] = (mu, p_eff, r_eff, c_eff)

    def well_pos_baseline(w):
        e, p = plate_of[w]
        mu, p_eff, r_eff, c_eff = pos_pred[e]
        return (mu + p_eff.get(p, 0.0) + r_eff.get(row_of[w], 0.0)
                + c_eff.get(col_of[w], 0.0))

    def sel_z(wells):
        b = np.array([well_pos_baseline(w) for w in wells])
        es = np.array([plate_of[w][0] for w in wells])
        z = np.zeros(len(wells))
        for e in set(es):
            m = es == e
            sd = b[m].std()
            z[m] = (b[m] - b[m].mean()) / (sd if sd > 1e-12 else 1.0)
        return z

    def arm_ate(wells, weights=None):
        wt = np.ones(len(wells)) if weights is None else np.asarray(weights, float)
        num = den = 0.0
        for e in set(plate_of[w][0] for w in wells):
            sel = [i for i, w in enumerate(wells) if plate_of[w][0] == e]
            ys = np.array([wY[wells[i]] for i in sel])
            ws = wt[sel]
            ate_e = (veh_by_expt[e] - np.sum(ws * ys) / ws.sum()) / G_by_expt[e]
            num += ws.sum() * ate_e
            den += ws.sum()
        return float(num / den)

    # --- arm bookkeeping -----------------------------------------------------
    # arm_wells: construct keys for (compound,dose) and referencing the estimands of interest
    # doses_of: consturct key for compound, checks that hte I1,I2 guardsl ook at
    vocab = json.load(open(os.path.join(cfg.paths.nuisance_dir, "compound_vocab.json")))
    inv_vocab = {v: k for k, v in vocab.items()}
    arm_wells = defaultdict(set)
    for r in np.where(pop_mask & ~isctl)[0]:
        arm_wells[(int(comp[r]), lx[r])].add(well_id[r])
    arm_wells = {k: sorted(v) for k, v in arm_wells.items()}
    doses_of = defaultdict(list)
    for (c, d) in arm_wells:
        doses_of[c].append(d)

    # panel tier. Decide per compound, which of its doses are full, trace (low positivity), or thin (more than trace, but still low positivity)
    panel_ci = {n: int(vocab[n]) for n in PANEL_TEST + list(PANEL_THIN_ONLY)}
    assign = {}   # compound_idx -> {"trace": [...], "thin": [...], "tier": str}
    for i, name in enumerate(sorted(PANEL_TEST)):
        ci = panel_ci[name]
        grid = sorted(doses_of[ci])
        assert grid == sorted(D8), f"{name}: grid {grid} != D8"
        rot = rotation(grid, i)
        assign[ci] = {**rot, "tier": "panel", "name": name}
    for name, thin in PANEL_THIN_ONLY.items():
        ci = panel_ci[name]
        for d in thin:
            assert d in doses_of[ci], (name, d)
        assign[ci] = {"trace": [], "thin": thin, "tier": "panel", "name": name}

    # random tier: eligible = >=6 doses, every arm >=5 wells
    min_arm_wells = args.k_reserve + 3
    eligible = [c for c, dd in doses_of.items() if c not in {panel_ci[n] for n in panel_ci} and len(dd) >= 6 and all(len(arm_wells[(c, d)]) >= min_arm_wells for d in dd)]
    tier_cs = sorted(rng.choice(sorted(eligible), args.n_tier_compounds,  replace=False).tolist())
    for i, ci in enumerate(tier_cs):
        assign[ci] = {**rotation(sorted(doses_of[ci]), i), "tier": "random",  "name": inv_vocab.get(ci, str(ci))}
    print(f"[tier] eligible non-panel compounds: {len(eligible)}; sampled {len(tier_cs)} (seed {seed})")

    scored_arms = [(c, d) for c, a in assign.items() for d in doses_of[c]]
    trace_arms = [(c, d) for c, a in assign.items() for d in a["trace"]]
    thin_arms = [(c, d) for c, a in assign.items() for d in a["thin"]]
    print(f"[tier] scored arms {len(scored_arms)} "
          f"(trace {len(trace_arms)}, thin {len(thin_arms)}, "
          f"full {len(scored_arms) - len(trace_arms) - len(thin_arms)})")

    # --- LAYER 1: the reserve / hold out / test set. Draw k wells per arm, uniformly. this is identical across all evals! ----
    res_rng = np.random.default_rng(seed + 10_000)
    reserve = {}
    for a in scored_arms:
        ws = arm_wells[a]
        assert len(ws) >= min_arm_wells, (a, len(ws))
        reserve[a] = sorted(res_rng.choice(ws, args.k_reserve, replace=False))
    reserve_wells = {w for ws in reserve.values() for w in ws}

    def exp_wells(a):
        return [w for w in arm_wells[a] if w not in reserve_wells]

    # --- PLAN MODE: expected bias per gamma vs truth, allows us to make an informed decision and understand confounding we are baking in -----
    manip = [(a, args.keep_trace) for a in trace_arms] + \
            [(a, args.keep_thin) for a in thin_arms]
    if args.plan:
        print(f"\n[plan] {len(manip)} manipulated arms (trace keep={args.keep_trace}, thin keep={args.keep_thin}), k_reserve={args.k_reserve}, pmin={args.pmin} {'gamma':>6} {'mean bias':>10} {'sd':>7} {'pooled z':>9} {'min pi':>7}")
        for g in args.plan_gammas:
            biases, pimins = [], []
            for (a, keep) in manip:
                wells = exp_wells(a)
                z = sel_z(wells)
                pi = _calibrate_pi(np.exp(-g * z), keep, args.pmin)
                biases.append(arm_ate(wells, weights=pi) - arm_ate(wells))
                pimins.append(pi.min())
            b = np.array(biases)
            se = b.std(ddof=1) / np.sqrt(len(b))
            print(f"{g:6.1f} {b.mean():10.4f} {b.std(ddof=1):7.4f} "
                  f"{b.mean()/max(se,1e-12):9.2f} {min(pimins):7.3f}")
        return

    # =========================================================================
    # BUILD based on plan.
    # =========================================================================
    gamma = args.gamma
    out_dir = args.out_dir or os.path.join(os.path.dirname(cfg.paths.nuisance_dir), f"nuisances_tier_k{args.k_reserve}_pg{gamma:g}_s{seed}")
    os.makedirs(out_dir, exist_ok=True)

    row_of_well = defaultdict(list)
    for r in np.where(pop_mask)[0]:
        row_of_well[well_id[r]].append(r)
    reserve_rows = np.concatenate([row_of_well[w] for w in sorted(reserve_wells)])
    print(f"[build] reserve: {len(reserve_wells)} wells / {reserve_rows.size} rows "
          f"locked out of training (k={args.k_reserve} x {len(scored_arms)} arms)")

    # --- LAYER 2: stratify split of non-reserve/test set into train and holdout. 
    # well-grouped by putting all 4 sites of well to move together, stratified on condition, compound, and dose.
    split_mask = pop_mask.copy()
    split_mask[reserve_rows] = False
    dose_key = np.where(isctl, -10.0, lx)
    strat = {}
    for r in np.where(split_mask)[0]:
        w = well_id[r]
        k = (int(comp[r]), float(dose_key[r]))
        if w in strat:
            assert strat[w] == k, f"well {w} spans strata"
        else:
            strat[w] = k
    by_stratum = defaultdict(list)
    for w, k in strat.items():
        by_stratum[k].append(w)

    train_wells, hold_wells = [], []
    for k in sorted(by_stratum):
        ws = sorted(by_stratum[k])
        rng.shuffle(ws)
        n = len(ws)
        n_hold = int(round(args.holdout_frac * n)) if n >= 2 else 0
        hold_wells += ws[:n_hold]
        train_wells += ws[n_hold:]
    train_idx = np.sort(np.concatenate([row_of_well[w] for w in train_wells]))
    holdout_idx = np.sort(np.concatenate([row_of_well[w] for w in hold_wells]))
    train_well_set = set(train_wells)
    print(f"[build] split: train {train_idx.size:,} rows / {len(train_wells):,} wells; "
          f"holdout {holdout_idx.size:,} rows / {len(hold_wells):,} wells")

    # --- LAYER 3: for each trace/thin arm in training set, -------------
    # 1. compute retention probabilities based on confounding.  
    # 2. draw survivors
    # 3. record dr weights
    # 4. drop rows.
    pi_of_well, kept_of_arm, diag_arms = {}, {}, []
    for (a, keep) in manip:
        wells = [w for w in exp_wells(a) if w in train_well_set]
        z = sel_z(wells)
        pi = _calibrate_pi(np.exp(-gamma * z), keep, args.pmin)
        for _ in range(200):
            k_mask = rng.random(len(wells)) < pi
            if 1 <= k_mask.sum() <= keep + 1:
                break
        kept = [w for w, km in zip(wells, k_mask) if km]
        kept_of_arm[a] = kept
        for w, p in zip(wells, pi):
            pi_of_well[w] = float(p)

        truth_wells = reserve[a]
        truth = arm_ate(truth_wells)
        # SE from well-level spread within the reserve
        ys = np.array([wY[w] for w in truth_wells])
        se = float(ys.std(ddof=1) / np.sqrt(len(ys))
                   / np.mean([G_by_expt[plate_of[w][0]] for w in truth_wells]))
        mc_n, mc_i = [], []
        mrng = np.random.default_rng(hash(a) % 2**32)
        for _ in range(200):
            km = mrng.random(len(wells)) < pi
            if km.sum() < 1:
                continue
            kw = [w for w, kk in zip(wells, km) if kk]
            mc_n.append(arm_ate(kw))
            mc_i.append(arm_ate(kw, weights=[1.0 / p for p, kk in zip(pi, km) if kk]))
        full_exp = arm_ate(wells)
        diag_arms.append({
            "compound": assign[a[0]]["name"], "dose": a[1],
            "tier": assign[a[0]]["tier"],
            "kind": "trace" if a in trace_arms else "thin",
            "n_wells_exp": len(wells), "n_wells_kept": len(kept),
            "pi": [round(float(p), 4) for p in pi],
            "truth_reserve": round(truth, 4), "truth_se": round(se, 4),
            "ate_full_exp": round(full_exp, 4),
            "bias_naive": round(arm_ate(kept) - full_exp, 4) if kept else None,
            "mc_bias_naive": round(float(np.mean(mc_n)) - full_exp, 4),
            "mc_bias_ipw": round(float(np.mean(mc_i)) - full_exp, 4),
        })

    bmn = np.array([a["mc_bias_naive"] for a in diag_arms])
    bmi = np.array([a["mc_bias_ipw"] for a in diag_arms])
    n = len(diag_arms)
    for lab, b in (("MC mean naive", bmn), ("MC mean ipw  ", bmi)):
        se = b.std(ddof=1) / np.sqrt(n)
        print(f"[bias] {lab}: mean {b.mean():+.4f}  sd {b.std(ddof=1):.4f}  "
              f"z {b.mean()/max(se,1e-12):+.2f}   ({n} arms)")

    # --- final train_idx and weights ----------------------------------------
    drop_rows = []
    for (a, _) in manip:
        kept = set(kept_of_arm[a])
        for w in arm_wells[a]:
            if w in train_well_set and w not in kept and w not in reserve_wells:
                drop_rows += row_of_well[w]
    drop = np.zeros(n_total, dtype=bool)
    if drop_rows:
        drop[np.array(drop_rows, dtype=np.int64)] = True
    final_train = train_idx[~drop[train_idx]]
    w_arr = np.ones(final_train.size, dtype=np.float32)
    kept_row_pi = {}
    for (a, _) in manip:
        for w in kept_of_arm[a]:
            for r in row_of_well[w]:
                kept_row_pi[r] = pi_of_well[w]
    for i, r in enumerate(final_train):
        if int(r) in kept_row_pi:
            w_arr[i] = 1.0 / kept_row_pi[int(r)]
    n_up = int((w_arr != 1.0).sum())
    print(f"[build] final train {final_train.size:,} rows "
          f"({train_idx.size - final_train.size:,} thinned away); "
          f"{n_up} rows carry design weights, w max {w_arr.max():.1f}, "
          f"ESS/n {float((w_arr.sum()**2)/(np.square(w_arr).sum())/w_arr.size):.3f}")

    # --- guards --------------------------------------------------------------
    for ci, a in assign.items():
        tr_doses = sorted(set(lx[final_train][comp[final_train] == ci]))
        assert len(tr_doses) >= 4, f"I1 FAIL {a['name']}: {tr_doses}"
        full_doses = [d for d in doses_of[ci]  if d not in a["trace"] and d not in a["thin"]]
        assert len(full_doses) >= 2, f"I1b FAIL {a['name']}: {full_doses}"
    tier_set = set(assign)
    for d in sorted({d for (_, d) in trace_arms}):
        others = set(comp[final_train][(lx[final_train] == d)  & ~np.isin(comp[final_train], list(tier_set))])
        assert len(others) >= 2, f"I2 FAIL dose {d}: {len(others)} other compounds"
    print("[build] guards I1 (>=4 train doses) I1b (>=2 full doses) I2 (trace dose values trained elsewhere) PASS")

    # --- write ----------------------------------------------------------------
    payload = {
        "holdout_frac": args.holdout_frac, "seed": seed, "n_total": n_total,
        "n_strata": len(by_stratum),
        "stratify_by": ["compound_idx", "dose"],
        "grouped_by": "well_id",
        "population": {"disease_condition": str(cfg.population.disease_condition),  "cell_type": str(cfg.population.cell_type)},
        "n_train": int(final_train.size), "n_holdout": int(holdout_idx.size),
        "train_idx": final_train.tolist(), "holdout_idx": holdout_idx.tolist(),
        "rarity": {"active": False, "tag": ""},
        "tier": {
            "k_reserve": args.k_reserve, "gamma": gamma, "pmin": args.pmin,
            "keep_trace": args.keep_trace, "keep_thin": args.keep_thin,
            "confounder": "position (plate+row+col vehicle baseline)",
            "trace_arms": [[c, d] for (c, d) in trace_arms],
            "thin_arms": [[c, d] for (c, d) in thin_arms],
            "tier_compounds": {str(c): assign[c]["name"] for c in assign},
        },
    }
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(payload, f)
    np.savez(os.path.join(out_dir, "dr_weights_design.npz"), row_id=final_train.astype(np.int64), w=w_arr)
    with open(os.path.join(out_dir, "reserve.json"), "w") as f:
        json.dump({
            "k_reserve": args.k_reserve,
            "G_by_expt": G_by_expt, "veh_by_expt": veh_by_expt,
            "arms": [{
                "compound_idx": c, "compound": assign[c]["name"], "dose": d,
                "tier": assign[c]["tier"],
                "kind": ("trace" if (c, d) in set(trace_arms) else "thin" if (c, d) in set(thin_arms) else "full"),
                "reserve_wells": reserve[(c, d)],
                "truth": round(arm_ate(reserve[(c, d)]), 4),
            } for (c, d) in scored_arms]}, f, indent=1)
    for fn in ("nuisance_meta.json", "compound_vocab.json", "covariate_encoder.json"):
        shutil.copy2(os.path.join(cfg.paths.nuisance_dir, fn), os.path.join(out_dir, fn))
    with open(os.path.join(out_dir, "tier_meta.json"), "w") as f:
        json.dump({"gamma": gamma, "seed": seed, "k_reserve": args.k_reserve,
                   "pmin": args.pmin, "G_by_expt": G_by_expt,
                   "pooled_bias": {
                       "n_arms": n,
                       "mc_naive_mean": float(bmn.mean()), "mc_naive_sd": float(bmn.std(ddof=1)),
                       "mc_ipw_mean": float(bmi.mean()), "mc_ipw_sd": float(bmi.std(ddof=1))
                    },
                   "arms": diag_arms}, f, indent=2)
    
    # nu = the design action distribution (unthinned train pool), 
    # for learned alpha: fit_urr --nu_rows / export_urr_weights --mode counts target it.
    from src.data.build_nu_rows import build_nu_rows
    build_nu_rows(out_dir)
    print(f"[build] wrote {out_dir}")


if __name__ == "__main__":
    main()
