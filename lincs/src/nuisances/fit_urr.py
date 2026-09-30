"""URR Riesz fit

    L(alpha) =  E_{(X,A) ~ P}      [ alpha(X, A )^2 ]    -  2 E_{X ~ P_X, At ~ nu} [ alpha(X, At) ]

Copied from RxRx19a/src/nuisances/fit_urr.py (the URR loss, cross-fitting and
the ESS / tail gate), then adapted (IMPLEMENT.md §3.5; P2, P5):
  - no `infected` bit, no infected==1 train restriction, no --cell_type, no
    experiment / cell_type reads: the split is already the population;
  - X = alpha_cov_fields(cfg), the adjustment set shared with export and the
    generator (v1: empty); no --cov_blocks override, and --nu_rows does not
    empty X (step C fits `--adjustment_set syn_c --nu_rows ...`);
  - arms are (compound_idx, dose_level) (P2) for the common support and nu;
  - cross-fit folds are stratified on that arm (P5), and on the X stratum:
    each arm's rows alternate between the folds, so a row is scored by a net
    that saw its arm. Folds cover the whole nu pool (train rows and, in a
    thinning instance, the thinned-away rows, stratified separately), so each
    fold's P and nu are half-samples of the same design;
  - the val rows are drawn from the fold's pool and held out of BOTH legs, and
    every (nu arm, X stratum) of the product leg must have a factual fit row
    (--max_nu_gap_cells): with ~1 row per arm per fold, a nu arm without one
    has P_fit = 0 and alpha = nu / P runs to +inf;
  - control ids come from `__control__`; nu never contains controls;
  - --nu_rows (default nu_rows.npy, relative to the split dir) replaces RxRx's
    --nu_source; a thinning instance refuses to run without it (§3.8.1);
  - outputs are written together at the end and share a run_id; the previous
    <prefix>_meta.json is removed at the start, so export never picks up a
    crashed run's mix of old and new nets.

v1 known answer (§3.10): C is empty, so alpha(a) = nu(a) / P(a) = 1/(1 - pi0)
on treated rows (pi0 = the DMSO share of the fit rows, ~1.066) and ~0 on DMSO.

Outputs (split dir): <prefix>_fold{0,1}.pt (+ .spec.json), <prefix>_meta.json,
<prefix>_folds.npy (folds depend on X, so each prefix keeps its own).
nuisance_meta.json is build_dataset's; this only checks it.

Run from lincs/ (CPU job):
    python -m src.nuisances.fit_urr [--nuisance_dir ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import (  # noqa: E402
    CONTROL_COMPOUND_NAME, covariate_blocks, load_covariate_encoder)
from src.data.splits import arm_keys, load_splits, resolve_split_file  # noqa: E402
from src.nuisances.alpha_net import AlphaNet  # noqa: E402
from src.nuisances.knn_dr import ess, tail_index  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, alpha_cov_fields, apply_paths_args, config_from_args,
    format_role_summary, invariance_env_fields)

CONST_BASELINE = -1.0

# Divergence checks on loss
DIVERGENCE_L = 1e3
DIVERGENCE_A = 1e3

def folds_filename(prefix: str) -> str:
    return f"{prefix}_folds.npy"


def urr_loss(net, cov, comp, lx, ic, comp_t, lx_t, ic_t, ridge: float = 0.0, shrink: float = 0.0):
    """L = E[alpha(X,A)^2] - 2 E[alpha(X,At)],   At ~ nu drawn independently of X.

    `cov` is the row's covariates X. (comp, lx, ic) is its FACTUAL action A.
    (comp_t, lx_t, ic_t) are (B, M) actions resampled from nu.
    """
    B, M = comp_t.shape
    a_obs = net(cov, comp, lx, ic).squeeze(-1)                      # (B,)   at (X_i, A_i)

    cov_e = cov.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
    a_prd = net(cov_e, comp_t.reshape(-1), lx_t.reshape(-1),  ic_t.reshape(-1)).squeeze(-1).view(B, M)          # (B,M)  at (X_i, At)

    sq = (a_obs ** 2).mean()
    # shrink anchors alpha at the null (==1): with few rows per discrete action the unpenalised minimiser can memorise count noise.
    pen = shrink * ((a_obs - 1.0) ** 2).mean() if shrink else 0.0
    return sq - 2.0 * a_prd.mean() + ridge * sq + pen, a_obs


def stratified_folds(pool_rows: np.ndarray, key: np.ndarray, seed: int, n_total: int) -> np.ndarray:
    """(n_total,) int8 cross-fit fold per row of `pool_rows`, -1 elsewhere (P5).

    Rows are grouped by `key` (arm | X stratum | train or not); each group is
    shuffled and alternates 0/1 from a random start. So every arm with >= 2
    train rows in a stratum has a train row in each fold (random halves leave
    35% of treated train rows without their arm in the other fold).
    """
    rng = np.random.default_rng(seed)
    folds = np.full(n_total, -1, dtype=np.int8)
    pool = np.asarray(pool_rows, dtype=np.int64)
    keys, inv = np.unique(key[pool], return_inverse=True)
    order = np.argsort(inv, kind="stable")
    ends = np.cumsum(np.bincount(inv, minlength=keys.size))
    start = 0
    for e in ends:
        rows = pool[order[start:e]]
        rows = rows[rng.permutation(rows.size)]
        first = int(rng.integers(0, 2))
        folds[rows] = (np.arange(rows.size) + first) % 2
        start = e
    return folds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="AlphaNet is tiny: a CPU job by default (IMPLEMENT.md §3.13).")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_perm", type=int, default=4, help="Marginal-draw actions per row (>=2).")
    p.add_argument("--ridge", type=float, default=0.0)
    p.add_argument("--shrink_to_one", type=float, default=0.0, help="Penalty weight on (alpha-1)^2: defaults alpha to the null unless the data insists.")
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target_support", default="common", choices=("common", "all"), help="Support of nu: 'common' = arms with train rows in every X stratum; 'all' = every treated arm of the population (step C).")
    p.add_argument("--out_prefix", default="alpha_urr",  help="Basename for the saved nets/meta (default alpha_urr -> alpha_urr_fold{0,1}.pt + alpha_urr_meta.json).")
    p.add_argument("--nu_rows", default="nu_rows.npy", help="The .npy of row ids defining nu (the UNTHINNED train pool build_tiered_split writes), relative to the split dir. 'none' = this fold's fit rows (refused on a thinning instance).")
    p.add_argument("--max_nu_gap_cells", type=int, default=0, help="Refuse a fold whose product leg has more (nu arm, X stratum) cells than this without a factual fit row (alpha is unbounded there).")
    p.add_argument("--positive", type=int, default=1, help="Softplus head: alpha >= 0, as a density ratio must be.")
    p.add_argument("--min_nu_rows", type=int, default=1000, help="Refuse a nu support with fewer rows (lower it only for --limit smoke builds).")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    if args.n_perm < 2:
        raise ValueError("n_perm >= 2")

    cfg = apply_paths_args(config_from_args(args), args)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)

    nz = cfg.paths.nuisance_dir
    run_id = uuid.uuid4().hex[:12]
    meta_path = os.path.join(nz, f"{args.out_prefix}_meta.json")
    # n_compounds and cov_dim come from what build_dataset wrote
    with open(os.path.join(nz, "compound_vocab.json")) as f:
        vocab = json.load(f)
    n_compounds = len(vocab)
    with open(os.path.join(nz, "nuisance_meta.json")) as f:
        nmeta = json.load(f)

    # load arrays
    meta = load_from_disk(cfg.paths.tabular_dataset_dir).select_columns(
        ["cov_vec", "compound_idx", "log10_conc", "is_control", "dose_level", "pert_id"])
    cov_all = np.asarray(meta["cov_vec"], dtype=np.float32)
    comp_all = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx_all = np.asarray(meta["log10_conc"], dtype=np.float32)
    ic_all = np.asarray(meta["is_control"], dtype=np.float32)
    arm_all = arm_keys(comp_all, np.asarray(meta["dose_level"], dtype=np.float64), ic_all)
    cov_dim = cov_all.shape[1]
    if (nmeta["n_compounds"], nmeta["cov_dim"]) != (n_compounds, cov_dim):
        raise ValueError(f"nuisance_meta.json {nmeta} disagrees with the vocab ({n_compounds}) / cov_vec "
                         f"({cov_dim}); rebuild with src.data.build_dataset")

    splits = load_splits(cfg)
    train_idx = splits["train_idx"]

    # ---------------------------------------------------------------------
    # derive X (column indices): the adjustment set shared with export and the generator
    enc = load_covariate_encoder(os.path.join(nz, "covariate_encoder.json"), cfg.covariates)
    blocks = covariate_blocks(enc)
    print(f"[urr] roles: {format_role_summary(cfg)}")
    keep = list(alpha_cov_fields(cfg))
    print(f"[urr] alpha(A, C, E): C={list(cfg.adjustment_set) or '(empty)'} "
          f"E={list(invariance_env_fields(cfg))} -> cov blocks "
          f"{keep or '(empty -- alpha depends on A only)'}")
    cov_idx = sorted(i for b in keep for i in blocks[b])

    # The population mask: controls always, compounds per --population_compounds.
    pop_mask_all = np.ones(len(comp_all), dtype=bool)
    if cfg.population.compounds:
        _want_ids = {int(v) for k, v in vocab.items() if k in set(cfg.population.compounds)}
        _ctl_ids = {int(vocab[CONTROL_COMPOUND_NAME])}
        pop_mask_all = np.isin(comp_all, list(_want_ids | _ctl_ids))
        print(f"[urr] nu restricted to the population: {len(_want_ids)} compounds, {int(pop_mask_all.sum()):,}/{len(pop_mask_all):,} rows eligible")

    # get support of the target intervention distribution nu, under the chosen X
    strata = np.array(["".join(map(str, r)) for r in (cov_all[:, cov_idx] > 0.5).astype(np.int8)], dtype=object)

    # Overlap is a property of the ACTION, the (compound, dose_level) arm: a compound can span every stratum while one of its arms sits in only one. Check support on the rows f is estimated from -- the train set
    per_stratum: dict[tuple, set] = {}
    for i in train_idx:
        if ic_all[i] >= 0.5 or not pop_mask_all[i]:
            continue                              # controls are the reference, not a target
        per_stratum.setdefault(strata[i], set()).add(arm_all[i])
    common = set.intersection(*per_stratum.values()) if per_stratum else set()
    _seen = set().union(*per_stratum.values()) if per_stratum else set()
    if args.target_support == "common":
        nu_arms = common
        print(f"[urr] arm-level common support: {len(nu_arms)}/{len(_seen)} (compound, dose_level) arms span all {len(per_stratum)} strata; dropped {len(_seen) - len(nu_arms)} for positivity")
    else:
        nu_arms = {a for a, isc, pm in zip(arm_all, ic_all, pop_mask_all) if isc < 0.5 and pm}
        print(f"[urr] target support = every treated arm of the population ({len(nu_arms)}); "
              f"{len(nu_arms) - len(common)} lack train rows in some stratum")
    nu_compounds = {int(a.split("|")[0]) for a in nu_arms}
    if not nu_compounds:
        raise RuntimeError("target support is empty -- no compound spans every stratum")
    nu_mask_all = np.isin(arm_all, sorted(nu_arms))

    # the nu pool: the unthinned design pool (nu_rows.npy), or the train rows ('none')
    use_nu_rows = args.nu_rows.lower() != "none"
    if not use_nu_rows and splits["tier"].get("active"):
        raise ValueError("a thinning instance needs --nu_rows: nu must see the unablated design (IMPLEMENT.md §3.8.1)")
    nu_rows_path = resolve_split_file(nz, args.nu_rows) if use_nu_rows else ""
    fold_pool = train_idx
    if nu_rows_path:
        if not os.path.isfile(nu_rows_path):
            raise FileNotFoundError(f"{nu_rows_path} missing; build_tiered_split writes it (or pass --nu_rows none)")
        fold_pool = np.load(nu_rows_path).astype(np.int64)
        if not np.isin(train_idx, fold_pool).all():
            raise ValueError(f"{nu_rows_path} does not contain every train row; it belongs to another split")
        fold_pool = np.sort(fold_pool[pop_mask_all[fold_pool]])

    # cross-fit folds over the pool, stratified on (arm, X stratum, train or not) (P5)
    in_train = np.zeros(len(comp_all), dtype=bool)
    in_train[train_idx] = True
    fold_key = arm_all + "|X=" + strata + np.where(in_train, "|train", "|nu").astype(object)
    folds = stratified_folds(fold_pool, fold_key, cfg.seed, len(comp_all))
    fold_tr = folds[train_idx].astype(np.int64)
    print(f"[urr] train={len(train_idx):,}  nu pool={len(fold_pool):,}  n_compounds={n_compounds}  "
          f"train rows per fold {np.bincount(fold_tr).tolist()} (stratified, seed {cfg.seed})")

    # check some debugs
    pi0 = float(ic_all[train_idx].mean())
    print(f"[urr] X = {keep}  -> cov cols {len(cov_idx)}/{cov_dim}, "
          f"{len(per_stratum)} strata")
    print(f"[urr] nu support = {len(nu_compounds)} compounds / {len(nu_arms)} arms "
          f"({args.target_support}); {nu_mask_all.mean():.1%} of rows carry one")
    print(f"[urr] head = {'softplus (alpha >= 0)' if args.positive else 'linear'}")
    print(f"[urr] loss: E[a(X,A)^2] - 2 E[a(X,At)],  At ~ nu  (quadratic at FACTUAL, "
          f"linear at PRODUCT)")
    print(f"[urr] DMSO share of train pi0 = {pi0:.4f}; with X empty and nu = the treated train "
          f"marginal, alpha = 1/(1-pi0) = {1 / (1 - pi0):.4f} on treated rows, 0 on DMSO")
    print(f"[urr] constant baseline L(a==1) = {CONST_BASELINE:+.4f}  <- must beat this\n")

    def T(a):
        return torch.tensor(a, device=dev)

    # inputs validated: from here on the previous fit is superseded
    if os.path.isfile(meta_path):
        os.remove(meta_path)
        print(f"[urr] removed the previous {os.path.basename(meta_path)}; export refuses {args.out_prefix} until this run finishes")

    # now CF.
    meta_out, nets_out = {}, {}
    for f_ in (0, 1):
        # cross-fit: fit on the pool rows of the OTHER fold; score the train rows of fold f_
        pool_f = fold_pool[folds[fold_pool] == 1 - f_]
        rng = np.random.default_rng(args.seed + f_)
        nv = max(int(args.val_frac * len(pool_f)), min(512, len(pool_f) // 4))
        val = np.zeros(len(comp_all), dtype=bool)
        val[rng.choice(pool_f, nv, replace=False)] = True
        # The val rows are held out of BOTH legs. With ~1 row per arm per fold, an arm whose
        # only fit row went to val would sit in nu with P_fit(a) = 0, and alpha = nu/P runs to +inf there.
        fit_rows = pool_f[in_train[pool_f]]
        vi, ti = fit_rows[val[fit_rows]], fit_rows[~val[fit_rows]]
        nu_f = pool_f[~val[pool_f]]
        ti_nu = nu_f[nu_mask_all[nu_f]]
        print(f"[urr] fold {f_}: fit={len(ti):,}  val={len(vi):,}  nu={len(ti_nu):,} rows"
              + (f" from {os.path.basename(nu_rows_path)} -- alpha targets that design action distribution"
                 if nu_rows_path else " (the fit rows)"))
        if len(ti_nu) < args.min_nu_rows:
            raise RuntimeError(f"only {len(ti_nu)} rows on the nu support -- too few (--min_nu_rows {args.min_nu_rows})")
        # positivity on the product support: every (nu arm, X stratum of the fit rows) needs a factual fit row
        arms_nu = set(arm_all[ti_nu].tolist())
        strata_ti = set(strata[ti].tolist())
        cells = {(a, x) for a, x in zip(arm_all[ti], strata[ti]) if a in arms_nu}
        n_gap = len(arms_nu) * len(strata_ti) - len(cells)
        n_gap_arms = len(arms_nu - set(arm_all[ti].tolist()))
        print(f"[urr] fold {f_}: product support {len(arms_nu):,} nu arms x {len(strata_ti)} X strata; "
              f"{n_gap} cells ({n_gap_arms} arms) have no factual fit row")
        if n_gap > args.max_nu_gap_cells:
            raise RuntimeError(
                f"fold {f_}: {n_gap} (nu arm, X stratum) cells have no factual fit row (> --max_nu_gap_cells "
                f"{args.max_nu_gap_cells}); alpha = nu/P is unbounded there. Shrink X or the target support, "
                f"or raise the limit only if the net is meant to share across those cells.")

        # setup model.
        net = AlphaNet(cov_dim=cov_dim, n_compounds=n_compounds, cov_idx=cov_idx, positive=bool(args.positive)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        brng = np.random.default_rng(args.seed + 100 + f_)

        # fixed val batch.
        vrng = np.random.default_rng(args.seed + 500 + f_)
        vj = ti_nu[vrng.integers(0, len(ti_nu), (len(vi), args.n_perm))]
        V = (T(cov_all[vi]), T(comp_all[vi]), T(lx_all[vi]), T(ic_all[vi]),
             T(comp_all[vj]), T(lx_all[vj]), T(ic_all[vj]))

        # train loop
        best, best_state, bad = float("inf"), None, 0
        for step in range(1, args.steps + 1):
            net.train()
            ix = ti[brng.integers(0, len(ti), args.batch_size)]

            # marginal draws: whole observed actions resampled from nu-support rows
            jx = ti_nu[brng.integers(0, len(ti_nu), (args.batch_size, args.n_perm))]
            loss, _ = urr_loss(
                net, T(cov_all[ix]), T(comp_all[ix]), T(lx_all[ix]), T(ic_all[ix]),
                T(comp_all[jx]), T(lx_all[jx]), T(ic_all[jx]),
                ridge=args.ridge, shrink=args.shrink_to_one,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            if step % args.eval_every == 0 or step == 1:
                net.eval()
                with torch.no_grad():
                    vl, va = urr_loss(net, *V)
                vl = float(vl)
                a = va.cpu().numpy()
                print(f"   step {step:6d}  L_val={vl:+.4f}  (beats const by "
                      f"{CONST_BASELINE - vl:+.4f})  mean(a)={a.mean():.3f} "
                      f"std={a.std():.3f} max={a.max():.2f}", flush=True)
                # Divergence check.
                if vl < -DIVERGENCE_L or a.max() > DIVERGENCE_A:
                    raise RuntimeError(
                        f"URR DIVERGED at step {step}: L_val={vl:.1f}, "
                        f"max(alpha)={a.max():.1f}, mean(alpha)={a.mean():.1f}.\n"
                        f"  alpha is running to +inf on target actions with no observed "
                        f"support -- the Int1 overlap assumption is violated for\n"
                        f"    X = {keep}   nu = {args.target_support} "
                        f"({len(nu_compounds)} compounds).\n"
                        f"  Shrink the conditioning set or the target "
                        f"support (--target_support common). Do NOT just add steps."
                    )
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= args.patience:
                        print(f"   early stop @ {step}")
                        break

        net.load_state_dict(best_state)
        nets_out[f_] = net

        # score the HELD-OUT fold, so out of sample computation of URR.
        score_rows = train_idx[fold_tr == f_]
        net.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(score_rows), 8192):
                r = score_rows[s : s + 8192]
                outs.append(net(T(cov_all[r]), T(comp_all[r]), T(lx_all[r]),
                                T(ic_all[r])).squeeze(-1).cpu().numpy())

        # some checks.
        a = np.concatenate(outs).astype(np.float64)
        is_c = ic_all[score_rows] >= 0.5
        pi0_f = float(ic_all[ti].mean())
        kh = tail_index(np.abs(a))
        print(f"   -> fold {f_}: L_val={best:+.4f}   alpha(held-out): mean={a.mean():.4f} "
              f"std={a.std():.4f} min={a.min():.3f} max={a.max():.2f} "
              f"ESS={100*ess(np.abs(a))/len(a):.1f}% tail_k={kh:.3f}   "
              f"treated mean={a[~is_c].mean():.4f} DMSO mean={a[is_c].mean():.4f} "
              f"(v1 answer {1 / (1 - pi0_f):.4f} / 0)\n")
        meta_out[f"fold{f_}"] = {
            "L_val": best, "const_baseline": CONST_BASELINE,
            "beats_constant_by": CONST_BASELINE - best,
            "alpha_mean": float(a.mean()), "alpha_std": float(a.std()),
            "alpha_min": float(a.min()), "alpha_max": float(a.max()),
            "tail_khat": kh, "ess_frac": float(ess(np.abs(a)) / len(a)),
            "n_scored": int(len(a)), "n_fit": int(len(ti)), "n_val": int(len(vi)), "n_nu": int(len(ti_nu)),
            "nu_gap_cells": int(n_gap), "nu_gap_arms": int(n_gap_arms),
            "alpha_mean_treated": float(a[~is_c].mean()), "alpha_std_treated": float(a[~is_c].std()),
            "alpha_mean_dmso": float(a[is_c].mean()) if is_c.any() else float("nan"),
            "alpha_max_dmso": float(a[is_c].max()) if is_c.any() else float("nan"),
            "pi0_fit": pi0_f,
        }

    print("=" * 72)
    print("URR FIT SUMMARY")
    print("=" * 72)
    for k, v in meta_out.items():
        print(f"  {k}: L_val={v['L_val']:+.4f}  beats const by {v['beats_constant_by']:+.4f}"
              f"   alpha mean={v['alpha_mean']:.3f} std={v['alpha_std']:.3f}"
              f"  max={v['alpha_max']:.2f}  ESS={100*v['ess_frac']:.1f}%"
              f"  tail_k={v['tail_khat']:.3f}")

    # A fit is USABLE only if it clears all four.
    #  learned    alpha actually varies with (X, A).
    #  mean ~ 1   the score identity E_P[alpha] = 1, proper density
    #  tail k     Hill index < 0.5  <=>  E[alpha^2] finite. (instability of weight check)
    #  ESS        see if ESS is reasonble too
    checks = {}
    for k, v in meta_out.items():
        checks[k] = {
            "learned":  v["beats_constant_by"] > 0.01 and v["alpha_std"] > 0.05,
            "mean_1":   abs(v["alpha_mean"] - 1.0) < 0.25,
            "tail_ok":  v["tail_khat"] < 0.5,
            "ess_ok":   v["ess_frac"] > 0.10,
        }
    ok = all(all(c.values()) for c in checks.values())
    print("\n  gate:")
    for k, c in checks.items():
        v = meta_out[k]
        print("   ", k, "  ".join(f"{n}={'PASS' if p else 'FAIL'}" for n, p in c.items()),
              f"   [beats_const={v['beats_constant_by']:+.3f}  std={v['alpha_std']:.3f}"
              f"  ess={v['ess_frac']:.3f}  k={v['tail_khat']:.2f}]")
    meta_out["gate"] = {"per_fold": checks, "usable": ok}

    meta_out["eval_model_for_fold"] = {
        "0": f"{args.out_prefix}_fold0.pt",
        "1": f"{args.out_prefix}_fold1.pt",
    }
    meta_out["fit"] = {
        "adjustment_set": list(cfg.adjustment_set), "X": keep, "cov_idx": cov_idx,
        "target_support": args.target_support, "n_nu_arms": len(nu_arms),
        "nu_rows": os.path.basename(nu_rows_path) if nu_rows_path else None, "n_nu_pool": int(len(fold_pool)),
        "run_id": run_id, "folds_sha1": hashlib.sha1(folds.tobytes()).hexdigest(),
        "split_fingerprint": splits["split_fingerprint"], "population": splits["population"],
        "fold_seed": cfg.seed, "seed": args.seed, "pi0_train": pi0,
        "args": {k: v for k, v in vars(args).items()},
    }

    if ok:
        print("\n  VERDICT: alpha is USABLE. It varies with (X, A), E[alpha] ~ 1 emerged")
        print("  on its own, the tail has finite variance, and the ESS is healthy.")
        print("  Next:  python -m src.nuisances.export_urr_weights --mode net   then  --dr_mode weighted")
    else:
        print("\n  VERDICT: alpha is NOT usable -- do NOT train the DR arm on it.")
        if any(not c["learned"] for c in checks.values()):
            print("   * learned=FAIL: alpha is ~constant, so w ~ 1 and DR == conditional.")
        if any(not c["mean_1"] for c in checks.values()):
            print("   * mean_1=FAIL: E[alpha] != 1, so the estimand is not the one you want.")
        if any(not (c["tail_ok"] and c["ess_ok"]) for c in checks.values()):
            print("   * tail/ESS=FAIL: heavy-tailed weights = OVERLAP failure. Shrink the")
            print("     conditioning set or the target support; more steps will not help.")

    # nets, folds and meta together, one run_id (export checks it)
    for f_, net in nets_out.items():
        net._spec["run_id"] = run_id
        net.save(os.path.join(nz, f"{args.out_prefix}_fold{f_}.pt"))
    np.save(os.path.join(nz, folds_filename(args.out_prefix)), folds)
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\n[urr] wrote {meta_path}, {args.out_prefix}_fold{{0,1}}.pt, {folds_filename(args.out_prefix)} (run {run_id})")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
