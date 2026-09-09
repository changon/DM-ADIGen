"""URR Riesz fit

    L(alpha) =  E_{(X,A) ~ P}      [ alpha(X, A )^2 ]    -  2 E_{X ~ P_X, At ~ f_A} [ alpha(X, At) ] 

Outputs (cfg.paths.nuisance_dir):
    alpha_urr_fold{0,1}.pt, <prefix>_meta.json   -- this fit
    fold_assignment.npy, nuisance_meta.json

Run:  sbatch scripts/fit_urr.slurm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import (  # noqa: E402
    covariate_blocks, load_covariate_encoder)
from src.data.splits import load_splits  # noqa: E402
from src.nuisances.alpha_net import AlphaNet  # noqa: E402
from src.nuisances.knn_dr import ess, tail_index  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, alpha_cov_fields, config_from_args, format_role_summary, invariance_env_fields)

CONST_BASELINE = -1.0 

# Divergence checks on loss
DIVERGENCE_L = 1e3
DIVERGENCE_A = 1e3

def urr_loss(net, cov, comp, lx, ic, inf, comp_t, lx_t, ic_t, ridge: float = 0.0, shrink: float = 0.0):
    """L = E[alpha(X,A)^2] - 2 E[alpha(X,At)],   At ~ f_A drawn independently of X.

    (cov, inf) are the row's covariates X. (comp, lx, ic) is its FACTUAL action A.
    (comp_t, lx_t, ic_t) are (B, M) actions resampled from the empirical marginal.
    """
    B, M = comp_t.shape
    a_obs = net(cov, comp, lx, ic, inf).squeeze(-1)                      # (B,)   at (X_i, A_i)

    cov_e = cov.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
    inf_e = inf.view(B, 1).expand(B, M).reshape(B * M)
    a_prd = net(cov_e, comp_t.reshape(-1), lx_t.reshape(-1),  ic_t.reshape(-1), inf_e).squeeze(-1).view(B, M)          # (B,M)  at (X_i, At)

    sq = (a_obs ** 2).mean()
    # shrink anchors alpha at the null (==1): with ~15 rows per discrete action the unpenalised minimiser memorises count noise (ESS ~1%).
    pen = shrink * ((a_obs - 1.0) ** 2).mean() if shrink else 0.0
    return sq - 2.0 * a_prd.mean() + ridge * sq + pen, a_obs

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_perm", type=int, default=4, help="Marginal-draw actions per row (>=2).")
    p.add_argument("--ridge", type=float, default=0.0)
    p.add_argument("--shrink_to_one", type=float, default=0.0, help="Penalty weight on (alpha-1)^2: defaults alpha to the null unless the data insists. The failed 2026-09-08 fits (ESS ~1%%, inverted vs design) had 0.")
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cov_blocks", default=None, help="One-hot blocks of cov_vec alpha conditions on -- the adjustment set.")
    p.add_argument("--target_support", default="common", choices=("common", "all"), help="Support of the target intervention distribution nu.")
    p.add_argument("--out_prefix", default="alpha_urr",  help="Basename for the saved nets/meta (default alpha_urr -> alpha_urr_fold{0,1}.pt + alpha_urr_meta.json).")
    p.add_argument("--nu_source", default="train", choices=("train", "full"),  help="Where the TARGET intervention distribution nu is drawn from")
    p.add_argument("--nu_rows", default="", help="Explicit .npy of row indices defining nu (e.g. the UNTHINNED train pool of a tiered split, so alpha targets the design action distribution). Overrides --nu_source.")
    p.add_argument("--positive", type=int, default=1, help="Softplus head: alpha >= 0, as a density ratio must be.")
    p.add_argument("--cell_type", default="all", choices=("all", "HRCE", "VERO"),  help="STRATIFY on cell line instead of adjusting for it. motivated by rxrx19a")
    p.add_argument("--nuisance_dir", type=str, default="", help="Override cfg.paths.nuisance_dir with a prebuilt split dir (build_tiered_split.py). Supplies splits.json, vocab, covariate encoder; the fitted alpha lands there.")
    add_adjustment_set_cli(p)
    args = p.parse_args()
    if args.n_perm < 2:
        raise ValueError("n_perm >= 2")

    cfg = config_from_args(args)
    dev = torch.device(args.device)

    if args.nuisance_dir:
        if not os.path.isfile(os.path.join(args.nuisance_dir, "splits.json")):
            raise FileNotFoundError(f"{args.nuisance_dir} has no splits.json")
        cfg.paths.nuisance_dir = args.nuisance_dir

    base_nz = nz = cfg.paths.nuisance_dir
    os.makedirs(nz, exist_ok=True)
    # n_compounds and cov_dim come from what build_dataset wrote
    with open(os.path.join(base_nz, "compound_vocab.json")) as f:
        n_compounds = len(json.load(f))

    # load arrays
    meta = load_from_disk(cfg.paths.tabular_dataset_dir)
    cov_all = np.asarray(meta["cov_vec"], dtype=np.float32)
    comp_all = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx_all = np.asarray(meta["log10_conc"], dtype=np.float32)
    ic_all = np.asarray(meta["is_control"], dtype=np.float32)
    dis = np.array([str(x) for x in meta["disease_condition"]])
    inf_all = (dis == cfg.population.disease_condition).astype(np.float32)

    # splits + cross-fit folds.
    train_idx = np.asarray(load_splits(cfg)["train_idx"], dtype=np.int64)
    _rng = np.random.RandomState(cfg.seed)
    _perm = train_idx[_rng.permutation(train_idx.shape[0])]
    _half = _perm.shape[0] // 2
    folds = np.full(len(meta), -1, dtype=np.int8)
    folds[_perm[:_half]] = 0
    folds[_perm[_half:]] = 1
    np.save(os.path.join(nz, "fold_assignment.npy"), folds)

    # 3. which pop
    _inf_tr = inf_all[train_idx] == 1
    if len(folds) == len(train_idx):
        folds = folds[_inf_tr]
    train_idx = train_idx[_inf_tr]
    print(f"[urr] restricted to infected==1: {int(_inf_tr.sum()):,}/{len(_inf_tr):,} train rows")

    # 3b. which cell line
    _expt_all = np.array([str(x) for x in meta["experiment"]])
    _ct_all = np.array([str(x) for x in meta["cell_type"]])

    def _apply_mask(mask: np.ndarray, label: str) -> None:
        nonlocal folds, train_idx
        if len(folds) == len(train_idx):
            folds = folds[mask]
        train_idx = train_idx[mask]
        print(f"[urr] {label}: kept {int(mask.sum()):,}/{len(mask):,} train rows")

    if args.cell_type != "all":
        _apply_mask(_ct_all[train_idx] == args.cell_type, f"cell_type=={args.cell_type}")

    fold_tr = (folds if len(folds) == len(train_idx) else folds[train_idx]).astype(np.int64)
    cov_dim = cov_all.shape[1]

    # ---------------------------------------------------------------------
    # derive X (column indices)
    enc = load_covariate_encoder(
        os.path.join(base_nz, "covariate_encoder.json"), cfg.covariates)
    blocks = covariate_blocks(enc)
    # A design nu (--nu_rows) is an ACTION-distribution target: the matching
    # alpha is action-marginal, so default X to empty rather than the role-E
    # covariates (which stratify the support and truncate nu -- the 2026-09-08
    # divergence). Explicit --cov_blocks still overrides.
    if args.nu_rows and args.cov_blocks is None:
        args.cov_blocks = ""
        print("[urr] --nu_rows given -> cov_blocks defaulted to EMPTY "
              "(action-marginal alpha); pass --cov_blocks to override")
    # None = take the shared declaration; "" = an explicit empty override.
    if args.cov_blocks is None:
        print(f"[urr] roles: {format_role_summary(cfg)}")
        keep = list(alpha_cov_fields(cfg))
        print(f"[urr] alpha(A, C, E): C={list(cfg.adjustment_set) or '(empty)'} "
              f"E={list(invariance_env_fields(cfg))} -> cov blocks "
              f"{keep or '(empty -- alpha == 1)'}")
        cov_idx = sorted(i for b in keep for i in blocks[b])
    elif args.cov_blocks.strip() == "all":
        cov_idx = list(range(cov_dim))
        keep = list(blocks)
    else:
        print(f"[urr] *** --cov_blocks OVERRIDE {args.cov_blocks!r} (CaseConfig says {list(cfg.adjustment_set)}). The generator must be trained with a matching adjustment_set or validate_against will fire.")
        keep = [b.strip() for b in args.cov_blocks.split(",") if b.strip()]
        bad = [b for b in keep if b not in blocks]
        if bad:
            raise ValueError(f"unknown cov block(s) {bad}; have {list(blocks)}")
        cov_idx = sorted(i for b in keep for i in blocks[b])
    if "plate" in keep:
        print("[urr] *** WARNING: conditioning on PLATE. f(a|x)=0 on 94.6% of the product measure -> alpha is unbounded and this fit WILL diverge. ***")

    # The Population mask. nu is the target intervention distribution,
    pop_mask_all = np.ones(len(comp_all), dtype=bool)
    if cfg.population.compounds:
        _vocab = json.load(open(os.path.join(base_nz, "compound_vocab.json")))
        _want_ids = {int(v) for k, v in _vocab.items() if k in set(cfg.population.compounds)}
        _ctl_ids = {int(v) for k, v in _vocab.items() if k == cfg.action.control_token}
        pop_mask_all = np.isin(comp_all, list(_want_ids | _ctl_ids))
        print(f"[urr] nu restricted to the population: {len(_want_ids)} compounds, {int(pop_mask_all.sum()):,}/{len(pop_mask_all):,} rows eligible")

    # get support of the target intervention distribution nu, under the chosen X
    strata = [tuple(r) for r in (cov_all[:, cov_idx] > 0.5).astype(np.int8)]

    # Overlap is a property of the ACTION, and the action is (compound, dose): a compound can span both strata while its top dose sits in only one.  Check support on the rows f is estimated from -- the ablated train set

    arm_all = list(zip(comp_all.tolist(), np.round(lx_all, 6).tolist()))
    per_stratum: dict[tuple, set] = {}
    for i in train_idx:
        if ic_all[i] >= 0.5 or not pop_mask_all[i]:
            continue                              # controls are the reference, not a target
        per_stratum.setdefault(strata[i], set()).add(arm_all[i])
    common = set.intersection(*per_stratum.values()) if per_stratum else set()
    if args.target_support == "common":
        nu_arms = common
        _seen = set().union(*per_stratum.values()) if per_stratum else set()
        _drop = len(_seen) - len(nu_arms)
        print(f"[urr] arm-level common support: {len(nu_arms)}/{len(_seen)} (compound,dose) arms span all {len(per_stratum)} strata; dropped {_drop} for positivity")
    else:
        nu_arms = {a for a, isc in zip(arm_all, ic_all) if isc < 0.5}
        print("[urr] *** WARNING: nu = full marginal. 94.6% of it is unsupported. ***")
    nu_compounds = {int(c) for c, _ in nu_arms}
    if not nu_compounds:
        raise RuntimeError("target support is empty -- no compound spans every stratum")
    _nu_set = set(nu_arms)
    nu_mask_all = np.array([a in _nu_set for a in arm_all], dtype=bool)

    # check some debugs
    print(f"[urr] train={len(train_idx):,}  n_compounds={n_compounds}")
    print(f"[urr] X = {keep}  -> cov cols {len(cov_idx)}/{cov_dim}, "
          f"{len(per_stratum)} strata")
    print(f"[urr] nu support = {len(nu_compounds)} compounds "
          f"({args.target_support}); {nu_mask_all.mean():.1%} of rows carry one")
    print(f"[urr] head = {'softplus (alpha >= 0)' if args.positive else 'linear'}")
    print(f"[urr] loss: E[a(X,A)^2] - 2 E[a(X,At)],  At ~ nu  (quadratic at FACTUAL, "
          f"linear at PRODUCT)")
    print(f"[urr] constant baseline L(a==1) = {CONST_BASELINE:+.4f}  <- must beat this\n")

    def T(a):
        return torch.tensor(a, device=dev)

    # now CF.
    meta_out = {}
    for f_ in (0, 1):
        # cross-fit: train on rows not in fold f_, score rows in fold f_
        fit_rows = train_idx[fold_tr != f_]
        rng = np.random.default_rng(args.seed + f_)
        perm = rng.permutation(len(fit_rows))
        nv = max(512, int(args.val_frac * len(fit_rows)))
        vi, ti = fit_rows[perm[:nv]], fit_rows[perm[nv:]]
        print(f"[urr] fold {f_}: fit={len(ti):,}  val={len(vi):,}")

        # Rows whose action is drawable under nu, the prod measure
        if args.nu_rows:
            pool = np.load(args.nu_rows).astype(np.int64)
            pool = pool[inf_all[pool] == 1]
            pool = pool[pop_mask_all[pool]]
            if len(folds) > pool.max():   # rows never in the thinned/confounded train carry fold -1
                pool = pool[folds[pool] != f_]
            ti_nu = pool[nu_mask_all[pool]]
            print(f"[urr] fold {f_}: nu drawn from --nu_rows ({len(ti_nu):,} rows) -- alpha targets that design action distribution")
        elif args.nu_source == "full": # not in use. use --nu_rows for a design nu.
            full_train = np.asarray(json.load(open(os.path.join(base_nz, "splits.json")))["train_idx"], dtype=np.int64)
            full_train = full_train[inf_all[full_train] == 1]
            full_train = full_train[pop_mask_all[full_train]]
            full_fold = folds[full_train] if len(folds) > full_train.max() else None
            pool = full_train if full_fold is None else full_train[full_fold != f_]
            ti_nu = pool[nu_mask_all[pool]]
            print(f"[urr] fold {f_}: nu drawn from the dir's full train "
                  f"({len(ti_nu):,} rows)")
        else: # on train mask, get nu
            ti_nu = ti[nu_mask_all[ti]]
        if len(ti_nu) < 1000:
            raise RuntimeError(f"only {len(ti_nu)} rows on the nu support -- too few")
        print(f"[urr] fold {f_}: nu-support rows for the product leg = {len(ti_nu):,}")

        # setup model.
        net = AlphaNet(cov_dim=cov_dim, n_compounds=n_compounds, cov_idx=cov_idx, positive=bool(args.positive)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        brng = np.random.default_rng(args.seed + 100 + f_)

        # fixed val batch.
        vrng = np.random.default_rng(args.seed + 500 + f_)
        vj = ti_nu[vrng.integers(0, len(ti_nu), (len(vi), args.n_perm))]
        V = (T(cov_all[vi]), T(comp_all[vi]), T(lx_all[vi]), T(ic_all[vi]), T(inf_all[vi]),
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
                T(inf_all[ix]), T(comp_all[jx]), T(lx_all[jx]), T(ic_all[jx]),
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
                      f"std={a.std():.3f} max={a.max():.2f}")
                # Divergence check.
                if vl < -DIVERGENCE_L or a.max() > DIVERGENCE_A:
                    raise RuntimeError(
                        f"URR DIVERGED at step {step}: L_val={vl:.1f}, "
                        f"max(alpha)={a.max():.1f}, mean(alpha)={a.mean():.1f}.\n"
                        f"  alpha is running to +inf on target actions with no observed "
                        f"support -- the Int1 overlap assumption is violated for\n"
                        f"    X = {keep}   nu = {args.target_support} "
                        f"({len(nu_compounds)} compounds).\n"
                        f"  Shrink the conditioning set (drop plate) or the target "
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
        net.save(os.path.join(nz, f"{args.out_prefix}_fold{f_}.pt"))

        # score the HELD-OUT fold, so out of sample computation of URR.
        score_rows = train_idx[fold_tr == f_]
        net.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(score_rows), 8192):
                r = score_rows[s : s + 8192]
                outs.append(net(T(cov_all[r]), T(comp_all[r]), T(lx_all[r]),
                                T(ic_all[r]), T(inf_all[r])).squeeze(-1).cpu().numpy())

        # some checks.
        a = np.concatenate(outs).astype(np.float64)
        kh = tail_index(np.abs(a))
        print(f"   -> fold {f_}: L_val={best:+.4f}   alpha(held-out): mean={a.mean():.4f} "
              f"std={a.std():.4f} min={a.min():.3f} max={a.max():.2f} "
              f"ESS={100*ess(np.abs(a))/len(a):.1f}% tail_k={kh:.3f}\n")
        meta_out[f"fold{f_}"] = {
            "L_val": best, "const_baseline": CONST_BASELINE,
            "beats_constant_by": CONST_BASELINE - best,
            "alpha_mean": float(a.mean()), "alpha_std": float(a.std()),
            "alpha_min": float(a.min()), "alpha_max": float(a.max()),
            "tail_khat": kh, "ess_frac": float(ess(np.abs(a)) / len(a)),
        }

    gains = [v["beats_constant_by"] for v in meta_out.values()]
    spreads = [v["alpha_std"] for v in meta_out.values()]
    print("=" * 72)
    print("URR FIT SUMMARY   (old kernel loss had minimiser alpha == 1 identically)")
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

    if ok:
        print("\n  VERDICT: alpha is USABLE. It varies with (X, A), E[alpha] ~ 1 emerged")
        print("  on its own, the tail has finite variance, and the ESS is healthy.")
        print("  Next:  python -m src.nuisances.export_urr_weights   then  --dr_mode knn_dr")
    else:
        print("\n  VERDICT: alpha is NOT usable -- do NOT train the DR arm on it.")
        if any(not c["learned"] for c in checks.values()):
            print("   * learned=FAIL: alpha is ~constant, so w ~ 1 and DR == conditional.")
        if any(not c["mean_1"] for c in checks.values()):
            print("   * mean_1=FAIL: E[alpha] != 1, so the estimand is not the one you want.")
        if any(not (c["tail_ok"] and c["ess_ok"]) for c in checks.values()):
            print("   * tail/ESS=FAIL: heavy-tailed weights = OVERLAP failure. Shrink the")
            print("     conditioning set or the target support; more steps will not help.")

    with open(os.path.join(nz, f"{args.out_prefix}_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\n[urr] wrote {os.path.join(nz, args.out_prefix + '_meta.json')}")

    # Merge, not overwrite: this fit owns n_compounds and cov_dim
    _nm_path = os.path.join(nz, "nuisance_meta.json")
    _nm = {}
    if os.path.exists(_nm_path):
        with open(_nm_path) as f:
            _nm = json.load(f)
    _nm.update({"n_compounds": int(n_compounds), "cov_dim": int(cov_dim)})
    with open(_nm_path, "w") as f:
        json.dump(_nm, f, indent=2)
    print(f"[urr] wrote {_nm_path} (n_compounds={n_compounds}, cov_dim={cov_dim}, "
          f"{len(_nm)} keys)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
