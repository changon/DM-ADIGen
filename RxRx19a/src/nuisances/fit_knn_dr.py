"""Precompute ADIGen's kNN doubly-robust training weights.

Pipeline (before diffusion training):

    1. score alpha at (X_i, A_i), cross-fitted (row in fold f is scored by the alpha net trained on the OTHER fold)
    2. two-query kNN AIPW      ->  w_i, the per-sample loss multiplier

Output: data/nuisances/dr_weights_knn.npz
    {row_id, w, alpha_raw}                        <- TRAIN rows (clipped, for SGD)

Run:  sbatch scripts/fit_knn_dr.slurm
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

from src.data.rarity import (  # noqa: E402
    add_rarity_cli_args,
    rarity_from_args,
    tagged_nuisance_dir,
)

from src.data.build_dataset import (  # noqa: E402
    covariate_blocks, load_covariate_encoder)
from src.data.splits import load_splits  # noqa: E402
from src.nuisances.alpha_net import AlphaNet, spec_path  # noqa: E402
from src.nuisances.knn_dr import (  # noqa: E402
    ess,
    knn_dr_weights,
    tail_index,
)
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, alpha_cov_fields, config_from_args, default_config,
    format_role_summary, invariance_env_fields)

def _describe(name: str, w: np.ndarray) -> None:
    print(f"  {name:14} mean={w.mean():+.4f} median={np.median(w):+.4f} "
          f"std={w.std():.4f} min={w.min():+.3f} max={w.max():+.3f}  "
          f"ESS={ess(np.abs(w)):,.0f}/{len(w):,} ({100*ess(np.abs(w))/len(w):.1f}%)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=12, help="kNN neighbours per query.")
    p.add_argument("--bandwidth", type=float, default=None,
                   help="Dose kernel bandwidth (default: cfg.action.continuous_kernel_bandwidth).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_clip_neg", action="store_true",
                   help="Keep negative weights (evaluate unclipped; training wants them clipped).")
    p.add_argument("--out", default=None)
    p.add_argument("--alpha_prefix", default="alpha_urr",help="Which fit_urr run supplies alpha. 'alpha_urr' (default) "
                        "drew its target actions from the ABLATED pool, so alpha "
                        "is blind to a dose-biased ablation by construction -- "
                        "training the DR generator on those weights reweights "
                        "nothing. Pass 'alpha_urr_nufull' for the --nu_source "
                        "full fit. The output .npz is tagged with the prefix so "
                        "the two weight sets cannot overwrite each other.")
    p.add_argument("--variant", default="aipw",
                   choices=("aipw", "ipw"),
                   help="aipw (default): two-query kNN AIPW weight. "
                        "ipw: w = alpha directly (single-query, no augmentation) -- "
                        "the ablation isolating the augmentation term.")
    add_adjustment_set_cli(p)
    add_rarity_cli_args(p)
    args = p.parse_args()

    cfg = config_from_args(args)
    rcfg = rarity_from_args(args)
    device = torch.device(args.device)
    
    # check rarity case
    base_nz = cfg.paths.nuisance_dir   # covariate_encoder.json lives here, untagged
    if rcfg.active:
        cfg.paths.nuisance_dir = tagged_nuisance_dir(cfg, rcfg)
        print(f"[knn-dr] rarity active: tag={rcfg.tag} -> {cfg.paths.nuisance_dir}")
    nz = cfg.paths.nuisance_dir
    bw = args.bandwidth if args.bandwidth is not None else float(
        cfg.action.continuous_kernel_bandwidth)

    with open(os.path.join(nz, "nuisance_meta.json")) as f:
        n_compounds = int(json.load(f)["n_compounds"])

    # load vecs
    meta = load_from_disk(cfg.paths.tabular_dataset_dir)
    cov_all = np.asarray(meta["cov_vec"], dtype=np.float32)
    comp_all = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx_all = np.asarray(meta["log10_conc"], dtype=np.float32)
    ic_all = np.asarray(meta["is_control"], dtype=np.int64)
    dis = np.array([str(x) for x in meta["disease_condition"]])
    inf_all = (dis == cfg.population.disease_condition).astype(np.float32)

    # splits for the CF
    splits = load_splits(cfg)
    train_idx = np.asarray(splits["train_idx"], dtype=np.int64)
    folds = np.load(os.path.join(nz, "fold_assignment.npy"))
    if len(folds) == len(train_idx):
        fold_tr = folds.astype(np.int64)
    else:                                     # stored over all rows
        fold_tr = folds[train_idx].astype(np.int64)
    print(f"[knn-dr] train rows={len(train_idx):,}  folds={np.bincount(fold_tr)}  "
          f"k={args.k}  dose bandwidth={bw}")

    cov = cov_all[train_idx]
    comp = comp_all[train_idx]
    lx = lx_all[train_idx]
    ic = ic_all[train_idx]
    inf = inf_all[train_idx]

    # Nuisances condition as
    #   alpha(C, E, A) 
    #   psi(C, A)
    _blocks = covariate_blocks(load_covariate_encoder(
        os.path.join(base_nz, "covariate_encoder.json"), cfg.covariates))
    _c_names = tuple(cfg.adjustment_set)
    _unknown = [b for b in _c_names if b not in _blocks]
    if _unknown:
        raise ValueError(
            f"adjustment_set names {_unknown}, absent from cov_vec blocks "
            f"{list(_blocks)}; mark them adjustable=True in spec.FIELDS and "
            f"re-run build_dataset.")
    _c_idx = sorted(i for b in _c_names for i in _blocks[b])
    cov_c = cov[:, _c_idx]

    # check alpha fields and psi fields are kept straight for DR weight building.
    _want_alpha = sorted(i for b in alpha_cov_fields(cfg) for i in _blocks[b])
    for _f in (0, 1):
        _sp_p = spec_path(os.path.join(nz, f"{args.alpha_prefix}_fold{_f}.pt"))
        if not os.path.exists(_sp_p):
            continue
        with open(_sp_p) as _fh:
            _got = list(json.load(_fh).get("cov_idx", []))
        if _got != _want_alpha:
            _inv = {i: b for b, idx in _blocks.items() for i in idx}
            raise ValueError(
                f"alpha/psi ROLE MISMATCH in {nz}.\n"
                f"  alpha_urr_fold{_f} was fit on cov_idx={_got} "
                f"-> blocks {sorted({_inv.get(i, '?') for i in _got})}\n"
                f"  this arm declares C={list(cfg.adjustment_set) or '[]'} "
                f"E={list(invariance_env_fields(cfg))} "
                f"-> alpha X should be {list(alpha_cov_fields(cfg))} "
                f"(cov_idx={_want_alpha})\n"
                f"  Both DR legs must be configured from the same declaration. "
                f"Re-run fit_urr with these roles, or point --adjustment_set / "
                f"--environment_set at the roles the nets were fit under.")
    print(f"[knn-dr] roles: {format_role_summary(cfg)}")
    print(f"[knn-dr] alpha conditions on the net's recorded cov_idx (C+E); "
          f"psi conditions on C={list(_c_names) or '(empty)'} -> "
          f"{len(_c_idx)}/{cov.shape[1]} cov columns")
    if not _c_idx:
        print("[knn-dr] NOTE C is empty, so psi matches donors within the "
              "(compound, is_control, dose-band) bucket only -- correct when "
              "there is nothing to adjust for, but then DR == conditional.")

    # Train URR
    nets = {}
    for f_ in (0, 1):
        p = os.path.join(nz, f"{args.alpha_prefix}_fold{f_}.pt")
        if not os.path.exists(spec_path(p)):
            raise FileNotFoundError(
                f"{p} has no .spec.json -- it predates the URR fix. Re-run "
                f"`python -m src.nuisances.fit_urr` before fitting DR weights."
            )
        net = AlphaNet.load(p, map_location=device).to(device)
        net.eval()
        nets[f_] = net

    _meta_p = os.path.join(nz, f"{args.alpha_prefix}_meta.json")
    if not os.path.exists(_meta_p):
        _meta_p = os.path.join(nz, "urr_meta.json")
    _emf = {}
    if os.path.exists(_meta_p):
        with open(_meta_p) as _mf:
            _emf = (json.load(_mf) or {}).get("eval_model_for_fold") or {}

    def _net_for_fold(f_: int) -> int:
        fname = _emf.get(str(f_))
        if fname:
            return int(fname.rsplit("_fold", 1)[1].split(".")[0])
        return f_          # identity: fit_urr's convention

    print(f"[knn-dr] fold->net map: "
          + ", ".join(f"{f_}->{_net_for_fold(f_)}" for f_ in (0, 1))
          + ("  (from meta)" if _emf else "  (identity default)"))

    def _apply(net, r, cov_, comp_, lx_, ic_, inf_):
        return net(
            torch.tensor(cov_[r], device=device),
            torch.tensor(comp_[r], device=device),
            torch.tensor(lx_[r], device=device),
            torch.tensor(ic_[r], dtype=torch.float32, device=device),
            torch.tensor(inf_[r], device=device),
        ).squeeze(-1).cpu().numpy()

    def score_alpha(cov_, comp_, lx_, ic_, inf_, fold_=None) -> np.ndarray:
        """alpha scored out of sample"""
        n_ = len(comp_)
        out = np.zeros(n_, dtype=np.float64)
        with torch.no_grad():
            if fold_ is None:
                for s in range(0, n_, 4096):
                    r = np.arange(s, min(s + 4096, n_))
                    out[r] = 0.5 * (_apply(nets[0], r, cov_, comp_, lx_, ic_, inf_)
                                    + _apply(nets[1], r, cov_, comp_, lx_, ic_, inf_))
            else:
                for f_ in (0, 1):
                    rows = np.where(fold_ == f_)[0]
                    if len(rows) == 0:
                        continue
                    net_f = nets[_net_for_fold(f_)]
                    for s in range(0, len(rows), 4096):
                        r = rows[s : s + 4096]
                        out[r] = _apply(net_f, r, cov_, comp_, lx_, ic_, inf_)
        return out

    alpha_raw = score_alpha(cov, comp, lx, ic, inf, fold_=fold_tr)

    print("\n[knn-dr] stage 1 -- alpha at factual pairs")
    _describe("alpha_raw", alpha_raw)
    k_hat = tail_index(np.abs(alpha_raw))
    print(f"  tail index k_hat={k_hat:.3f}   E[a^2] finite (k<0.5): {k_hat < 0.5}   "
          f"PSIS usable (k<0.7): {k_hat < 0.7}")
    if alpha_raw.std() < 0.1:
        print("  !! WARNING: alpha is nearly CONSTANT. Per ADIGen: alpha == 1 <=> no confounding. Refit the nuisances.")

    # -- 2. build the training weight per --variant -------------------------
    if args.variant == "ipw":
        # IPW ablation
        w = alpha_raw.astype(np.float64)
        print("\n[knn-dr] variant=ipw -- w = raw alpha (single-query IPW, no augmentation)")
        _describe("w (final)", w)
    else:
        # DR
        print("\n[knn-dr] stage 2 -- two-query kNN AIPW weights"
              "\n         (plug-in leg at DONOR covariates, correction at observed)")
        w = knn_dr_weights(
            cov=cov_c, compound=comp, log10_conc=lx, is_control=ic,
            alpha=alpha_raw, folds=fold_tr,
            k=args.k, bandwidth=bw, seed=args.seed,
            clip_neg=not args.no_clip_neg,
        )
        _describe("w (final)", w.astype(np.float64))

    # Tag the file with the alpha it came from
    _tag = "" if args.alpha_prefix == "alpha_urr" else f"_{args.alpha_prefix}"
    out = args.out or os.path.join(nz, f"dr_weights_knn{_tag}.npz")
    np.savez(out, row_id=train_idx, w=w,
             alpha_raw=alpha_raw.astype(np.float32))
    print(f"\n[knn-dr] wrote {out}")
    print("[knn-dr] use with: train_diffusion.py --dr_mode knn_dr")

if __name__ == "__main__":
    main()
