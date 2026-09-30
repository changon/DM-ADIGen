"""Turn fitted URR alpha into the dr_weights npz that training uses.

Copied from RxRx19a/src/nuisances/export_urr_weights.py, then adapted
(IMPLEMENT.md §3.5; D1, P1, P2). Bridges fit_urr -> train_diffusion: writes
{row_id, w} aligned to the split's train_idx.

Weight policy:
  - treated train rows: w = alpha(x, a)
  - vehicle rows: w = 1 (the vehicle arm keeps its weight, so the generator still learns Y(0))
  - --clip caps the tail (0 = off)
The trainer normalises w to mean 1.

Modes:
  net    (primary, P1) the cross-fitted AlphaNet from fit_urr: a fold-f row is
         scored by <prefix>_fold{f}.pt, the net fit on the OTHER fold (RxRx's
         export used fold{1-f}, the net fit on the row itself). Each fold's
         treated mean must reproduce fit_urr's out-of-fold mean.
         -> dr_weights_urr.npz
  counts (fallback) the closed-form discrete URR nu(a) / f_train(a) with add-k
         smoothing, keyed on the dose_level arm (P2), and on (arm, C) when the
         adjustment set C is non-empty; nu = nu_rows.npy. No torch.
         -> dr_weights_counts.npz
v1 known answers: counts gives w = 1 exactly (nu = the train pool); net gives
~1/(1 - pi0) = 1.066 on treated rows. In a thinning instance the weights are
compared with dr_weights_design.npz.

    python -m src.nuisances.export_urr_weights --mode net     [--nuisance_dir ...]
    python -m src.nuisances.export_urr_weights --mode counts  [--nuisance_dir ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import (  # noqa: E402
    _atomic_write, covariate_blocks, load_covariate_encoder)
from src.data.splits import (  # noqa: E402
    arm_keys, load_splits, population_rows, resolve_split_file)
from src.nuisances.knn_dr import ess  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, alpha_cov_fields, apply_paths_args, config_from_args)

DEFAULT_OUT = {"net": "dr_weights_urr.npz", "counts": "dr_weights_counts.npz"}


def counts_weights(rows: np.ndarray, nu: np.ndarray, key: np.ndarray, smooth_k: float) -> np.ndarray:
    """Closed-form discrete URR per action key: p_nu(k) / p_train(k), add-k smoothed
    over the actions of the scored rows. Identical pools give exactly 1."""
    acts, tr_cnt = np.unique(key[rows], return_counts=True)
    nu_map = dict(zip(*np.unique(key[nu], return_counts=True)))
    A = len(acts)
    p_tr = (tr_cnt + smooth_k) / (len(rows) + smooth_k * A)
    p_nu = (np.array([nu_map.get(k, 0) for k in acts]) + smooth_k) / (len(nu) + smooth_k * A)
    w_by_act = dict(zip(acts, (p_nu / p_tr).astype(np.float32)))
    return np.array([w_by_act[k] for k in key[rows]], dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("net", "counts"), default="net", help="'net' (primary): the cross-fitted AlphaNet from fit_urr. 'counts' (fallback): smoothed nu(a)/f_train(a) per arm, no torch.")
    p.add_argument("--nu_rows", default="nu_rows.npy", help="counts mode: the nu pool (build_tiered_split emits it), relative to the split dir.")
    p.add_argument("--smooth_k", type=float, default=0.5, help="counts mode: add-k smoothing on both action distributions.")
    p.add_argument("--prefix", default="alpha_urr", help="net mode: fit_urr's --out_prefix.")
    p.add_argument("--out", default=None, help="Default dr_weights_urr.npz (net) / dr_weights_counts.npz (counts).")
    p.add_argument("--clip", type=float, default=0.0, help="Cap on w (0 = off).")
    p.add_argument("--allow_unusable", action="store_true", help="net mode: export even if fit_urr's gate failed.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8192)
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    a = p.parse_args()

    cfg = apply_paths_args(config_from_args(a), a)
    nz = cfg.paths.nuisance_dir
    splits = load_splits(cfg)
    train_idx = splits["train_idx"]

    meta = load_from_disk(cfg.paths.tabular_dataset_dir).select_columns(
        ["cov_vec", "compound_idx", "log10_conc", "is_control", "dose_level", "pert_id"])
    cov = np.asarray(meta["cov_vec"], dtype=np.float32)
    comp = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx = np.asarray(meta["log10_conc"], dtype=np.float32)
    ic = np.asarray(meta["is_control"], dtype=np.float32)
    pop = population_rows(cfg, meta["pert_id"], ic)

    # X: the adjustment set shared with fit_urr and the generator
    enc = load_covariate_encoder(os.path.join(nz, "covariate_encoder.json"), cfg.covariates)
    blocks = covariate_blocks(enc)
    keep = list(alpha_cov_fields(cfg))
    cov_idx = sorted(i for b in keep for i in blocks[b])

    w = np.ones(len(train_idx), dtype=np.float32)
    score = ic[train_idx] < 0.5
    rows = train_idx[score]

    if a.mode == "counts":
        # Closed-form discrete URR: per action, w = p_nu / p_train with smoothing.
        # nu = the design (unthinned) pool, so on a tiered split this estimates the design weights from data.
        nu_path = resolve_split_file(nz, a.nu_rows)
        nu = np.load(nu_path).astype(np.int64)
        if not np.isin(train_idx, nu).all():
            raise ValueError(f"{nu_path} does not contain every train row; it belongs to another split")
        nu = nu[(ic[nu] < 0.5) & pop[nu]]
        key = arm_keys(comp, np.asarray(meta["dose_level"], dtype=np.float64), ic)
        if cov_idx:   # (arm, C) once C is non-empty (P2)
            cstr = np.array(["".join(map(str, r)) for r in (cov[:, cov_idx] > 0.5).astype(np.int8)], dtype=object)
            key = key + "|C=" + cstr
        out = counts_weights(rows, nu, key, a.smooth_k)
        print(f"[export] mode=counts: {len(np.unique(key[rows])):,} actions (X={keep or 'empty'}), "
              f"nu={len(nu):,} treated rows from {os.path.basename(nu_path)}, smooth_k={a.smooth_k}")
    else:
        import torch
        from src.nuisances.alpha_net import AlphaNet, spec_path
        dev = torch.device(a.device)
        with open(os.path.join(nz, f"{a.prefix}_meta.json")) as f:
            umeta = json.load(f)
        if umeta["fit"]["split_fingerprint"] != splits["split_fingerprint"]:
            raise RuntimeError(f"{a.prefix} was fit on split {umeta['fit']['split_fingerprint']}, not the "
                               f"current {splits['split_fingerprint']}; re-run src.nuisances.fit_urr")
        if not umeta["gate"]["usable"]:
            msg = f"{a.prefix}_meta.json: the URR gate failed ({umeta['gate']['per_fold']})"
            if not a.allow_unusable:
                raise SystemExit(f"[export] {msg}; use --mode counts, or --allow_unusable to export anyway")
            print(f"[export] WARNING {msg}; exporting anyway (--allow_unusable)")
        folds_fn = f"{a.prefix}_folds.npy"   # written by fit_urr
        folds = np.load(os.path.join(nz, folds_fn))
        run_id = umeta["fit"]["run_id"]
        if hashlib.sha1(folds.tobytes()).hexdigest() != umeta["fit"]["folds_sha1"] or not np.isin(folds[train_idx], (0, 1)).all():
            raise RuntimeError(f"{folds_fn} is not the one fit_urr run {run_id} used; re-run fit_urr")
        net_paths = {f: os.path.join(nz, f"{a.prefix}_fold{f}.pt") for f in (0, 1)}
        for f, pth in net_paths.items():
            with open(spec_path(pth)) as fh:
                if json.load(fh).get("run_id") != run_id:
                    raise RuntimeError(f"{os.path.basename(pth)} is not from fit_urr run {run_id}; re-run fit_urr")
        nets = {f: AlphaNet.load(pth, map_location=dev).to(dev).eval() for f, pth in net_paths.items()}
        for f, net in nets.items():
            if net._spec["cov_idx"] != cov_idx or net._spec["cov_dim"] != cov.shape[1]:
                raise ValueError(f"{a.prefix}_fold{f} conditions on cov cols {net._spec['cov_idx']}, but "
                                 f"--adjustment_set gives {cov_idx} (X={keep}); export with the fit's adjustment set")
        row_fold = folds[rows]
        out = np.empty(len(rows), dtype=np.float32)
        with torch.no_grad():
            for f in (0, 1):
                sel = np.where(row_fold == f)[0]
                # cross-fit: fold-f rows are scored by fold{f}.pt, which fit_urr fit on the OTHER fold
                net = nets[f]
                for s in range(0, len(sel), a.batch):
                    r = rows[sel[s:s + a.batch]]
                    out[sel[s:s + a.batch]] = net(
                        torch.tensor(cov[r], device=dev),
                        torch.tensor(comp[r], device=dev),
                        torch.tensor(lx[r], device=dev),
                        torch.tensor(ic[r], device=dev),
                    ).squeeze(-1).float().cpu().numpy()
        for f in (0, 1):   # the same nets on the same rows as fit_urr's out-of-fold scoring
            got, want = float(out[row_fold == f].mean()), umeta[f"fold{f}"]["alpha_mean_treated"]
            if abs(got - want) > 1e-4:
                raise RuntimeError(f"fold {f}: exported treated mean {got:.6f} != fit_urr's out-of-fold {want:.6f}")
        print(f"[export] mode=net: {a.prefix}_fold{{0,1}}.pt (X={keep or 'empty'}, run {run_id}), cross-fitted; "
              f"per-fold treated means match fit_urr's out-of-fold scores")
    if a.clip > 0:
        n_clip = int((out > a.clip).sum())
        out = np.minimum(out, a.clip)
        print(f"[export] clipped {n_clip} weights at {a.clip}")
    w[score] = out

    dst = os.path.join(nz, a.out or DEFAULT_OUT[a.mode])
    _atomic_write(dst, lambda f: np.savez(f, row_id=train_idx, w=w, mode=np.array(a.mode),
                                          split_fingerprint=np.array(splits["split_fingerprint"])), mode="wb")
    wn = w / w.mean()
    nz1 = w[np.abs(w - 1.0) > 1e-6]
    print(f"[export] wrote {dst}  n={len(w):,}  scored={int(score.sum()):,}  w!=1: {len(nz1):,}")
    print(f"[export] w: mean {w.mean():.4f}  min {w.min():.3f}  max {w.max():.3f}  ESS/n {ess(w.astype(np.float64))/len(w):.3f}")
    print(f"[export] treated w: mean {w[score].mean():.4f} std {w[score].std():.4f}; vehicle w = 1 "
          f"({int((~score).sum()):,} rows). After the trainer's mean-1 normalisation: treated "
          f"{wn[score].mean():.4f}, vehicle {wn[~score].mean():.4f}")

    # validation against the design truth, when present
    dpath = os.path.join(nz, "dr_weights_design.npz")
    if os.path.isfile(dpath):
        dz = np.load(dpath)
        if not (np.array_equal(dz["row_id"], train_idx) and str(dz["split_fingerprint"]) == splits["split_fingerprint"]):
            raise RuntimeError(f"{dpath} belongs to another split; rebuild the split dir")
        wd = dz["w"].astype(np.float64)
        m = np.abs(wd - 1.0) > 1e-6
        # gamma = 0 gives constant design weights: no correlation to report
        c = (float(np.corrcoef(w[m], wd[m])[0, 1])
             if m.sum() > 2 and wd[m].std() > 0 and w[m].std() > 0 else float("nan"))
        print(f"[export] vs design (on {int(m.sum()):,} design-weighted rows): "
              f"corr {c:+.3f}  learned mean there {w[m].mean():.3f} "
              f"vs design {wd[m].mean():.3f}")


if __name__ == "__main__":
    main()
