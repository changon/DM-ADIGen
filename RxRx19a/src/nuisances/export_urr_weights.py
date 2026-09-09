"""use fitted URR alpha into the dr_weights npz that training uses.

Bridges fit_urr -> train_diffusion: evaluates the CROSS-FITTED alpha nets at every train row and writes {row_id, w} next to the split

Weight policy:
  - treated infected rows: w = alpha(x, a) from the other fold's net
  - control rows: w = 1  
  - --clip caps the tail (0 = off)

    python -m scripts.export_urr_weights \
        --nuisance_dir data/nuisances_tier_k2_pg2_s42 \
        --prefix alpha_urr_design --out dr_weights_urr.npz
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

from src.nuisances.alpha_net import AlphaNet  # noqa: E402
from src.nuisances.knn_dr import ess  # noqa: E402
from src.spec import default_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nuisance_dir", required=True)
    p.add_argument("--mode", choices=("net", "counts"), default="counts", help="'counts' = the closed-form discrete URR: smoothed empirical ratio nu(a)/f_train(a) per action, no net, no GPU. 'net' = evaluate the cross-fitted AlphaNet from fit_urr (currently fails its quality gate at this action granularity).")
    p.add_argument("--nu_rows", default="nu_rows_design.npy", help="counts mode: the nu pool (build_tiered_split emits it).")
    p.add_argument("--smooth_k", type=float, default=0.5, help="counts mode: add-k smoothing on both action distributions.")
    p.add_argument("--prefix", default="alpha_urr_design")
    p.add_argument("--out", default="dr_weights_urr.npz")
    p.add_argument("--clip", type=float, default=0.0, help="Cap on w (0 = off).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8192)
    a = p.parse_args()

    cfg = default_config()
    dev = torch.device(a.device)
    nz = a.nuisance_dir

    train_idx = np.asarray( json.load(open(os.path.join(nz, "splits.json")))["train_idx"], dtype=np.int64)
    folds = np.load(os.path.join(nz, "fold_assignment.npy"))

    meta = load_from_disk(cfg.paths.tabular_dataset_dir)
    cov = np.asarray(meta["cov_vec"], dtype=np.float32)
    comp = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx = np.asarray(meta["log10_conc"], dtype=np.float32)
    ic = np.asarray(meta["is_control"], dtype=np.float32)
    dis = np.array([str(x) for x in meta["disease_condition"]])
    inf = (dis == str(cfg.population.disease_condition)).astype(np.float32)

    w = np.ones(len(train_idx), dtype=np.float32)
    score = (ic[train_idx] < 0.5) & (inf[train_idx] == 1)
    rows = train_idx[score]

    if a.mode == "counts":
        # Closed-form discrete URR: per action a, w(a) = p_nu(a) / p_train(a) with smoothing. 
        # nu = the design (unthinned) pool, so on a tiered split this estimates exactly the design weights from data.
        nu = np.load(os.path.join(nz, a.nu_rows)).astype(np.int64)
        nu = nu[(ic[nu] < 0.5) & (inf[nu] == 1)]
        key = comp.astype(np.int64) * 100000 + np.round(lx * 1000).astype(np.int64)
        acts, tr_cnt = np.unique(key[rows], return_counts=True)
        nu_map = dict(zip(*np.unique(key[nu], return_counts=True)))
        ks, A = a.smooth_k, len(acts)
        p_tr = (tr_cnt + ks) / (len(rows) + ks * A)
        p_nu = (np.array([nu_map.get(k, 0) for k in acts]) + ks) / (len(nu) + ks * A)
        w_by_act = dict(zip(acts, (p_nu / p_tr).astype(np.float32)))
        out = np.array([w_by_act[k] for k in key[rows]], dtype=np.float32)
        print(f"[export] mode=counts: {A} actions, nu={len(nu):,} rows, smooth_k={ks}")
    else:
        nets = {f: AlphaNet.load(os.path.join(nz, f"{a.prefix}_fold{f}.pt"),  map_location=dev).to(dev).eval() for f in (0, 1)}
        row_fold = folds[rows]
        out = np.empty(len(rows), dtype=np.float32)
        with torch.no_grad():
            for f in (0, 1):
                sel = np.where(row_fold == f)[0]
                # cross-fit: fold-f rows scored by the net fit on the OTHER fold
                net = nets[1 - f]
                for s in range(0, len(sel), a.batch):
                    r = rows[sel[s:s + a.batch]]
                    out[sel[s:s + a.batch]] = net(
                        torch.tensor(cov[r], device=dev),
                        torch.tensor(comp[r], device=dev),
                        torch.tensor(lx[r], device=dev),
                        torch.tensor(ic[r], device=dev),
                        torch.tensor(inf[r], device=dev),
                    ).squeeze(-1).float().cpu().numpy()
        n_unassigned = int((row_fold < 0).sum())
        if n_unassigned:
            print(f"[export] WARNING: {n_unassigned} scored rows have fold -1 (scored by fold-1 net)")
    if a.clip > 0:
        n_clip = int((out > a.clip).sum())
        out = np.minimum(out, a.clip)
        print(f"[export] clipped {n_clip} weights at {a.clip}")
    w[score] = out

    dst = os.path.join(nz, a.out)
    np.savez(dst, row_id=train_idx, w=w)
    nz1 = w[np.abs(w - 1.0) > 1e-6]
    print(f"[export] wrote {dst}  n={len(w):,}  scored={int(score.sum()):,}  w!=1: {len(nz1):,}")
    print(f"[export] w: mean {w.mean():.4f}  min {w.min():.3f}  max {w.max():.3f}  ESS/n {ess(w.astype(np.float64))/len(w):.3f}")

    # validation against the design truth, when present
    dpath = os.path.join(nz, "dr_weights_design.npz")
    if os.path.isfile(dpath):
        dz = np.load(dpath)
        assert np.array_equal(dz["row_id"], train_idx)
        wd = dz["w"].astype(np.float64)
        m = np.abs(wd - 1.0) > 1e-6
        c = float(np.corrcoef(w[m], wd[m])[0, 1]) if m.sum() > 2 else float("nan")
        print(f"[export] vs design (on {int(m.sum()):,} design-weighted rows): "
              f"corr {c:+.3f}  learned mean there {w[m].mean():.3f} "
              f"vs design {wd[m].mean():.3f}")


if __name__ == "__main__":
    main()
