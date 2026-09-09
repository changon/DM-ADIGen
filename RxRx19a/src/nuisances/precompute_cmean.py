"""Precompute mu_hat(a) = E[z | a] in NORMALISED latent space, for the tau=0 conditional-mean auxiliary loss.

TRAINING ROWS ONLY.

    python -m src.nuisances.precompute_cmean [--out cmean.npz]

Writes to the nuisance dir: keys, mu (n_groups, C, H, W) float32, counts, and the train_idx it was built from (train_diffusion checks it).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    LatentSpec, default_latent_path, normalize_latents, open_latents)
from src.data.splits import load_splits  # noqa: E402
from src.spec import add_adjustment_set_cli, config_from_args  # noqa: E402


def group_key(comp: np.ndarray, lx: np.ndarray, ic: np.ndarray) -> np.ndarray:
    """One key per ACTION: (compound, dose, is_control). Controls collapse to a single dose-free key, matching how the model sees them (dose = NaN)."""
    d = np.where(ic == 1, np.nan, np.round(lx, 3))
    return np.array([f"{int(c)}|{'ctl' if int(k) == 1 else f'{v:+.3f}'}" for c, v, k in zip(comp, d, ic)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="cmean.npz")
    p.add_argument("--min_n", type=int, default=8, help="Skip actions with fewer than this many TRAIN rows -- their mean is noisier than the effect it supervises.")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--nuisance_dir", type=str, default="", help="Prebuilt split dir (build_tiered_split.py). Supplies splits.json and receives the output npz.")
    add_adjustment_set_cli(p)
    a = p.parse_args()
    cfg = config_from_args(a)
    if a.nuisance_dir:
        if not os.path.isfile(os.path.join(a.nuisance_dir, "splits.json")):
            raise FileNotFoundError(f"{a.nuisance_dir} has no splits.json")
        cfg.paths.nuisance_dir = a.nuisance_dir

    # load 
    meta = load_from_disk(cfg.paths.tabular_dataset_dir)
    comp = np.asarray(meta["compound_idx"], dtype=np.int64)
    lx = np.asarray(meta["log10_conc"], dtype=np.float64)
    ic = np.asarray(meta["is_control"], dtype=np.int64)

    # align with training procedure
    train_idx = np.asarray(load_splits(cfg)["train_idx"], dtype=np.int64)
    keys_all = group_key(comp, lx, ic)
    kt = keys_all[train_idx]
    uniq, inv = np.unique(kt, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    print(f"[cmean] train rows {len(train_idx):,}  actions {len(uniq):,}  "
          f"n/action min {counts.min()} median {int(np.median(counts))} "
          f"max {counts.max()}", flush=True)

    # get latent
    spec = LatentSpec.load(default_latent_path(cfg))
    z = open_latents(spec)
    shp = z.shape[1:]
    print(f"[cmean] latents {z.shape} normalized={spec.normalized}", flush=True)

    # train_idx is already sorted, so row order and `inv` order agree. train restriction here.
    assert np.all(np.diff(train_idx) > 0), "train_idx must be sorted ascending"
    D = int(np.prod(shp))
    acc = torch.zeros((len(uniq), D), dtype=torch.float32)
    inv_t = torch.from_numpy(inv.astype(np.int64))
    t0 = time.time()

    # iterate, and collect
    for s in range(0, len(train_idx), a.batch):
        rows = train_idx[s:s + a.batch]
        zz = torch.from_numpy(np.asarray(z[rows], dtype=np.float32))
        zz = normalize_latents(zz, spec).reshape(len(rows), D)
        acc.index_add_(0, inv_t[s:s + a.batch], zz)
        if (s // a.batch) % 5 == 0:
            done = s + len(rows)
            print(f"[cmean] {done:,}/{len(train_idx):,}   {done/max(time.time()-t0,1e-9):.0f} rows/s", flush=True)

    # mean comp
    mu = (acc / torch.from_numpy(counts).float()[:, None]).reshape( (-1,) + tuple(shp)).numpy().astype(np.float32)
    keep = counts >= a.min_n
    print(f"[cmean] keeping {int(keep.sum()):,}/{len(uniq):,} actions "
          f"with >= {a.min_n} rows", flush=True)

    dst = os.path.join(cfg.paths.nuisance_dir, a.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, keys=uniq[keep], mu=mu[keep], counts=counts[keep], train_idx=train_idx, min_n=a.min_n)
    print(f"[cmean] wrote {dst}  mu {mu[keep].shape}", flush=True)


if __name__ == "__main__":
    main()
