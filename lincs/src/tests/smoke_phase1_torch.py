"""GPU smoke test of the Phase 1 torch path (IMPLEMENT.md §3.10). Imports PyTorch:
run it on a GPU node, never on the dev box.

Checks, on the v1 artifacts of a build (build_tiered_split + expr_stats; the
last check also needs fit_urr + export_urr_weights --mode net):
  1. LincsDataset: shapes, dtypes, z-space (train DMSO mean 0, train std 1),
     rows = an independent normalisation of expr.npy, vehicle dose sentinel;
     load_y=False path; the holdout loads.
  2. DataLoader batch -> build_cond_spec (default, include_env, --adjustment_set
     syn_c) -> cond_from_batch (vehicle dose NaN) -> CondEmbedder on the device
     (finite, with and without the CFG drop mask) + calibrate(dose_probe).
  3. AlphaNet without `infected`: in_dim, save/load round trip; urr_loss on
     the device recovers the batch's known answer (nu = the batch's treated
     rows: alpha -> B / B_treated on treated rows, -> 0 on vehicles).
  4. The exported dr_weights_urr.npz re-scored on the device with the
     cross-fitted nets (fold f by fold{f}.pt), matching the CPU export and
     fit_urr's out-of-fold means.
Exits nonzero on any failure.

    python -m src.tests.smoke_phase1_torch --device cuda [--data_dir ...] [--nuisance_dir ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    LincsDataset, build_cond_spec, cond_from_batch, dose_probe, get_dataloader)
from src.data.splits import load_splits  # noqa: E402
from src.models.conditioning import CondEmbedder  # noqa: E402
from src.nuisances.alpha_net import AlphaNet  # noqa: E402
from src.nuisances.fit_urr import urr_loss  # noqa: E402
from src.spec import (  # noqa: E402
    CONTEXT_FIELDS, add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=256)
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[smoke1] --device cuda but CUDA is unavailable")
    dev = torch.device(args.device)
    torch.manual_seed(0)
    print(f"[smoke1] torch {torch.__version__} device {dev}"
          + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""))
    nz = cfg.paths.nuisance_dir
    s = load_splits(cfg)
    tr, ho = s["train_idx"], s["holdout_idx"]
    n_compounds = json.load(open(os.path.join(nz, "nuisance_meta.json")))["n_compounds"]

    # === 1. dataset ===========================================================
    ds = LincsDataset(cfg, indices=tr)
    y = ds.y.numpy()
    G = cfg.outcome.n_genes
    check(len(ds) == tr.size and y.shape == (tr.size, G) and y.dtype == np.float32,
          f"train dataset: {len(ds):,} rows, y {y.shape} {y.dtype}")
    check(tuple(ds.context.shape) == (tr.size, len(CONTEXT_FIELDS)) and ds.context.dtype == torch.int64
          and ds.cov_vec.shape[0] == tr.size, f"context {tuple(ds.context.shape)}, cov_vec {tuple(ds.cov_vec.shape)}")
    ctl = ds.is_control.numpy().astype(bool)
    dmean = np.abs(y[ctl].astype(np.float64).mean(0)).max()
    sd = y.astype(np.float64).std(0)
    check(dmean < 1e-4 and np.abs(sd - 1).max() < 1e-3,
          f"z-space: train DMSO mean max |.| {dmean:.1e}, train std in [{sd.min():.5f}, {sd.max():.5f}]; "
          f"range [{y.min():.1f}, {y.max():.1f}] (unclamped)")
    em = json.load(open(os.path.join(nz, "expr_meta.json")))
    expr = np.load(cfg.paths.expr_npy, mmap_mode="r")
    pick = np.random.default_rng(0).choice(tr.size, size=min(64, tr.size), replace=False)
    code = {p_: i for i, p_ in enumerate(em["plates"])}
    from datasets import load_from_disk
    det_plate = np.asarray(load_from_disk(cfg.paths.tabular_dataset_dir)["det_plate"], dtype=object)
    cen, mu, sdv = np.asarray(em["centre"]), np.asarray(em["mean"]), np.asarray(em["std"])
    want = np.stack([(np.asarray(expr[tr[i]], dtype=np.float64) - cen[code[det_plate[tr[i]]]] - mu) / sdv for i in pick])
    check(np.abs(y[pick] - want).max() < 1e-4,
          f"y rows = (x - centre[plate] - mean) / std recomputed from expr.npy ({pick.size} rows)")
    check(bool((ds.log10_conc[ds.is_control.bool()] == cfg.action.control_log10_sentinel).all())
          and bool((ds.compound_idx[ds.is_control.bool()] == 0).all()),
          "vehicle rows: compound_idx 0, log10_conc sentinel in the table")
    ds_noy = LincsDataset(cfg, indices=ho, load_y=False)
    ds_ho = LincsDataset(cfg, indices=ho)
    check(ds_noy.y is None and "y" not in ds_noy[0] and bool(torch.isfinite(ds_ho.y).all()),
          f"load_y=False skips Y; holdout ({len(ds_ho):,} rows) loads finite")

    # === 2. batch -> cond spec -> embedder ======================================
    dl = get_dataloader(cfg, args.batch, indices=tr, shuffle=True)
    batch = next(iter(dl))
    check(set(batch) >= {"y", "compound_idx", "log10_conc", "is_control", "cov_vec", "context", "row_id"}
          and tuple(batch["y"].shape) == (args.batch, G), f"DataLoader batch keys {sorted(batch)}, y {tuple(batch['y'].shape)}")
    lx_tr = ds.log10_conc.numpy()[~ctl]
    spec = build_cond_spec(cfg, n_compounds, lx_tr)
    by = {f.name: f for f in spec}
    check(list(spec.names) == ["compound", "is_control", "dose"] and by["compound"].cardinality == n_compounds
          and by["dose"].kind == "cont" and by["dose"].nullable,
          f"cond spec (v1): {[(f.name, f.role, f.kind, f.cardinality) for f in spec]}")
    spec_e = build_cond_spec(cfg, n_compounds, lx_tr, include_env=True)
    spec_c = build_cond_spec(cfg, n_compounds, lx_tr, adjustment_set=("syn_c",))
    check([(f.name, f.role, f.cardinality) for f in spec_e if f.role == "E"] == [("plate", "E", len(em["plates"]))]
          and [(f.name, f.role, f.cardinality) for f in spec_c if f.role == "C"] == [("syn_c", "C", 2)],
          "include_env adds plate (E); --adjustment_set syn_c adds syn_c (C)")
    for sp, tag in ((spec, "v1"), (spec_e, "env"), (spec_c, "C=syn_c")):
        cond = cond_from_batch(batch, sp, device=dev)
        vc = batch["is_control"].bool().to(dev)
        nan_ok = bool(torch.isnan(cond["dose"][vc]).all()) and bool(torch.isfinite(cond["dose"][~vc]).all())
        emb = CondEmbedder(sp, 64).to(dev)
        emb.init_weights()
        fac = emb.calibrate(dose_probe(lx_tr))
        c0 = emb(cond)
        c1 = emb(cond, drop=torch.rand(args.batch, device=dev) < 0.5)
        check(nan_ok and tuple(c0.shape) == (args.batch, 64) and bool(torch.isfinite(c0).all())
              and bool(torch.isfinite(c1).all()) and c0.device.type == dev.type,
              f"{tag}: cond_from_batch (vehicle dose NaN) -> CondEmbedder {tuple(c0.shape)} on {c0.device} "
              f"finite with and without CFG drop; calibrate {fac}")

    # === 3. AlphaNet + URR loss ===================================================
    cov_dim = ds.cov_vec.shape[1]
    net = AlphaNet(n_compounds=n_compounds, cov_dim=cov_dim, cov_idx=[], positive=True).to(dev)
    first = net.mlp[0]
    B = batch["compound_idx"].shape[0]
    args_net = (batch["cov_vec"].to(dev), batch["compound_idx"].to(dev),
                batch["log10_conc"].to(dev), batch["is_control"].float().to(dev))
    a0 = net(*args_net)
    check(first.in_features == 0 + 32 + 2 and tuple(a0.shape) == (B,),
          f"AlphaNet without infected: in_features {first.in_features} (= |X| 0 + embed 32 + 2), alpha {tuple(a0.shape)}")
    with tempfile.TemporaryDirectory() as td:
        net.save(os.path.join(td, "a.pt"))
        net2 = AlphaNet.load(os.path.join(td, "a.pt"), map_location=dev).to(dev)
        check(torch.allclose(net2(*args_net), a0), "AlphaNet save / load round trip")
    # nu = the batch's treated rows, so the minimiser is B / B_treated on treated rows, 0 on vehicles
    veh = args_net[3] > 0.5
    trt_rows = torch.nonzero(~veh).squeeze(-1)
    M = 8
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.0)
    gen = torch.Generator(device=dev).manual_seed(0)
    losses = []
    for _ in range(400):
        jt = trt_rows[torch.randint(0, trt_rows.numel(), (B, M), device=dev, generator=gen)]
        loss, _ = urr_loss(net, *args_net, args_net[1][jt], args_net[2][jt], args_net[3][jt])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    with torch.no_grad():
        a1 = net(*args_net)
    want_t = B / float(trt_rows.numel())
    check(all(np.isfinite(losses)) and abs(float(a1[~veh].mean()) / want_t - 1) < 0.03
          and float(a1[veh].mean()) < 0.1,
          f"urr_loss on {dev} recovers the known answer: alpha treated {float(a1[~veh].mean()):.4f} vs "
          f"B/B_treated {want_t:.4f}, vehicle {float(a1[veh].mean()):.4f} vs 0 ({int(veh.sum())} vehicles; "
          f"L {losses[0]:+.4f} -> {losses[-1]:+.4f}, optimum {-want_t:+.4f})")

    # === 4. exported net weights re-scored on the device =========================
    wp = os.path.join(nz, "dr_weights_urr.npz")
    if os.path.isfile(wp) and os.path.isfile(os.path.join(nz, "alpha_urr_fold0.pt")):
        wz = np.load(wp)
        um = json.load(open(os.path.join(nz, "alpha_urr_meta.json")))
        folds = np.load(os.path.join(nz, "alpha_urr_folds.npy"))
        nets = {f: AlphaNet.load(os.path.join(nz, f"alpha_urr_fold{f}.pt"), map_location=dev).to(dev).eval() for f in (0, 1)}
        w = np.ones(tr.size, dtype=np.float32)
        trt = ~ctl
        with torch.no_grad():
            for f in (0, 1):
                m = trt & (folds[tr] == f)
                mt = torch.from_numpy(m)
                w[m] = nets[f](ds.cov_vec[mt].to(dev), ds.compound_idx[mt].to(dev), ds.log10_conc[mt].to(dev),
                                   ds.is_control[mt].float().to(dev)).cpu().numpy()
        err = float(np.abs(w - wz["w"]).max())
        fm = [abs(float(w[trt & (folds[tr] == f)].mean()) - um[f"fold{f}"]["alpha_mean_treated"]) for f in (0, 1)]
        check(np.array_equal(wz["row_id"], tr) and err < 1e-4 and max(fm) < 1e-4,
              f"dr_weights_urr.npz re-scored on {dev} with fold f <- fold{{f}}.pt (max |diff| {err:.1e}); "
              f"per-fold treated means = fit_urr's out-of-fold means (|diff| {max(fm):.1e})")
    else:
        print("skip  no dr_weights_urr.npz / alpha_urr nets yet")

    if FAILS:
        print(f"[smoke1] {len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("[smoke1] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
