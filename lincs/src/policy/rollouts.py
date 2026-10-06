"""Rollouts of one trained generator at every decision (POLICY_LEARNING.md §7, P5). Needs a GPU.

For every context (compound, line) and every dose level of the compound, draws
L samples from the generator and saves their mean over L (and the per-sample
norm of the vehicle-centred sample, for a nonlinear utility). The generated
vehicle mean of each line comes from real vehicle rows of that line, sampled
through the same path `evaluate` uses. Everything downstream is numpy
(`src.policy.learn`).

The arm's provenance is checked against the BASE split, as `evaluate` does for a
tiered arm (its unthinned pool plus the holdout must reproduce the base split).

    python -m src.policy.rollouts --data_dir data/core5_24h --run_dir runs/core5_24h/<run> --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import CONTEXT_SOURCE_COLUMNS, _atomic_write, context_for_rows  # noqa: E402
from src.data.expr_stats import load_expr_meta  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.models import read_arch_spec  # noqa: E402
from src.policy.targets import ROLLOUTS_NAME, decision_targets  # noqa: E402
from src.spec import add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args  # noqa: E402

TARGET_ID_BASE = 10_000_000          # noise seeds of generated decisions: never a table row id


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--gen_epoch", type=int, default=499)
    p.add_argument("--which_wgt", default="ema", choices=("train", "ema"))
    p.add_argument("--L", type=int, default=8, help="Samples per (context, dose).")
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--n_vehicle", type=int, default=64, help="Real vehicle rows per line to sample the generated vehicle at.")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help=f"Default <run_dir>/eval_artifacts/{ROLLOUTS_NAME}")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    t0 = time.time()
    import torch

    from datasets import load_from_disk

    from src.eval.generation import (SampleTarget, check_arm_against_data, generate_batch, load_arm, row_seed,
                                     targets_from_rows)

    run_dir = os.path.abspath(args.run_dir)
    arch = read_arch_spec(os.path.join(run_dir, f"checkpoint-{args.gen_epoch:04d}"))
    if arch is None:
        raise SystemExit(f"[rollouts] no arch.json under {run_dir}")
    if args.adjustment_set is None:        # the arm's own C, so the launcher need not know the arm
        args.adjustment_set = ",".join(arch.get("adjustment_set") or ())
    cfg = apply_paths_args(config_from_args(args), args)
    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[rollouts] --device cuda but CUDA is unavailable on this node")
    splits = load_splits(cfg)                                   # the BASE split
    expr_meta = load_expr_meta(cfg, splits=splits)
    with open(os.path.join(cfg.paths.nuisance_dir, "nuisance_meta.json")) as fh:
        n_comp = int(json.load(fh)["n_compounds"])
    model, scheduler, arch, ckpt = load_arm(run_dir, epoch=args.gen_epoch, which=args.which_wgt,
                                            device=dev, sampler="ddim", n_inference_steps=args.num_inference_steps)
    checked = check_arm_against_data(arch, cfg, splits, expr_meta, n_compounds=n_comp)
    print(f"[rollouts] {run_dir} ({args.which_wgt}, epoch {args.gen_epoch}) checked against {cfg.paths.data_dir}", flush=True)

    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(list(dict.fromkeys(["compound_idx", "dose_level", "log10_conc", "is_control", "cell_id"]
                                               + list(CONTEXT_SOURCE_COLUMNS)))).to_pandas())
    T = decision_targets(meta)
    ctx_codes = context_for_rows(cfg, meta, T["ctx_row"])
    targets = [SampleTarget(int(T["ctx_compound"][c]), float(x), 0, tuple(int(v) for v in ctx_codes[c]))
               for c, x in zip(T["t_ctx"], T["t_log10_conc"])]
    n_t = len(targets)
    lines = sorted(set(T["ctx_line"].tolist()))
    print(f"[rollouts] {T['ctx_compound'].size:,} contexts over {len(lines)} lines, {n_t:,} (context, dose) "
          f"targets x L = {args.L} = {n_t * args.L:,} samples", flush=True)

    # ---- the generated vehicle of each line, from real vehicle rows ------------------
    ctl = meta["is_control"].values.astype(bool)
    line_all = meta["cell_id"].values.astype(str)
    rng = np.random.default_rng([args.seed, 4_242])
    veh_mean = np.zeros((len(lines), int(model.n_genes)), dtype=np.float64)
    veh_n = np.zeros(len(lines), dtype=np.int64)
    for li, l in enumerate(lines):
        rows = np.flatnonzero(ctl & (line_all == l))
        if rows.size == 0:
            raise RuntimeError(f"line {l} has no vehicle row")
        pick = np.sort(rng.choice(rows, min(args.n_vehicle, rows.size), replace=False))
        vt = targets_from_rows(cfg, meta, pick)
        tot = np.zeros(int(model.n_genes), dtype=np.float64)
        for rep in range(args.L):
            x = generate_batch(model, scheduler, vt, args.num_inference_steps, dev, seed=args.seed,
                               row_seeds=[row_seed(args.seed, int(r), rep) for r in pick])
            if not torch.isfinite(x).all():
                raise RuntimeError("non-finite vehicle samples")
            tot += x.to(torch.float64).cpu().numpy().sum(axis=0)
        veh_mean[li] = tot / (pick.size * args.L)
        veh_n[li] = pick.size * args.L
    line_of_ctx = np.array([lines.index(l) for l in T["ctx_line"]], dtype=np.int64)

    # ---- the decisions ------------------------------------------------------------------
    G = int(model.n_genes)
    tot = np.zeros((n_t, G), dtype=np.float64)
    norm = np.zeros((n_t, args.L), dtype=np.float32)
    n_chunks = (n_t + args.batch_size - 1) // args.batch_size
    for rep in range(args.L):
        for ci in range(n_chunks):
            lo, hi = ci * args.batch_size, min((ci + 1) * args.batch_size, n_t)
            seeds = [row_seed(args.seed, TARGET_ID_BASE + i, rep) for i in range(lo, hi)]
            x = generate_batch(model, scheduler, targets[lo:hi], args.num_inference_steps, dev,
                               seed=args.seed, row_seeds=seeds)
            if not torch.isfinite(x).all():
                raise RuntimeError(f"non-finite samples at rep {rep} chunk {ci}")
            xn = x.to(torch.float64).cpu().numpy()
            tot[lo:hi] += xn
            cen = xn - veh_mean[line_of_ctx[T["t_ctx"][lo:hi]]]
            norm[lo:hi, rep] = np.sqrt((cen ** 2).sum(axis=1)).astype(np.float32)
        print(f"[rollouts] rep {rep + 1}/{args.L} done ({time.time() - t0:.0f}s)", flush=True)
    out = args.out or os.path.join(run_dir, "eval_artifacts", ROLLOUTS_NAME)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    label = {"run_dir": run_dir, "run": os.path.basename(run_dir), "arch": arch.get("arch"),
             "dr_mode": arch.get("dr_mode"), "dr_weights_file": arch.get("dr_weights_file"),
             "dr_weight_norm": (arch.get("train_args") or {}).get("dr_weight_norm"),
             "nuisance_dir": arch.get("nuisance_dir"), "split_fingerprint": arch.get("split_fingerprint"),
             "tier": arch.get("tier"), "adjustment_set": list(arch.get("adjustment_set") or ()),
             "gen_epoch": args.gen_epoch, "which_wgt": args.which_wgt, "L": args.L,
             "num_inference_steps": args.num_inference_steps, "seed": args.seed, "checked": checked,
             "elapsed_sec": round(time.time() - t0, 1)}
    _atomic_write(out, lambda f: np.savez(
        f, ctx_compound=T["ctx_compound"], ctx_line=T["ctx_line"], ctx_row=T["ctx_row"],
        t_ctx=T["t_ctx"], t_dose_index=T["t_dose_index"], t_dose_level=T["t_dose_level"],
        t_log10_conc=T["t_log10_conc"], mean=(tot / args.L).astype(np.float32), norm_centred=norm,
        lines=np.array(lines), vehicle_mean=veh_mean.astype(np.float32), vehicle_n=veh_n,
        label=np.array(json.dumps(label))), mode="wb")
    print(f"[rollouts] -> {out}  ({label['elapsed_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
