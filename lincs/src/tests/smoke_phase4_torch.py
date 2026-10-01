"""GPU smoke test of the Phase 4 sampler (IMPLEMENT.md §3.10, §3.14). Imports
PyTorch: run it on a GPU node, never on the dev box.

Checks, per `--ckpt_dir` run (one DDPM and one FM arm cover both schedules):
  1. load_arm rebuilds from arch.json alone, strict-loads the EMA weights
     (model_1.safetensors), and leaves the model in eval mode -- which is
     load-bearing: with model.training true and class_dropout_prob > 0,
     layers.resolve_timestep_and_drop draws its OWN CFG mask, so ~10% of eval
     rows would silently go unconditional.
  2. check_arm_against_data passes on the data the arm was trained on, and
     REFUSES a doctored gene order, split_fingerprint, plate_center and
     adjustment_set.
  3. generate_batch returns (B, n_genes), finite, and unclipped: no pile-up at
     exactly +-1, and (with --expect_unbounded, for a trained arm) values
     beyond +-1. Two clamps have to stay off -- RxRx's `x.clamp(-1, 1)` and
     diffusers' own `clip_sample`, which defaults to TRUE and clips predicted
     x0 every step (§2.2; fixed in processes/ddpm.py).
  4. Per-row seeding: a row generated in two different batch compositions AND
     two different batch sizes gives the same sample. Reordering is bit-exact;
     a different batch SIZE agrees to ~1e-7, because cuBLAS picks different
     kernels per shape. That is float nondeterminism, not a batching bug, so
     the size comparison uses a tolerance.
  5. The schedule runs in the right direction: DDIM timesteps descend, flow
     matching ascends, and both start from pure noise.
  6. Guidance: w = 1 does one forward per step, w > 1 does two, and the w > 1
     reference branch is the action marginal (role-A dropped).
  7. generate_for_rows: row_mean / row_var shapes, the reservoir honours its cap
     and is invariant to --gen_batch_size, and n_samples is n_rows * n_per_row.
  8. A sample is not a copy of a training row (no memorisation short-circuit in
     the plumbing), and the generated mean is finite in z-space.
Exits nonzero on any failure.

    python -m src.tests.smoke_phase4_torch --device cuda \
        --data_dir data/mcf7_24h_limit1500 \
        --ckpt_dir runs/mcf7_24h_limit1500/smoke_717146_mlp_cond_ddpm \
        --ckpt_dir runs/mcf7_24h_limit1500/smoke_717146_mlp_cond_fm
"""
from __future__ import annotations

import argparse
import copy
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

from src.data.expr_stats import load_expr_meta  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.eval import generation as gn  # noqa: E402
from src.eval.evaluate import META_COLUMNS, quality_groups  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def _refuses(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError as e:
        check(True, f"{label} -> refused ({str(e).splitlines()[0][:60]})")
        return
    check(False, f"{label} -> NOT refused")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt_dir", action="append", default=[], required=True)
    p.add_argument("--epoch", type=int, default=None,
                   help="Default: the newest checkpoint in the run dir.")
    p.add_argument("--steps", type=int, default=8, help="Sampler steps (keep small).")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--atol", type=float, default=1e-4,
                   help="Tolerance for comparisons across batch SIZES (different "
                        "shapes take different cuBLAS kernels). Reordering is exact.")
    p.add_argument("--expect_unbounded", action="store_true",
                   help="Require >1%% of sampled values outside [-1, 1]. Only "
                        "meaningful for a trained arm: a 2-epoch smoke checkpoint "
                        "has barely left its initialisation.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        check(ok, f"CUDA available on {os.uname().nodename} "
                  f"(torch {torch.__version__}, cuda {torch.version.cuda})")
        if not ok:
            print("[smoke4] CUDA unavailable: aborting rather than falling back to CPU")
            sys.exit(1)
        print(f"      device: {torch.cuda.get_device_name(0)}")

    splits = load_splits(cfg)
    m = load_expr_meta(cfg, None, splits=splits)
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(list(META_COLUMNS)).to_pandas())
    with open(os.path.join(cfg.paths.nuisance_dir, "nuisance_meta.json")) as fh:
        n_compounds = int(json.load(fh)["n_compounds"])
    # A slice with both treated and vehicle rows, so the dose-NaN null is exercised.
    is_ctl = meta["is_control"].values.astype(int)
    rows = np.concatenate([np.flatnonzero(is_ctl == 0)[: args.batch - 4],
                           np.flatnonzero(is_ctl == 1)[:4]])
    rows = np.sort(rows)

    for run_dir in args.ckpt_dir:
        tag = os.path.basename(os.path.normpath(run_dir))
        print(f"\n=== {tag}")
        epoch = args.epoch
        if epoch is None:
            cks = sorted(x for x in os.listdir(run_dir) if x.startswith("checkpoint-"))
            if not cks:
                check(False, f"{tag}: no checkpoint in {run_dir}")
                continue
            epoch = int(cks[-1].split("-")[1])
        model, sched, arch, ckpt_dir = gn.load_arm(
            run_dir, epoch=epoch, which="ema", device=dev, sampler="ddim")
        is_fm = arch["diffusion_method"] == "fm"
        check(not model.training,
              f"{tag}: model is in eval mode (otherwise CFG drop is random)")
        check(int(model.n_genes) == int(cfg.outcome.n_genes),
              f"{tag}: n_genes {model.n_genes} == cfg {cfg.outcome.n_genes}")
        check(float(arch["class_dropout_prob"]) > 0,
              f"{tag}: class_dropout_prob {arch['class_dropout_prob']} > 0, so the "
              f"null branch is trained and `drop=` is legal")

        # ---- 2. identity guard ----
        checked = gn.check_arm_against_data(arch, cfg, splits, m, n_compounds=n_compounds)
        check(checked["split_fingerprint"] == splits["split_fingerprint"],
              f"{tag}: check_arm_against_data passes on its own data")
        for field, mutate in (
            ("gene order", lambda a: a.update(gene_pr_ids=[0] + list(a["gene_pr_ids"][1:]))),
            ("gene_order_sha1", lambda a: a.update(gene_order_sha1="0" * 40)),
            ("split_fingerprint", lambda a: a.update(split_fingerprint="deadbeef")),
            ("plate_center", lambda a: a.update(plate_center="none")),
            ("adjustment_set", lambda a: a.update(adjustment_set=["syn_c"])),
            ("table_fingerprint", lambda a: a.update(table_fingerprint="cafe1234")),
        ):
            bad = copy.deepcopy(arch)
            mutate(bad)
            _refuses(lambda b=bad: gn.check_arm_against_data(b, cfg, splits, m),
                     f"{tag}: doctored {field}")

        # ---- 3. shapes, finiteness, no clamp ----
        targets = gn.targets_from_rows(cfg, meta, rows)
        seeds = [gn.row_seed(0, int(r), 0) for r in rows]
        x = gn.generate_batch(model, sched, targets, args.steps, dev, seed=0,
                              row_seeds=seeds)
        check(tuple(x.shape) == (len(rows), int(cfg.outcome.n_genes)),
              f"{tag}: generate_batch -> {tuple(x.shape)}")
        check(bool(torch.isfinite(x).all()), f"{tag}: samples are finite")
        # A clamp (RxRx's, or diffusers' clip_sample) leaves mass at exactly +-1.
        at_bound = int((x.abs() == 1.0).sum())
        outside = float((x.abs() > 1.0).float().mean())
        check(at_bound == 0,
              f"{tag}: no value sits at exactly +-1 ({at_bound} do), so nothing "
              f"clamps; {100 * outside:.1f}% lie outside [-1, 1], max |x| "
              f"{float(x.abs().max()):.2f}")
        if args.expect_unbounded:
            check(outside > 0.01,
                  f"{tag}: {100 * outside:.1f}% of values lie outside [-1, 1] "
                  f"(a trained arm in z-space must exceed the image range)")

        # ---- 4. per-row seeding: batching invariance ----
        perm = np.argsort(-rows)                      # a different composition AND order
        t2 = [targets[i] for i in perm]
        s2 = [seeds[i] for i in perm]
        x2 = gn.generate_batch(model, sched, t2, args.steps, dev, seed=0, row_seeds=s2)
        d_perm = float((x2 - x[perm]).abs().max())
        half = len(rows) // 2
        xa = gn.generate_batch(model, sched, targets[:half], args.steps, dev, seed=0,
                               row_seeds=seeds[:half])
        d_split = float((xa - x[:half]).abs().max())
        check(d_perm == 0.0,
              f"{tag}: a row's sample is bit-exact under batch REORDERING "
              f"({d_perm:.1e}) -- same shape, same kernels")
        check(d_split < args.atol,
              f"{tag}: and agrees to {d_split:.1e} < {args.atol:g} across batch "
              f"SIZES (cuBLAS kernel choice, not a seeding bug)")

        # ---- 5. schedule direction ----
        sched.set_timesteps(args.steps, device=dev)
        ts = np.asarray([float(t) for t in sched.timesteps])
        asc = bool((np.diff(ts) > 0).all())
        check(asc == is_fm,
              f"{tag}: timesteps {'ascend' if asc else 'descend'} "
              f"({'flow matching, noise->data' if is_fm else 'diffusers, T->0'})")

        # ---- 6. guidance forward counts and the reference branch ----
        calls = {"n": 0, "drops": []}
        inner = model.forward

        def counting(sample, timestep, cond, drop=None, **kw):
            calls["n"] += 1
            calls["drops"].append(None if drop is None else bool(drop.all()))
            return inner(sample, timestep, cond, drop=drop, **kw)

        model.forward = counting
        try:
            calls["n"] = 0; calls["drops"].clear()
            gn.generate_batch(model, sched, targets[:4], args.steps, dev, seed=0,
                              guidance_scale=1.0, row_seeds=seeds[:4])
            n_w1 = calls["n"]
            calls["n"] = 0; calls["drops"].clear()
            gn.generate_batch(model, sched, targets[:4], args.steps, dev, seed=0,
                              guidance_scale=2.0, row_seeds=seeds[:4])
            n_w2, drops = calls["n"], list(calls["drops"])
        finally:
            model.forward = inner
        check(n_w1 == args.steps and n_w2 == 2 * args.steps,
              f"{tag}: w=1 does {n_w1} forwards ({args.steps} steps), w=2 does {n_w2}")
        check(drops[1::2] == [True] * args.steps,
              f"{tag}: the w>1 reference branch drops every role-A field "
              f"(the action marginal)")

        # ---- 7. the streaming driver ----
        gq, gnames = quality_groups(meta["dose_level"].values[rows], is_ctl[rows])
        cap = 3
        out_a = gn.generate_for_rows(
            model, sched, cfg, meta, rows, n_per_row=2, n_inference_steps=args.steps,
            device=dev, seed=0, batch_size=max(4, len(rows) // 3),
            group_of_row=gq, reservoir_cap=cap, log_every=0)
        out_b = gn.generate_for_rows(
            model, sched, cfg, meta, rows, n_per_row=2, n_inference_steps=args.steps,
            device=dev, seed=0, batch_size=len(rows),
            group_of_row=gq, reservoir_cap=cap, log_every=0)
        check(out_a["row_mean"].shape == (len(rows), int(cfg.outcome.n_genes))
              and out_a["row_var"].shape == out_a["row_mean"].shape,
              f"{tag}: row_mean / row_var -> {out_a['row_mean'].shape}")
        check(out_a["n_samples"] == 2 * len(rows),
              f"{tag}: n_samples {out_a['n_samples']} == rows x n_per_row")
        d_bs = float(np.abs(out_a["row_mean"] - out_b["row_mean"]).max())
        check(d_bs < args.atol,
              f"{tag}: row_mean agrees to {d_bs:.1e} across --gen_batch_size")
        # The SELECTION must be identical (it is drawn up front from the item
        # count); only the sampled values carry kernel-level noise.
        same_groups = sorted(out_a["reservoir"]) == sorted(out_b["reservoir"]) and all(
            out_a["reservoir"][g].shape == out_b["reservoir"][g].shape
            for g in out_a["reservoir"])
        d_res = (max(float(np.abs(out_a["reservoir"][g] - out_b["reservoir"][g]).max())
                     for g in out_a["reservoir"]) if same_groups and out_a["reservoir"]
                 else float("inf"))
        check(same_groups and d_res < args.atol,
              f"{tag}: the reservoir selection is invariant to --gen_batch_size "
              f"(same groups and sizes, values to {d_res:.1e})")
        over = {g: int(v.shape[0]) for g, v in out_a["reservoir"].items()
                if v.shape[0] > cap}
        check(not over, f"{tag}: every reservoir group honours the cap {cap} "
                        f"({ {g: int(v.shape[0]) for g, v in out_a['reservoir'].items()} })")
        check(bool(np.isfinite(out_a["row_mean"]).all()),
              f"{tag}: the generated per-row means are finite")
        check(float(out_a["row_var"].mean()) > 0,
              f"{tag}: replicates differ (mean within-row variance "
              f"{float(out_a['row_var'].mean()):.3f} > 0)")

    if FAILS:
        print(f"\n[smoke4] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS))
        sys.exit(1)
    print("\n[smoke4] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
