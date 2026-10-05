"""Step-A torch-side checks (`STEP_A.md` §5, A8): the cell line as C. Needs a GPU node.

On a `core5_24h` build and one of its tier dirs:

  data      LincsDataset loads the multi-line table; the context carries the
            cell line with one level per line of the population
  roles     build_cond_spec with --adjustment_set cell_id gives cell_id role C
            with that cardinality; without it (the `naive` arm) the generator has
            no line input at all
  CFG       the drop mask nulls the ACTION fields only: with every row dropped,
            changing the cell line still changes the embedding. A droppable C
            would turn the conditional arm into the naive one under guidance
  weights   the tier's counts weights align with its train rows, are exactly 1
            off the scored compounds, and at gamma > 0 are larger on the
            under-kept line group, in opposite directions in the two dose halves
  P2        the target group of every treated train row is its ARM

    python -m src.tests.smoke_phase6_torch --device cuda --data_dir data/core5_24h_limit11000 \
        --tier data/core5_24h_limit11000/nuisances_tier_Ccell_id_k0_g1_s42_G2-A375-HA1E
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

from src.data.build_dataset import ContextEncoder  # noqa: E402
from src.data.dataset import LincsDataset, build_cond_spec, cond_from_batch  # noqa: E402
from src.data.splits import arm_keys, dose_half, load_splits  # noqa: E402
from src.models.conditioning import CondEmbedder  # noqa: E402
from src.nuisances.weight_norm import train_groups  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args,
    line_group_sign, line_groups)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--tier", required=True, help="A step-A tier dir (gamma > 0).")
    p.add_argument("--n_rows", type=int, default=4000)
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    args.adjustment_set = "cell_id"
    args.nuisance_dir = args.tier
    cfg = apply_paths_args(config_from_args(args), args)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        check(ok, f"CUDA available on {os.uname().nodename} (torch {torch.__version__})")
        if not ok:
            print("[smoke6] CUDA unavailable: aborting rather than falling back to CPU")
            sys.exit(1)
    lines = sorted(cfg.population.cell_ids)
    lg = line_groups(cfg.population)
    check(lg is not None and len(lines) > 1, f"population {cfg.population.name}: lines {lines}, groups {lg}")

    # ---- data ----
    splits = load_splits(cfg)
    tr = splits["train_idx"]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(tr, min(args.n_rows, tr.size), replace=False))
    ds = LincsDataset(cfg, indices=idx)
    enc = ContextEncoder.load_or_build(cfg)
    check(sorted(enc.levels["cell_id"]) == lines,
          f"context encoder has one cell_id level per line {enc.levels['cell_id']}")
    tbl = (load_from_disk(cfg.paths.tabular_dataset_dir)
           .select_columns(["compound_idx", "dose_level", "is_control", "cell_id"]).to_pandas())
    check(ds.y.shape == (idx.size, cfg.outcome.n_genes) and bool(torch.isfinite(ds.y).all()),
          f"LincsDataset loads {idx.size:,} train rows of the multi-line table, y finite {tuple(ds.y.shape)}")
    from torch.utils.data import default_collate
    batch = default_collate([ds[i] for i in range(64)])

    # ---- roles ----
    with open(os.path.join(cfg.paths.nuisance_dir, "nuisance_meta.json")) as fh:
        n_comp = int(json.load(fh)["n_compounds"])
    lx = ds.log10_conc.numpy()[ds.is_control.numpy() == 0]
    spec = build_cond_spec(cfg, n_comp, lx, adjustment_set=("cell_id",))
    roles = {f.name: f.role for f in spec}
    card = {f.name: getattr(f, "cardinality", None) for f in spec}
    check(roles.get("cell_id") == "C" and int(card["cell_id"]) == len(lines),
          f"--adjustment_set cell_id: cell_id has role C, cardinality {card.get('cell_id')} (roles {roles})")
    spec_naive = build_cond_spec(cfg, n_comp, lx, adjustment_set=())
    check("cell_id" not in {f.name for f in spec_naive},
          f"the naive arm's spec has no line input (fields {[f.name for f in spec_naive]})")

    # ---- CFG ----
    cond = cond_from_batch(batch, spec, device=dev)
    check("cell_id" in cond and cond["cell_id"].shape[0] == 64,
          f"cond_from_batch carries cell_id {tuple(cond['cell_id'].shape)}")
    codes = cond["cell_id"].detach().cpu().numpy()
    want = np.array([enc.levels["cell_id"].index(c) for c in tbl["cell_id"].values[idx[:64]]])
    check(np.array_equal(codes, want), "the cell_id codes are the rows' own lines (encoder order)")
    emb = CondEmbedder(spec, hidden_size=64).to(dev)
    drop = torch.ones(64, dtype=torch.bool, device=dev)
    e_keep, e_drop = emb(cond, drop=None), emb(cond, drop=drop)
    check(bool(torch.isfinite(e_keep).all()) and bool(torch.isfinite(e_drop).all()),
          "CondEmbedder is finite with and without the CFG drop mask")
    alt = dict(cond)
    alt["cell_id"] = (alt["cell_id"] + 1) % len(lines)
    check(not torch.allclose(emb(alt, drop=drop), e_drop),
          "CFG drop leaves C: changing the cell line still changes the fully dropped embedding")
    alt_a = dict(cond)
    alt_a["compound"] = (alt_a["compound"] + 1) % n_comp
    check(torch.allclose(emb(alt_a, drop=drop), e_drop),
          "CFG drop nulls A: changing the compound does not change the fully dropped embedding")

    # ---- weights on the tier ----
    T = args.tier
    ts = json.load(open(os.path.join(T, "splits.json")))
    tier = ts["tier"]
    wz = np.load(os.path.join(T, "dr_weights_counts.npz"))
    check(np.array_equal(wz["row_id"], tr), f"counts weights row_id == the tier's train_idx ({tr.size:,} rows)")
    comp = tbl["compound_idx"].values.astype(np.int64)
    dl = tbl["dose_level"].values.astype(np.float64)
    ic = tbl["is_control"].values.astype(np.int64)
    scored = np.isin(comp[tr], [int(v) for v in tier["scored_compounds"].values()]) & (ic[tr] == 0)
    w = wz["w"].astype(np.float64)
    check(bool((w[~scored] == 1.0).all()), f"weights are exactly 1 on the {int((~scored).sum()):,} unscored and vehicle rows")
    half = dose_half(comp, dl, ic)[tr]
    g2 = line_group_sign(tbl["cell_id"].values[tr], cfg.population) > 0
    gamma = float(tier["gamma"])
    m = lambda h, g: float(w[scored & (half == h) & (g2 == g)].mean())
    lo, hi = m(0, True) - m(0, False), m(1, True) - m(1, False)
    if gamma > 0:
        check(lo * hi < 0, f"gamma {gamma:g}: G2's mean weight minus G1's flips sign between dose "
                           f"halves (low {lo:+.3f}, high {hi:+.3f})")
    else:
        print(f"      gamma 0: G2 - G1 weight difference low {lo:+.3f}, high {hi:+.3f} (no lever expected)")

    # ---- P2: group = arm ----
    grp = train_groups(cfg, "cell_id", tr)
    ak = arm_keys(comp, dl, ic)[tr]
    check(bool((np.asarray(grp).astype(str) == np.asarray(ak).astype(str)).all()),
          "P2's target group of every train row is its arm (the positivity cell minus the line)")

    if FAILS:
        print(f"\n[smoke6] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS))
        sys.exit(1)
    print("\n[smoke6] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
