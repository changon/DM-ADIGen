"""GPU smoke test of the Phase 5 step-C torch path (IMPLEMENT.md §3.8.2). Imports
PyTorch: run it on a GPU node, never on the dev box.

  1. LincsDataset with the injection on: y differs from the uninjected dataset by
     EXACTLY syn_c * beta * v, row for row, and `syn_meta` is exposed so the
     trainer can record beta in arch.json.
  2. The injection is refused when it cannot be trusted: no syn_meta.json, a
     syn_effect that disagrees with it, and a syn_seed that is not the table's
     own syn_c draw.
  3. The step-C conditioning contract: build_cond_spec with
     --adjustment_set syn_c promotes syn_c to role C, cond_from_batch carries it,
     and CFG drop nulls role A while LEAVING C -- a C the generator could drop
     would break the g-formula arm.
  4. The DR legs line up: dr_weights_counts.npz on a tiered dir is aligned to its
     train_idx, and its weights vary with syn_c on scored compounds while staying
     1 on unscored ones (§3.8.1 success criterion 4).
Exits nonzero on any failure.

    python -m src.tests.smoke_phase5_torch --device cuda \
        --tier data/mcf7_24h/nuisances_tier_Csyn_c_k0_g1_s42
    # step C2 (§3.8.4): the same checks under the per-compound injection
    python -m src.tests.smoke_phase5_torch --device cuda --syn_meta syn_meta_compound_r1.json \
        --tier data/mcf7_24h/nuisances_tier_Csyn_c_k0_g1_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.dataset import LincsDataset, build_cond_spec, cond_from_batch  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.data.synthetic import (  # noqa: E402
    SYN_META, directions_for, load_syn_meta, resolve_syn_meta_path, table_syn_seed)
from src.models.conditioning import CondEmbedder  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def _refuses(fn, label: str, *types, expect: str = "") -> None:
    """`fn` must raise, AND its message must mention `expect` -- otherwise a test
    can pass on an unrelated failure that happens to fire first."""
    try:
        fn()
    except types or (RuntimeError, FileNotFoundError) as e:
        msg = str(e)
        if expect and expect not in msg:
            check(False, f"{label} -> refused, but for another reason "
                         f"(wanted {expect!r}): {msg.splitlines()[0][:90]}")
            return
        check(True, f"{label} -> refused ({msg.splitlines()[0][:64]})")
        return
    except Exception as e:                       # noqa: BLE001
        check(False, f"{label} -> raised the wrong error: {type(e).__name__}: {e}")
        return
    check(False, f"{label} -> NOT refused")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--tier", default=None, help="A tiered split dir (for the DR-leg check).")
    p.add_argument("--n_rows", type=int, default=3000, help="Rows to load (keep it small).")
    p.add_argument("--syn_meta", default=SYN_META,
                   help="The injection under test: syn_meta.json (step C) or e.g. "
                        "syn_meta_compound_r1.json (step C2). config_from_args carries "
                        "it into every cfg built below.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        check(ok, f"CUDA available on {os.uname().nodename} (torch {torch.__version__}, "
                  f"cuda {torch.version.cuda})")
        if not ok:
            print("[smoke5] CUDA unavailable: aborting rather than falling back to CPU")
            sys.exit(1)
        print(f"      device: {torch.cuda.get_device_name(0)}")

    nz = cfg.paths.nuisance_dir
    NAME = cfg.syn_meta_name
    sp = resolve_syn_meta_path(cfg)
    if not os.path.isfile(sp):
        print(f"[smoke5] no {sp}; run `python -m src.data.synthetic --syn_effect ...` first")
        sys.exit(1)
    with open(sp) as fh:
        raw = json.load(fh)
    eff, seed = float(raw["syn_effect"]), int(raw["syn_seed"])
    idx = np.arange(min(int(args.n_rows), len(load_splits(cfg)["train_idx"]) * 2), dtype=np.int64)

    # ---- 1. the injection, end to end through LincsDataset ---------------
    plain = LincsDataset(cfg, indices=idx)
    cfg_syn = apply_paths_args(config_from_args(args), args)
    cfg_syn.outcome = replace(cfg_syn.outcome, syn_effect=eff, syn_seed=seed)
    syn = LincsDataset(cfg_syn, indices=idx)
    check(plain.syn_meta is None and syn.syn_meta is not None,
          "syn_meta is None without the injection and populated with it "
          "(so the trainer can pin beta in arch.json)")
    m = load_syn_meta(cfg_syn, n_genes=cfg.outcome.n_genes)
    print(f"      injection under test: {m['name']} (mode {m['mode']}"
          + (f", rho {m['rho']:g}, {m['n_directions']} directions" if m["mode"] != "global" else "")
          + f", sha1 {m['v_sha1'][:12]})")
    tbl = (load_from_disk(cfg.paths.tabular_dataset_dir)
           .select_columns(["syn_c", "compound_idx"]).to_pandas())
    sc = tbl["syn_c"].values.astype(np.int64)[idx]
    ci = tbl["compound_idx"].values.astype(np.int64)[idx]
    # Each syn_c = 1 row moves along ITS compound's direction (one shared v in
    # step C; v_k per compound in step C2).
    want = torch.from_numpy(
        (sc[:, None] * m["beta"] * directions_for(m, ci)).astype(np.float32))
    got = syn.y - plain.y
    err = float((got - want).abs().max())
    check(err < 3e-3,
          f"y_syn - y_plain == syn_c * beta * v{'_k' if m['mode'] != 'global' else ''} "
          f"row for row (max |diff| {err:.2e}, on a shift of norm {m['beta']:.2f})")
    if m["mode"] != "global":
        n_k = int(np.unique(ci[sc == 1]).size)
        check(n_k > 1, f"the loaded rows span {n_k} compounds with syn_c = 1, so the "
                       f"per-compound directions are actually exercised")
    check(float(got[sc == 0].abs().max()) == 0.0,
          f"syn_c=0 rows are bit-identical to the uninjected dataset")
    check(bool(torch.isfinite(syn.y).all()), "the injected y is finite")

    # ---- 2. refusals -----------------------------------------------------
    bad = apply_paths_args(config_from_args(args), args)
    bad.outcome = replace(bad.outcome, syn_effect=eff + 0.5, syn_seed=seed)
    _refuses(lambda: LincsDataset(bad, indices=idx[:64]),
             "a syn_effect that disagrees with syn_meta.json", RuntimeError,
             expect="syn_effect")
    bad2 = apply_paths_args(config_from_args(args), args)
    bad2.outcome = replace(bad2.outcome, syn_effect=eff, syn_seed=table_syn_seed(cfg) + 7)
    _refuses(lambda: LincsDataset(bad2, indices=idx[:64]),
             "a syn_seed that is not the table's syn_c draw", RuntimeError,
             expect="syn_seed")
    # A tiered dir holds no syn_meta.json of its own and MUST resolve to the base
    # build's: the injection belongs to the population and the table, not to a
    # split, and two copies could drift (a drifted beta looks exactly like
    # step-C bias). So this is a positive check, not a refusal.
    if args.tier and not os.path.isfile(os.path.join(args.tier, NAME)):
        shared = apply_paths_args(config_from_args(args), args)
        shared.outcome = replace(shared.outcome, syn_effect=eff, syn_seed=seed)
        shared.paths.nuisance_dir = os.path.abspath(args.tier)
        got = load_syn_meta(shared, n_genes=cfg.outcome.n_genes)
        check(abs(got["beta"] - m["beta"]) < 1e-12
              and got["v_sha1"] == m["v_sha1"]
              and resolve_syn_meta_path(shared) == os.path.abspath(sp),
              f"a tiered dir with no {NAME} resolves to the base build's "
              f"(same beta {got['beta']:.4f}, same v {got['v_sha1'][:12]})")
    else:
        print(f"skip  the fallback check needs a --tier dir with no {NAME}")

    # ...but a build that has no syn_meta.json anywhere must still refuse.
    LIMIT = os.path.join(os.path.dirname(cfg.paths.data_dir), "mcf7_24h_limit1500")
    if os.path.isdir(LIMIT) and not os.path.isfile(
            os.path.join(LIMIT, "nuisances", NAME)):
        import copy as _copy
        ns = _copy.copy(args)
        ns.data_dir, ns.nuisance_dir = LIMIT, None
        lim = apply_paths_args(config_from_args(ns), ns)
        lim.outcome = replace(lim.outcome, syn_effect=eff, syn_seed=table_syn_seed(lim))
        _refuses(lambda: LincsDataset(lim, indices=np.arange(64, dtype=np.int64)),
                 f"a build with no {NAME} anywhere", FileNotFoundError,
                 expect=NAME)
    else:
        print(f"skip  the refusal check needs a build with no {NAME} "
              f"(looked in {LIMIT})")

    # ---- 3. the step-C conditioning contract -----------------------------
    with open(os.path.join(nz, "nuisance_meta.json")) as fh:
        n_comp = int(json.load(fh)["n_compounds"])
    cfg_c = apply_paths_args(config_from_args(args), args)
    cfg_c.adjustment_set = ("syn_c",)
    cfg_c.__post_init__()
    lx = plain.log10_conc.numpy()[plain.is_control.numpy() == 0]
    spec = build_cond_spec(cfg_c, n_comp, lx, adjustment_set=("syn_c",))
    roles = {f.name: f.role for f in spec}
    check(roles.get("syn_c") == "C",
          f"build_cond_spec promotes syn_c to role C (roles {roles})")
    from torch.utils.data import default_collate
    batch = default_collate([plain[i] for i in range(32)])
    cond = cond_from_batch(batch, spec, device=dev)
    check("syn_c" in cond and cond["syn_c"].shape[0] == 32,
          f"cond_from_batch carries syn_c {tuple(cond['syn_c'].shape)}")
    emb = CondEmbedder(spec, hidden_size=64).to(dev)
    drop = torch.ones(32, dtype=torch.bool, device=dev)
    e_keep = emb(cond, drop=None)
    e_drop = emb(cond, drop=drop)
    check(bool(torch.isfinite(e_keep).all()) and bool(torch.isfinite(e_drop).all()),
          "CondEmbedder is finite with and without the CFG drop mask")
    alt = dict(cond)
    alt["syn_c"] = 1 - alt["syn_c"]
    check(not torch.allclose(emb(alt, drop=drop), e_drop),
          "CFG drop leaves C: flipping syn_c still changes the dropped embedding "
          "(a droppable C would break the conditional arm)")

    # ---- 4. the DR legs on a tiered dir ----------------------------------
    T = args.tier
    if T and os.path.isdir(T):
        ts = json.load(open(os.path.join(T, "splits.json")))
        wz = np.load(os.path.join(T, "dr_weights_counts.npz"))
        tr = np.asarray(ts["train_idx"], dtype=np.int64)
        check(np.array_equal(wz["row_id"], tr),
              f"{os.path.basename(T)}: dr_weights_counts row_id == the tier's train_idx "
              f"({tr.size:,} rows)")
        df = (load_from_disk(cfg.paths.tabular_dataset_dir)
              .select_columns(["compound_idx", "is_control", "syn_c"]).to_pandas())
        comp = df["compound_idx"].values[tr]
        ctl = df["is_control"].values.astype(bool)[tr]
        sc_tr = df["syn_c"].values.astype(np.int64)[tr]
        scored = np.isin(comp, [int(v) for v in ts["tier"]["scored_compounds"].values()])
        w = wz["w"]
        check(bool(np.allclose(w[~scored & ~ctl], 1.0)),
              f"{os.path.basename(T)}: weights are exactly 1 on unscored treated rows")
        s0, s1 = w[scored & ~ctl & (sc_tr == 0)], w[scored & ~ctl & (sc_tr == 1)]
        gamma = float(ts["tier"]["gamma"])
        varies = s0.size and s1.size and abs(float(s0.mean()) - float(s1.mean())) > 1e-6
        if gamma > 0:
            check(bool(varies),
                  f"{os.path.basename(T)}: at gamma={gamma} the weights vary with syn_c on "
                  f"scored rows (mean {float(s0.mean()):.3f} vs {float(s1.mean()):.3f})")
        else:
            print(f"      gamma=0: syn_c means {float(s0.mean()):.3f} / "
                  f"{float(s1.mean()):.3f} (no systematic difference expected)")
    else:
        print("skip  no --tier dir given for the DR-leg check")

    if FAILS:
        print(f"\n[smoke5] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS))
        sys.exit(1)
    print("\n[smoke5] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
