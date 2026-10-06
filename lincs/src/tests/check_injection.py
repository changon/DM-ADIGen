"""Checks of an injected build (POLICY_LEARNING.md §14; `src.data.inject_modifier`). No PyTorch.

  copy      the table, gene order, context encoder, every copied split dir's files
            (splits.json, expr_meta*.json, weights, nu_rows, vocab, encoders) are
            byte-identical to the source build's; no retargeting weights were copied
  switch    recomputed from the table with `splits.dose_half` and `spec.line_group_sign`
            it equals the stored one; vehicles 0; only the group's thin half; the signs
            are per (compound, group) and roughly balanced
  expr      untouched rows are bit-identical; on a sample of injected rows the z-space
            delta under the base expr_meta equals beta * m * s_c (atol 1e-3); the
            signature file equals `compound_signatures` on the source build
  frame     two `learn.load_frame` calls (source and injected build, the same tier name)
            give mu_h(injected) - mu_h(source) = beta * m on thin-half cells with a
            holdout well, and 0 elsewhere, for utility A; the injected frame's axis is
            the stored signature

    python -m src.tests.check_injection --data_dir data/core5_24h_inj
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.inject_modifier import META_FILE, sha1_of  # noqa: E402
from src.policy.learn import THIN_HALF  # noqa: E402
from src.data.splits import dose_half, load_splits  # noqa: E402
from src.policy import learn as pl  # noqa: E402
from src.spec import (  # noqa: E402
    CaseConfig, POPULATIONS, Paths, add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args,
    line_group_sign, population_of_build)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n_sample", type=int, default=3000)
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)
    out = os.path.normpath(cfg.paths.data_dir)
    with open(os.path.join(out, META_FILE)) as fh:
        inj = json.load(fh)
    src = inj["source_data_dir"]
    pop = population_of_build(src)
    scfg = CaseConfig(population=POPULATIONS[pop]); scfg.adjustment_set = cfg.adjustment_set
    scfg.paths = Paths(population=pop, data_dir=src, train_output_dir=str(PROJECT_ROOT / "runs" / os.path.basename(src)))
    scfg.__post_init__()
    print(f"[check_injection] {out} <- {src}; beta {inj['beta']:.3f} = {inj['mult']:g} x sigma_w {inj['sigma_w']:.3f}")

    # ---- copy ------------------------------------------------------------------------------
    same = filecmp.dircmp(os.path.join(src, "lincs_tabular"), os.path.join(out, "lincs_tabular"))
    check(not same.diff_files and not same.left_only and not same.right_only, "copy: the table is byte-identical")
    for fn in ("gene_order.json", "context_encoder.json"):
        check(filecmp.cmp(os.path.join(src, fn), os.path.join(out, fn), shallow=False), f"copy: {fn} identical")
    for d in inj["copied_split_dirs"]:
        a, b = os.path.join(src, d), os.path.join(out, d)
        fa = sorted(f for f in os.listdir(a) if not f.startswith("dr_weights_retarget_") and not f.endswith(".tmp") and os.path.isfile(os.path.join(a, f)))
        fb = sorted(f for f in os.listdir(b) if os.path.isfile(os.path.join(b, f)))
        check(fa == fb and all(filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False) for f in fa)
              and not any(f.startswith("dr_weights_retarget_") for f in fb),
              f"copy: {d}: {len(fb)} files identical, no retargeting weights")
    with open(os.path.join(out, "population_qc.json")) as fh:
        qc = json.load(fh)
    check(qc.get("injection", {}).get("beta") == inj["beta"] and qc["table_fingerprint"] == inj["source_table_fingerprint"],
          "copy: population_qc.json carries the injection block and the source table fingerprint")

    # ---- switch -------------------------------------------------------------------------------
    from datasets import load_from_disk
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "cell_id", "det_plate"]).to_pandas())
    N = len(meta)
    comp = meta["compound_idx"].values.astype(np.int64); dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool)
    lines = sorted(set(meta["cell_id"].values)); line_t = np.array([lines.index(c) for c in meta["cell_id"].values])
    g2 = line_group_sign(np.array(lines), cfg.population) > 0
    signs = np.load(os.path.join(out, inj["signs_file"])); switch = np.load(os.path.join(out, inj["switch_file"]))
    half = dose_half(comp, dl, ctl)
    thin = np.where(g2[line_t], THIN_HALF["G2"], THIN_HALF["G1"])
    want = np.where(~ctl & (half == thin), signs[comp, np.where(g2[line_t], 1, 0)], 0).astype(np.int8)
    check(np.array_equal(want, switch) and sha1_of(switch) == inj["switch_sha1"], "switch: recomputed from the table = stored")
    check((switch[ctl] == 0).all() and (switch[~ctl & (half != thin)] == 0).all() and (switch[~ctl & (half == thin)] != 0).all(),
          "switch: vehicles 0, the favoured half 0, every thin-half treated row injected")
    tol = 3.0 / np.sqrt(2.0 * (signs.shape[0] - 1))           # 3 sd of a fair coin over 2 n_comp draws
    check(signs[0].tolist() == [0, 0] and set(np.unique(signs[1:]).tolist()) == {-1, 1}
          and abs(float((signs[1:] > 0).mean()) - 0.5) < tol and not np.array_equal(signs[1:, 0], signs[1:, 1]),
          f"switch: signs per (compound, group) in {{-1, +1}}, +1 share {float((signs[1:] > 0).mean()):.3f} (within {tol:.3f} of 1/2), the two groups independent")

    # ---- expr ---------------------------------------------------------------------------------
    xs = np.load(scfg.paths.expr_npy, mmap_mode="r"); xo = np.load(cfg.paths.expr_npy, mmap_mode="r")
    rng = np.random.default_rng(0)
    un = rng.choice(np.flatnonzero(switch == 0), min(args.n_sample, int((switch == 0).sum())), replace=False)
    check(np.array_equal(np.asarray(xs[np.sort(un)]), np.asarray(xo[np.sort(un)])), f"expr: {un.size:,} untouched rows bit-identical")
    V = np.load(os.path.join(out, inj["signature_file"])).astype(np.float64)
    em = load_expr_meta(scfg, splits=load_splits(scfg))
    pc = plate_codes(meta["det_plate"].values, em["plates"])
    ij = np.sort(rng.choice(np.flatnonzero(switch != 0), min(args.n_sample, int((switch != 0).sum())), replace=False))
    dz = normalize_expr(np.asarray(xo[ij]), pc[ij], em).astype(np.float64) - normalize_expr(np.asarray(xs[ij]), pc[ij], em).astype(np.float64)
    want_dz = inj["beta"] * switch[ij][:, None] * V[comp[ij]]
    check(np.abs(dz - want_dz).max() < 1e-3, f"expr: the z-space shift of {ij.size:,} injected rows = beta * m * s_c (max |err| {np.abs(dz - want_dz).max():.1e})")
    base = load_splits(scfg)
    hold = np.zeros(N, bool); hold[base["holdout_idx"]] = True
    zs = normalize_expr(np.asarray(xs), pc, em)
    V2 = pl.compound_signatures(zs, comp, ctl, line_t, dl, hold, int(comp.max()) + 1)
    check(np.abs(V2 - V).max() < 1e-5 and np.allclose(np.sqrt((V[1:] ** 2).sum(axis=1)), 1, atol=1e-5),
          "expr: the signature file = compound_signatures on the source build, unit rows")
    del zs

    # ---- frame --------------------------------------------------------------------------------
    tiers = [d for d in inj["copied_split_dirs"] if d.startswith("nuisances_tier_") and not d.endswith(("_h1", "_h2"))]
    if not tiers:
        check(False, "frame: the source build had no PL tier to copy (run scripts/policy_cpu.sub on it first)")
        print(f"\n[check_injection] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS)); sys.exit(1)
    tier = sorted(t for t in tiers if "_g0_" not in t)[0] if any("_g0_" not in t for t in tiers) else tiers[0]
    Fs = pl.load_frame(scfg, os.path.join(src, tier), 0, signature_file=os.path.join(out, inj["signature_file"]))
    Fi = pl.load_frame(cfg, os.path.join(out, tier), 0)
    check(Fi["injection"] is not None and Fi["signature_file"] == os.path.join(out, inj["signature_file"])
          and np.array_equal(Fi["S_ctx"], Fs["S_ctx"]), "frame: the injected frame uses the stored signature by default")
    d = Fi["mu"]["holdout"]["A"] - Fs["mu"]["holdout"]["A"]
    thin_c = (Fi["half"] == Fi["thin_half"][:, None]) & Fi["valid"] & (Fi["cnt"]["holdout"] > 0)
    fav_c = Fi["valid"] & ~(Fi["half"] == Fi["thin_half"][:, None]) & (Fi["cnt"]["holdout"] > 0)
    m_ctx = signs[Fi["comp"], np.where(Fi["g2"][Fi["ctx_line"]], 1, 0)]
    err_thin = float(np.abs(d[thin_c] - (inj["beta"] * m_ctx[:, None] * np.ones_like(d))[thin_c]).max())
    err_fav = float(np.abs(d[fav_c]).max())
    check(err_thin < 1e-2 and err_fav < 1e-2,
          f"frame: the holdout truth moves by beta * m on thin-half cells (max |err| {err_thin:.1e}) and not elsewhere ({err_fav:.1e})")
    check(np.array_equal(Fi["thin_half"], Fs["thin_half"]) and np.array_equal(Fi["pi_b"], Fs["pi_b"]),
          "frame: the thin half and the logger are the source build's")
    if FAILS:
        print(f"\n[check_injection] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS)); sys.exit(1)
    print("\n[check_injection] all checks passed")


if __name__ == "__main__":
    main()
