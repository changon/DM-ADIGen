"""Read-back checks of a build_dataset output (Phase 0 smoke). No PyTorch.

Re-derives what it can independently of build_dataset: expression rows are
re-read from the GCTX one well at a time by inst_id, the gene order from
gene_info, the plate tokens from det_plate. Exits nonzero on any failure.

Run from lincs/ (a CPU job for the full population; ~1 min):
    python -m src.tests.check_build                       # data/mcf7_24h
    python -m src.tests.check_build --data_dir data/mcf7_24h_limit1500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import ContextEncoder  # noqa: E402
from src.spec import (  # noqa: E402
    CONTEXT_COL, CONTEXT_FIELDS, POPULATIONS, CaseConfig, Paths, decisions_record,
    line_groups, population_of_build)

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None, help="build dir (default: cfg.paths.data_dir)")
    p.add_argument("--n_sample", type=int, default=200, help="rows re-read from the GCTX by inst_id")
    p.add_argument("--population", default=None, choices=sorted(POPULATIONS),
                   help="default: the build's own (population_qc.json), else mcf7_24h")
    args = p.parse_args()

    import h5py
    from datasets import load_from_disk

    # The build names its own population (population_qc.json); --population may
    # confirm it but not contradict it.
    built = population_of_build(os.path.abspath(args.data_dir)) if args.data_dir else None
    name = args.population or built or "mcf7_24h"
    if built is not None and built != name:
        raise SystemExit(f"[check] --population {name} but {args.data_dir} is a build of {built!r}")
    cfg = CaseConfig(population=POPULATIONS[name])
    if args.data_dir:
        cfg.paths = Paths(population=cfg.population.name, data_dir=os.path.abspath(args.data_dir))
    P = cfg.paths
    print(f"[check] {P.data_dir}")
    df = load_from_disk(P.tabular_dataset_dir).to_pandas()
    expr = np.load(P.expr_npy, mmap_mode="r")
    go = json.load(open(P.gene_order_json))
    qc = json.load(open(P.population_qc_json))
    vocab = json.load(open(os.path.join(P.nuisance_dir, "compound_vocab.json")))
    nm = json.load(open(os.path.join(P.nuisance_dir, "nuisance_meta.json")))
    cov = json.load(open(os.path.join(P.nuisance_dir, "covariate_encoder.json")))
    ce = json.load(open(P.context_encoder_json))
    ctl = df["is_control"].astype(bool).values

    # --- the build was made under the current frozen decisions (spec.py) --------
    want_dec = json.loads(json.dumps(decisions_record(cfg)))
    stale = sorted(k for k in set(want_dec) | set(qc["decisions"])
                   if want_dec.get(k) != qc["decisions"].get(k))
    check(not stale, f"population_qc decisions = current spec.decisions_record (differ: {stale or 'none'})")

    # --- Y alignment ----------------------------------------------------------
    check(expr.shape == (len(df), cfg.outcome.n_genes) and expr.dtype == np.float32,
          f"expr.npy {expr.shape} {expr.dtype} for {len(df):,} table rows")
    check(bool(np.isfinite(expr).all()), "expr finite")
    check((df["row_id"].values == np.arange(len(df))).all(), "row_id = 0..N-1")
    check(df["inst_id"].is_unique and df["inst_id"].is_monotonic_increasing, "rows sorted by unique inst_id")
    gi = pd.read_csv(P.raw("gene_info"), sep="\t", dtype=str)
    lm_ids = gi.loc[gi["pr_is_lm"] == "1", "pr_gene_id"].astype(int).tolist()
    check(go["pr_gene_id"] == lm_ids and go["n_genes"] == cfg.outcome.n_genes,
          "gene_order.json = gene_info landmarks, in order")
    fp = {qc["table_fingerprint"], nm["table_fingerprint"], go["table_fingerprint"]}
    fp_table = hashlib.sha1(("\n".join(df["inst_id"]) + "\n#genes\n"
                             + ",".join(map(str, go["pr_gene_id"]))).encode()).hexdigest()[:16]
    check(fp == {fp_table}, f"table fingerprint {fp_table} recomputed = recorded {fp}")
    with h5py.File(P.raw("gctx"), "r") as f:
        wid = f["/0/META/COL/id"][:].astype(str)
        gid = f["/0/META/ROW/id"][:].astype(str)
        gpos = pd.Series(np.arange(gid.size), index=gid).loc[[str(g) for g in go["pr_gene_id"]]].values
        wpos = pd.Series(np.arange(wid.size), index=wid)
        pick = np.sort(np.random.default_rng(0).choice(len(df), size=min(args.n_sample, len(df)), replace=False))
        ok_pos = ok_val = True
        for i in pick:
            pos = int(wpos[df["inst_id"].iat[i]])
            ok_pos &= pos == int(df["gctx_col"].iat[i])
            ok_val &= np.array_equal(f["/0/DATA/0/matrix"][pos, :][gpos], expr[i])
    check(ok_pos, f"gctx_col = GCTX position of inst_id ({pick.size} rows)")
    check(ok_val, f"expr rows = direct GCTX reads by inst_id ({pick.size} rows)")

    # --- population + action encoding -------------------------------------------
    pop = cfg.population
    # completeness: table + QC-dropped wells = an independent re-filter of inst_info
    ii = pd.read_csv(P.raw("inst_info"), sep="\t", dtype=str, keep_default_na=False)
    want = ii[ii["pert_type"].isin(pop.pert_types) & ii["cell_id"].isin(pop.cell_ids)
              & (ii["pert_time"].astype(float) == pop.pert_time_h) & (ii["pert_time_unit"] == "h")]
    if qc["limit"] is not None:                       # a --limit build keeps whole plates
        want = want[want["det_plate"].isin(set(qc["plate_qc"]["plates"]))]
    dropped_plates = [d["plate"] for d in qc["plate_qc"]["dropped_plates"]]
    after_qc = want.loc[~want["det_plate"].isin(dropped_plates)]
    n_qc = len(want) - len(after_qc)
    multi = len(pop.cell_ids) > 1
    n_rule = 0
    if multi:
        # the every-line rule, re-derived from inst_info: a treated well stays only
        # if its compound has a treated well in every line after plate QC
        t_ = after_qc[after_qc["pert_type"] == "trt_cp"]
        nl_ = t_.groupby("pert_id")["cell_id"].nunique()
        ok_ids = set(nl_.index[nl_ == len(pop.cell_ids)])
        keep_ = (after_qc["pert_type"] != "trt_cp") | after_qc["pert_id"].isin(ok_ids)
        n_rule = int((~keep_).sum())
        after_qc = after_qc.loc[keep_]
        cc = qc.get("common_compounds") or {}
        check(cc.get("applies") is True and cc.get("n_treated_wells_dropped") == n_rule
              and cc.get("n_compounds_kept") == len(ok_ids),
              f"every-line rule re-derived from inst_info: {len(ok_ids):,} compounds in all "
              f"{len(pop.cell_ids)} lines, {n_rule:,} treated wells dropped (recorded {cc.get('n_treated_wells_dropped')})")
        per = df.loc[~ctl].groupby("pert_id")["cell_id"].nunique()
        check(bool((per == len(pop.cell_ids)).all()),
              f"every treated compound of the table has wells in all {len(pop.cell_ids)} lines")
        check(set(df["cell_id"]) == set(pop.cell_ids), f"all {len(pop.cell_ids)} lines present {sorted(set(df['cell_id']))}")
        lg = line_groups(pop)
        check(qc.get("line_groups") == ({k: list(v) for k, v in lg.items()} if lg else None),
              f"population_qc line groups = spec.LINE_GROUPS {qc.get('line_groups')}")
    else:
        check("common_compounds" not in qc, "a single-line build records no every-line rule")
    want_kept = set(after_qc["inst_id"])
    check(want_kept == set(df["inst_id"]) and n_qc == qc["plate_qc"]["n_wells_dropped"],
          f"table = re-filtered inst_info ({len(want):,}) minus QC-dropped wells ({n_qc})"
          + (f" minus the every-line rule ({n_rule})" if multi else ""))
    check(df["cell_id"].isin(pop.cell_ids).all() and np.allclose(df["pert_time"], pop.pert_time_h),
          f"population cell_ids {pop.cell_ids}, {pop.pert_time_h:g} h")
    check((df.loc[ctl, "pert_type"] == "ctl_vehicle").all() and (df.loc[~ctl, "pert_type"] == "trt_cp").all(),
          "is_control <=> pert_type == ctl_vehicle; others trt_cp")
    check((df.loc[ctl, "pert_id"] == "DMSO").all(), "every vehicle is DMSO")
    check((df.loc[ctl, "compound_idx"] == 0).all() and (df.loc[~ctl, "compound_idx"] > 0).all(),
          "compound_idx = 0 iff vehicle")
    check((df.loc[ctl, "conc"] == 0).all() and (df.loc[ctl, "log10_conc"] == cfg.action.control_log10_sentinel).all()
          and (df.loc[ctl, "dose_level"] == 0).all(), "vehicle conc 0, log10_conc sentinel, dose_level 0")
    check(df.loc[ctl, "pert_dose"].isna().all(), "vehicle pert_dose missing (no -666)")
    check(not (df.select_dtypes("number") == float(cfg.action.na_sentinel)).any().any(),
          "no -666 in any numeric column")
    check(np.allclose(df.loc[~ctl, "conc"], df.loc[~ctl, "pert_dose"], rtol=1e-6)
          and np.allclose(np.log10(df.loc[~ctl, "conc"]), df.loc[~ctl, "log10_conc"], atol=1e-5),
          "treated conc = pert_dose (uM), log10_conc = log10(conc)")
    lv = np.asarray(cfg.action.dose_levels_um)
    trt_dl = df.loc[~ctl, "dose_level"].values
    trt_c = df.loc[~ctl, "conc"].values.astype(np.float64)
    on = np.isin(trt_dl, lv)
    check(bool((np.abs(np.log10(trt_c[on]) - np.log10(trt_dl[on])) <= cfg.action.dose_level_tol_log10 + 1e-6).all()),
          "on-grid dose_level within tolerance of the dose")
    far = np.abs(np.log10(trt_c)[:, None] - np.log10(lv)[None, :]).min(1) > cfg.action.dose_level_tol_log10
    check(bool((far == ~on).all()), "off-grid dose_level exactly for doses outside tolerance")
    inv = {v: k for k, v in vocab.items()}
    check(inv[0] == "__control__" and all(inv[c] == p for c, p in zip(df.loc[~ctl, "compound_idx"], df.loc[~ctl, "pert_id"])),
          f"compound vocab on pert_id ({len(vocab)} incl. __control__)")
    check(set(vocab) - {"__control__"} == set(df.loc[~ctl, "pert_id"]), "vocab = treated pert_ids of the table")

    # --- plate tokens, covariates, syn_c, context --------------------------------
    parts = df["det_plate"].str.split("_")
    check((parts.str[0] == df["plate_map"]).all() and (parts.str[3] == df["replicate"]).all()
          and (parts.str[4] == df["batch"]).all(), "plate_map / replicate / batch from det_plate")
    check((df["det_plate"] + ":" + df["det_well"] == df["inst_id"]).all(), "inst_id = det_plate:det_well")
    check((df["det_well"].str[0] == df["well_row"]).all()
          and (df["det_well"].str[1:].astype(int) == df["well_col"]).all(), "well_row / well_col from det_well")
    cv = np.stack(df["cov_vec"].values)
    check(nm["n_compounds"] == len(vocab) and nm["cov_dim"] == cv.shape[1], f"nuisance_meta {nm}")
    off, ok_blocks = 0, True
    for c in cov["categorical_cols"]:
        w = len(cov["levels"][c])
        blk = cv[:, off:off + w]
        ok_blocks &= bool((blk.sum(1) == 1).all())
        ok_blocks &= bool((np.array(cov["levels"][c])[blk.argmax(1)] == df[c].astype(str).values).all())
        off += w
    check(ok_blocks and off == cv.shape[1], f"cov_vec = one-hot {cov['categorical_cols']} {cov['levels']}")
    arm = df["cell_id"] + "|" + df["pert_id"] + "|" + df["dose_level"].map(lambda v: f"{v:.6g}")
    s = df["syn_c"].astype(int) * 2 - 1
    imb_arm = int(s[~ctl].groupby(arm[~ctl]).sum().abs().max())
    imb_plate = int(s[ctl].groupby(df.loc[ctl, "det_plate"]).sum().abs().max())
    check(imb_arm <= 1 and imb_plate <= 1, f"syn_c balanced (max |n1-n0|: arm {imb_arm}, plate DMSO {imb_plate})")
    check(tuple(ce["fields"]) == CONTEXT_FIELDS and ce["levels"]["plate"] == sorted(df["det_plate"].unique())
          and ce["levels"]["syn_c"] == ["0", "1"], "context encoder from the filtered table")
    codes = ContextEncoder.load(P.context_encoder_json).encode(df)   # the Arrow round-tripped table
    check(codes.shape == (len(df), len(CONTEXT_FIELDS)) and (codes >= 0).all()
          and (codes[:, CONTEXT_COL["syn_c"]] == df["syn_c"].values).all(),
          f"ContextEncoder encodes every table row {codes.shape}")

    # --- plate QC -------------------------------------------------------------------
    pq = qc["plate_qc"]
    dropped = [d["plate"] for d in pq["dropped_plates"]]
    check(not df["det_plate"].isin(dropped).any(), f"QC-dropped plates absent from the table {dropped}")
    ratio = cfg.outcome.plate_qc_max_spread_ratio
    rule_ok = all((r["spread"] > ratio * pq["median_spread"][r["cell_id"]]) == r["dropped"]
                  for r in pq["plates"].values())
    check(rule_ok, f"recorded drops follow spread > {ratio} x median")
    # a multi-line --limit build may keep a plate whose treated wells the every-line
    # rule removed entirely; it then holds only vehicles, and is still a table plate
    check(set(pq["plates"]) - set(dropped) == set(df["det_plate"]), "kept plates = table plates")
    lines_of_plate = df.groupby("det_plate")["cell_id"].nunique()
    check(bool((lines_of_plate == 1).all()), "every plate holds one cell line")
    dmso = df.loc[ctl].groupby("det_plate").size()
    check(set(dmso.index) == set(df["det_plate"]), f"every plate has DMSO wells (min {dmso.min()})")
    # kept plates' spreads recomputed from expr.npy (1.4826 * MAD per gene, median over genes)
    x_all = np.asarray(expr)
    max_err = 0.0
    for plate, idx in df.index[ctl].groupby(df.loc[ctl, "det_plate"].values).items():
        x = x_all[np.asarray(idx)]
        sp = float(np.median(1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)))
        max_err = max(max_err, abs(sp - pq["plates"][plate]["spread"]))
    check(max_err < 1e-5, f"kept plates' DMSO spreads recomputed from expr.npy (max |err| {max_err:.1e})")

    S = qc["summary"]
    print(f"[check] n={S['n_wells']:,} treated={S['n_treated']:,} vehicle={S['n_vehicle']:,} "
          f"compounds={S['n_compounds']:,} plates={S['n_plates']} arms={S['n_arms']:,} "
          f"dropped={dropped} n_dropped_wells={pq['n_wells_dropped']} "
          f"arms_emptied={len(pq['arms_emptied'])} arms_to_1_well={pq['n_arms_reduced_to_one_well']}")
    if FAILS:
        print(f"[check] {len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("[check] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
