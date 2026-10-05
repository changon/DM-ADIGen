"""One-time build of the LINCS L1000 tabular dataset and expression matrix.

Copied from RxRx19a/src/data/build_dataset.py, then rewritten for GSE70138
(IMPLEMENT.md §3.2). Reads inst_info, filters the population (spec.PopulationSpec),
reads the 978 landmark genes of the retained wells from the Level 3 GCTX with raw
h5py, drops plates by DMSO-spread QC, encodes the action, assigns `syn_c`, and
writes, under cfg.paths.data_dir:

    lincs_tabular/          HF Arrow table, one row per retained well
    expr.npy                (N, 978) float32 raw Level 3, table row order
    gene_order.json         the 978 landmark pr_gene_ids, in column order
    population_qc.json      filters, plate QC spreads / drops, decisions, summary
    context_encoder.json    context levels, built from the FILTERED table
    nuisances/compound_vocab.json, covariate_encoder.json, nuisance_meta.json

No normalisation statistics are computed here: plate centres and per-gene stats
need the train split (src.data.expr_stats, after src.data.splits).

Run from lincs/ as a CPU job (the full population reads ~1.9 GB of the GCTX):
    python -m src.data.build_dataset                  # -> data/mcf7_24h/
    python -m src.data.build_dataset --limit 1500     # smoke -> data/mcf7_24h_limit1500/
    python -m src.data.build_dataset --population core5_24h                # step A -> data/core5_24h/
    python -m src.data.build_dataset --population core5_24h --limit 11000  # smoke: 2 whole plate maps

A population with more than one cell line (step A's core5_24h) keeps only
compounds with a treated well in EVERY line, applied after plate QC; its --limit
keeps whole plate MAPS (every line's and replicate's plates of a map), so the
smoke build still has each compound in each line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python -m src.data.build_dataset` from lincs/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import assign_syn_c, max_group_imbalance  # noqa: E402
from src.spec import (  # noqa: E402
    CONTEXT_FIELDS as _CONTEXT_FIELDS, DECLARED_LEVELS, POPULATIONS, CaseConfig, Paths,
    decisions_record, format_role_summary, line_groups)

CONTROL_COMPOUND_NAME = "__control__"
CONTROL_COMPOUND_IDX = 0  # index 0 is reserved for control

# Level 3 GCTX (HDF5). GCTX calls wells "columns" and genes "rows", but the
# matrix is stored (wells, genes): a well is a contiguous row, no transpose.
GCTX_MATRIX = "/0/DATA/0/matrix"
GCTX_WELL_IDS = "/0/META/COL/id"   # inst_id of each matrix row; NOT inst_info order
GCTX_GENE_IDS = "/0/META/ROW/id"   # pr_gene_id of each matrix column; gene_info order

# Plate QC: robust SD = 1.4826 * MAD. A plate needs a few DMSO wells for a MAD
# (and for centring later); every MCF7 24 h plate has 18-28.
MAD_TO_SD = 1.4826
MIN_DMSO_PER_PLATE = 3

TABLE_COLUMNS = [
    "row_id", "inst_id", "gctx_col", "cell_id", "pert_id", "pert_iname", "pert_mfc_id",
    "pert_type", "pert_dose", "pert_time",
    "det_plate", "det_well", "plate_map", "replicate", "batch", "well_row", "well_col",
    "compound_idx", "is_control", "conc", "log10_conc", "dose_level", "syn_c",
]


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
def read_tsv(path: str) -> pd.DataFrame:
    """A LINCS metadata TSV, every cell kept as its exact string (no NaN coercion)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def select_population(inst: pd.DataFrame, cfg: CaseConfig) -> tuple[pd.DataFrame, dict]:
    """Rows of inst_info in cfg.population, plus the row count after each filter."""
    pop = cfg.population
    counts = {"inst_info": int(len(inst))}
    m = inst["pert_type"].isin(pop.pert_types).values
    counts["pert_type"] = int(m.sum())
    # pert_time is the string "24.0": compare numerically, and in hours.
    t = pd.to_numeric(inst["pert_time"], errors="coerce").values
    m &= np.isclose(t, pop.pert_time_h) & (inst["pert_time_unit"] == "h").values
    counts["pert_time"] = int(m.sum())
    m &= inst["cell_id"].isin(pop.cell_ids).values
    counts["cell_id"] = int(m.sum())
    df = inst.loc[m].copy()
    if df.empty:
        raise ValueError(f"population {pop.key()} is empty in inst_info")
    return df, counts


def check_metadata(df: pd.DataFrame, cfg: CaseConfig, pert_info: pd.DataFrame,
                   cell_info: pd.DataFrame) -> None:
    """Assert the LINCS encodings the action encoding relies on. Raises, never converts."""
    a = cfg.action
    if df["inst_id"].duplicated().any():
        raise ValueError("duplicate inst_id in the population")
    joined = df["det_plate"] + ":" + df["det_well"]
    if not (joined == df["inst_id"]).all():
        raise ValueError("inst_id != det_plate:det_well for some wells")
    tok = df["det_plate"].str.split("_")
    want_time = pd.to_numeric(df["pert_time"]).map(lambda t: f"{t:g}H")
    if not ((tok.str[1] == df["cell_id"]).all() and (tok.str[2] == want_time).all()):
        raise ValueError("det_plate cell / time tokens disagree with cell_id / pert_time")
    missing_cells = sorted(set(df["cell_id"]) - set(cell_info["cell_id"]))
    if missing_cells:
        raise ValueError(f"cell_ids absent from cell_info: {missing_cells}")

    is_ctl = df[a.control_col].isin(a.control_values).values
    veh, trt = df.loc[is_ctl], df.loc[~is_ctl]
    bad_veh = sorted(set(veh["pert_id"]) - set(a.vehicle_pert_ids))
    if bad_veh:
        raise ValueError(
            f"vehicle wells with pert_id {bad_veh}; compound_idx=0 would merge them "
            f"with {list(a.vehicle_pert_ids)}. Declare them in ActionSpec.vehicle_pert_ids "
            f"only if they are the same control.")
    sentinel = float(a.na_sentinel)
    veh_dose = pd.to_numeric(veh[a.continuous_col], errors="coerce")
    if not ((veh_dose == sentinel).all() and (veh[a.dose_unit_col] == a.na_sentinel).all()):
        raise ValueError(f"vehicle dose/unit is not the LINCS {a.na_sentinel} sentinel for every vehicle well")
    units = sorted(set(trt[a.dose_unit_col]))
    if units != [a.dose_unit]:
        raise ValueError(f"treated dose units {units}; expected only {a.dose_unit!r} (asserted, not converted)")
    trt_dose = pd.to_numeric(trt[a.continuous_col], errors="coerce").values
    if not (np.isfinite(trt_dose).all() and (trt_dose > 0).all()):
        raise ValueError("treated wells with a missing or non-positive dose")
    pi = pert_info.set_index("pert_id")["pert_type"]
    trt_ids = pd.Series(sorted(set(trt["pert_id"])))
    absent = trt_ids[~trt_ids.isin(pi.index)].tolist()
    if absent:
        raise ValueError(f"{len(absent)} treated pert_ids absent from pert_info, e.g. {absent[:5]}")


def parse_plate(det_plate: pd.Series) -> pd.DataFrame:
    """det_plate `LJP005_MCF7_24H_X1_B17` -> plate_map LJP005, replicate X1, batch B17.
    Replicates may carry suffixes (`X3.A2`, `X2.L2.A2`); plate maps may contain dots (`REP.A002`)."""
    bad = det_plate[det_plate.str.count("_") != 4]
    if len(bad):
        raise ValueError(f"det_plate not of the form MAP_CELL_TIME_REP_BATCH, e.g. {bad.head(5).tolist()}")
    parts = det_plate.str.split("_", expand=True)
    if not parts[3].str.match(r"^X\d+(\.[A-Za-z0-9]+)*$").all():
        raise ValueError(f"unexpected replicate tokens: {sorted(set(parts[3]))[:10]}")
    if not parts[4].str.match(r"^B\d+$").all():
        raise ValueError(f"unexpected batch tokens: {sorted(set(parts[4]))[:10]}")
    return pd.DataFrame({"plate_map": parts[0].values, "replicate": parts[3].values,
                         "batch": parts[4].values}, index=det_plate.index)


def encode_action(df: pd.DataFrame, cfg: CaseConfig) -> pd.DataFrame:
    """Add is_control, pert_dose (NaN for vehicles; null once in Arrow), conc, log10_conc, dose_level.

    Vehicles: conc = 0, log10_conc = control_log10_sentinel, dose_level = 0; the
    LINCS -666 never reaches a numeric column. The generator later sees the
    vehicle dose as NaN (learned null) through cond_from_arrays.
    """
    a = cfg.action
    out = df.copy()
    is_ctl = out[a.control_col].isin(a.control_values).values
    out["is_control"] = is_ctl.astype(np.int8)
    dose = pd.to_numeric(out[a.continuous_col], errors="coerce").values.astype(np.float64)
    dose[is_ctl] = np.nan
    out["pert_dose"] = dose
    conc = np.where(is_ctl, 0.0, dose)
    out["conc"] = conc.astype(np.float32)
    log10_conc = np.full(len(out), a.control_log10_sentinel, dtype=np.float64)
    log10_conc[~is_ctl] = np.log10(conc[~is_ctl])
    out["log10_conc"] = log10_conc.astype(np.float32)
    out["dose_level"] = assign_dose_level(conc, is_ctl, cfg)
    return out


def assign_dose_level(conc: np.ndarray, is_control: np.ndarray, cfg: CaseConfig) -> np.ndarray:
    """(N,) float64 eval / strata dose: the nearest nominal level (uM) within
    `dose_level_tol_log10`, else the dose rounded to `dose_level_offgrid_decimals`
    d.p.; 0.0 for vehicles. Never a model input (the model sees log10_conc)."""
    a = cfg.action
    conc = np.asarray(conc, dtype=np.float64)
    is_control = np.asarray(is_control, dtype=bool)
    out = np.zeros(conc.shape[0], dtype=np.float64)
    trt = ~is_control
    if trt.any():
        lv = np.asarray(a.dose_levels_um, dtype=np.float64)
        dist = np.abs(np.log10(conc[trt])[:, None] - np.log10(lv)[None, :])
        j = dist.argmin(axis=1)
        near = dist[np.arange(j.size), j] <= a.dose_level_tol_log10
        out[trt] = np.where(near, lv[j], np.round(conc[trt], a.dose_level_offgrid_decimals))
    if (out[trt] <= 0).any():
        raise ValueError("a treated dose rounds to dose_level 0, colliding with the vehicle arm")
    return out


def build_compound_vocab(pert_ids: pd.Series) -> dict[str, int]:
    """Return {pert_id: int_idx} over TREATED pert_ids. Control gets index 0; rest sorted."""
    unique = sorted(set(pert_ids))
    vocab = {CONTROL_COMPOUND_NAME: CONTROL_COMPOUND_IDX}
    for i, name in enumerate(unique, start=1):
        vocab[name] = i
    return vocab


def limit_whole_plates(df: pd.DataFrame, limit: int, *, by_map: bool = False) -> pd.DataFrame:
    """Smoke subset: whole plates in sorted det_plate order until >= `limit` wells.
    Whole plates keep plate QC and DMSO-per-plate meaningful.

    `by_map` (a multi-line population): whole plate MAPS instead -- the first
    token of det_plate, i.e. every line's and replicate's plates of one layout --
    so each compound of the subset is still present in each line."""
    unit = df["det_plate"].str.split("_").str[0] if by_map else df["det_plate"]
    sizes = df.groupby(unit.values).size().sort_index()
    n_keep = int(np.searchsorted(sizes.cumsum().values, limit)) + 1
    keep = set(sizes.index[:n_keep])
    return df[unit.isin(keep).values].copy()


def common_compound_rows(df: pd.DataFrame, cell_ids) -> tuple[np.ndarray, dict]:
    """Keep-mask and record for the multi-line rule: a treated well stays only if
    its compound has a treated well in EVERY line of the population (after plate
    QC). Vehicles always stay. On a single-line population it keeps everything."""
    ctl = df["is_control"].values.astype(bool)
    lines = sorted(set(cell_ids))
    n_lines = df.loc[~ctl].groupby("pert_id")["cell_id"].nunique()
    ok = set(n_lines.index[n_lines == len(lines)])
    keep = ctl | df["pert_id"].isin(ok).values
    dropped = sorted(set(n_lines.index) - ok)
    rec = {"rule": ("a treated well is kept only if its pert_id has a treated well in every "
                    "cell line of the population, after plate QC; vehicles are always kept"),
           "applies": len(lines) > 1, "cell_ids": lines,
           "n_compounds_before": int(n_lines.size), "n_compounds_kept": int(len(ok)),
           "n_treated_wells_dropped": int((~keep).sum()),
           "dropped_compounds": [{"pert_id": p, "n_lines": int(n_lines[p])} for p in dropped]}
    return keep, rec


# ---------------------------------------------------------------------------
# covariates (copied from RxRx19a; declared levels win over a data scan)
# ---------------------------------------------------------------------------
def build_covariate_encoder(df: pd.DataFrame, cov_spec) -> dict:
    """Categorical -> one-hot column layout.
    Metadata only; `apply_covariate_encoder` does the encoding, so the layout can be reused.
    A field with DECLARED_LEVELS (syn_c) keeps its declared levels even when the
    data holds only some of them (e.g. a --limit build)."""
    levels: dict[str, list[str]] = {}
    for col in cov_spec.categorical_cols:
        if col in DECLARED_LEVELS:
            levels[col] = list(DECLARED_LEVELS[col])
        else:
            levels[col] = sorted_levels(df[col].unique().tolist())
    return {
        "categorical_cols": list(cov_spec.categorical_cols),
        "continuous_cols": list(cov_spec.continuous_cols),
        "levels": levels,
    }


def apply_covariate_encoder(df: pd.DataFrame, enc: dict) -> np.ndarray:
    """Return a (N, d) float32 array of one-hot covariates + continuous cols."""
    columns: list[np.ndarray] = []
    for col in enc["categorical_cols"]:
        cats = enc["levels"][col]
        col_vals = df[col].astype(str).values
        # one-hot via broadcast (avoid pandas.get_dummies to keep column order)
        oh = (col_vals[:, None] == np.array(cats)[None, :]).astype(np.float32)
        unseen = oh.sum(axis=1) != 1
        if unseen.any():
            raise ValueError(f"covariate {col!r}: values {sorted(set(col_vals[unseen]))[:5]} "
                             f"are not among the encoder levels {cats}")
        columns.append(oh)
    for col in enc["continuous_cols"]:
        columns.append(df[col].astype(np.float32).values.reshape(-1, 1))
    if not columns:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(columns, axis=1)


def covariate_encoder_path(cfg) -> str:
    return os.path.join(cfg.paths.nuisance_dir, "covariate_encoder.json")


def load_covariate_encoder(path: str, cov_spec) -> dict:
    """`covariate_encoder.json`, CHECKED against the current spec."""
    with open(path) as f:
        enc = json.load(f)
    want = tuple(cov_spec.categorical_cols)
    got = tuple(enc.get("categorical_cols", ()))
    if got != want:
        raise ValueError(
            f"covariate encoder at {path} was built for categorical_cols "
            f"{list(got)}, but spec.FIELDS now declares {list(want)}. cov_vec is "
            f"stale -- re-run `python -m src.data.build_dataset` to rebuild it.")
    return enc


def covariate_blocks(enc: dict) -> dict:
    """Field name -> its one-hot column indices in cov_vec, in layout order."""
    blocks, off = {}, 0
    for c in enc["categorical_cols"]:
        w = len(enc["levels"][c])
        blocks[c] = list(range(off, off + w))
        off += w
    return blocks


# ---------------------------------------------------------------------------
# GCTX
# ---------------------------------------------------------------------------
def gctx_well_positions(gctx_path: str, inst_ids) -> np.ndarray:
    """Matrix row of each inst_id, joined on the id (GCTX order != inst_info order)."""
    import h5py
    with h5py.File(gctx_path, "r") as f:
        ids = f[GCTX_WELL_IDS].asstr()[:]
    lut = pd.Series(np.arange(ids.size, dtype=np.int64), index=pd.Index(ids))
    if lut.index.duplicated().any():
        raise ValueError("duplicate inst_ids in the GCTX")
    inst_ids = pd.Index(inst_ids)
    missing = inst_ids[~inst_ids.isin(lut.index)]
    if len(missing):
        raise ValueError(f"{len(missing)} inst_ids absent from the GCTX, e.g. {list(missing[:3])}")
    return lut.loc[inst_ids].values.astype(np.int64)


def landmark_genes(gctx_path: str, gene_info: pd.DataFrame, n_genes: int) -> pd.DataFrame:
    """The landmark genes (pr_is_lm == 1) with their matrix column, in GCTX order."""
    import h5py
    with h5py.File(gctx_path, "r") as f:
        gids = f[GCTX_GENE_IDS].asstr()[:]
    if list(gids) != gene_info["pr_gene_id"].tolist():
        raise ValueError("GCTX gene ids are not gene_info.pr_gene_id in order")
    lm = gene_info["pr_is_lm"].values == "1"
    out = pd.DataFrame({
        "pr_gene_id": gene_info["pr_gene_id"].values[lm].astype(np.int64),
        "pr_gene_symbol": gene_info["pr_gene_symbol"].values[lm],
        "gctx_gene_pos": np.flatnonzero(lm).astype(np.int64),
    })
    if len(out) != n_genes:
        raise ValueError(f"{len(out)} landmark genes; OutcomeSpec.n_genes = {n_genes}")
    return out


def read_gctx_rows(gctx_path: str, rows: np.ndarray, cols: np.ndarray, *,
                   max_gap: int = 64, block_rows: int = 1024) -> np.ndarray:
    """(len(rows), len(cols)) float32, in the order of `rows`.

    Reads contiguous slabs of FULL matrix rows (sorted; gaps <= `max_gap` rows
    are read through; <= `block_rows` rows per slab) and slices `cols` in NumPy.
    No h5py fancy indexing on both axes of the contiguous dataset.
    """
    import h5py
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    srt = rows[order]
    if srt.size and (np.diff(srt) == 0).any():
        raise ValueError("duplicate rows requested")
    out = np.empty((rows.size, cols.size), dtype=np.float32)
    t0, n_read, n_slabs = time.time(), 0, 0
    with h5py.File(gctx_path, "r") as f:
        dset = f[GCTX_MATRIX]
        row_bytes = 4 * dset.shape[1]
        if srt.size and (srt[0] < 0 or srt[-1] >= dset.shape[0]):
            raise IndexError("row index out of range for the GCTX matrix")
        if cols.size and (cols.min() < 0 or cols.max() >= dset.shape[1]):
            raise IndexError("column index out of range for the GCTX matrix")
        i = 0
        while i < srt.size:
            j = i + 1
            while (j < srt.size and srt[j] - srt[j - 1] <= max_gap
                   and srt[j] - srt[i] < block_rows):
                j += 1
            lo, hi = int(srt[i]), int(srt[j - 1]) + 1
            slab = dset[lo:hi, :]
            out[order[i:j]] = slab[np.ix_(srt[i:j] - lo, cols)]
            n_read += hi - lo
            n_slabs += 1
            i = j
    print(f"[build] GCTX: {rows.size:,} wells via {n_slabs} slabs ({n_read:,} rows read, "
          f"{n_read * row_bytes / 2**30:.2f} GiB) in {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# plate QC (decision 6)
# ---------------------------------------------------------------------------
def plate_dmso_spread(expr: np.ndarray, plates: np.ndarray, is_control: np.ndarray) -> pd.DataFrame:
    """Per plate: n_dmso and spread = median over genes of 1.4826 * MAD over the
    plate's DMSO wells."""
    recs = []
    for p in sorted(set(plates)):
        x = expr[(plates == p) & is_control]
        if x.shape[0] < MIN_DMSO_PER_PLATE:
            raise ValueError(
                f"plate {p} has {x.shape[0]} DMSO wells (< {MIN_DMSO_PER_PLATE}); plate QC "
                f"and DMSO centring both need vehicle wells on every plate")
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        recs.append({"plate": p, "n_dmso": int(x.shape[0]),
                     "spread": float(np.median(MAD_TO_SD * mad))})
    return pd.DataFrame(recs).set_index("plate")


def plate_qc(df: pd.DataFrame, expr: np.ndarray, ratio: float) -> tuple[np.ndarray, dict]:
    """Keep-mask over rows and the QC record. A plate is dropped when its DMSO
    spread exceeds `ratio` x the median spread over plates of its cell line."""
    plates = df["det_plate"].values
    is_ctl = df["is_control"].values.astype(bool)
    sp = plate_dmso_spread(expr, plates, is_ctl)
    line = df.groupby("det_plate")["cell_id"].agg(lambda s: sorted(set(s)))
    if (line.str.len() != 1).any():
        raise ValueError("plates spanning more than one cell line")
    sp["cell_id"] = line.str[0].reindex(sp.index)
    med = sp.groupby("cell_id")["spread"].median()
    sp["ratio_to_median"] = sp["spread"] / sp["cell_id"].map(med)
    sp["dropped"] = sp["ratio_to_median"] > ratio
    dropped = sp.index[sp["dropped"]].tolist()
    keep = ~np.isin(plates, dropped)

    arm = arm_keys(df)
    trt = ~is_ctl
    n_before = pd.Series(arm[trt]).value_counts()
    n_after = pd.Series(arm[trt & keep]).value_counts().reindex(n_before.index, fill_value=0)
    n_wells = df.groupby("det_plate").size()
    rec = {
        "rule": (f"drop plates whose DMSO spread > {ratio} x the median spread over plates "
                 f"of the same cell_id; spread = median over genes of {MAD_TO_SD} * MAD "
                 f"over the plate's DMSO wells (raw Level 3)"),
        "max_spread_ratio": ratio,
        "median_spread": {k: float(v) for k, v in med.items()},
        "threshold": {k: float(v * ratio) for k, v in med.items()},
        "spread_range": [float(sp["spread"].min()), float(sp["spread"].max())],
        "dropped_plates": [{"plate": p, "spread": float(sp.at[p, "spread"]),
                            "ratio_to_median": float(sp.at[p, "ratio_to_median"]),
                            "n_wells": int(n_wells[p]), "n_dmso": int(sp.at[p, "n_dmso"])}
                           for p in dropped],
        "n_wells_dropped": int((~keep).sum()),
        "arms_emptied": sorted(n_after.index[n_after == 0].tolist()),
        # arm = (cell_id, pert_id, dose_level); single-well arms in total, and those QC caused
        "n_arms_single_well": {"before_qc": int((n_before == 1).sum()), "after_qc": int((n_after == 1).sum())},
        "n_arms_reduced_to_one_well": int(((n_before > 1) & (n_after == 1)).sum()),
        "plates": {p: {"cell_id": r.cell_id, "n_dmso": int(r.n_dmso), "spread": float(r.spread),
                       "ratio_to_median": float(r.ratio_to_median), "dropped": bool(r.dropped)}
                   for p, r in sp.iterrows()},
    }
    return keep, rec


def arm_keys(df: pd.DataFrame) -> np.ndarray:
    """Treated arm = (cell_id, pert_id, dose_level); vehicles share one arm per cell line."""
    dl = [f"{v:.6g}" for v in df["dose_level"].values]
    ctl = df["is_control"].values.astype(bool)
    return np.array([f"{c}|{CONTROL_COMPOUND_NAME if k else p}|{d}"
                     for c, p, d, k in zip(df["cell_id"].values, df["pert_id"].values, dl, ctl)],
                    dtype=object)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _atomic_save_dir(write_fn, dest: str) -> None:
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    write_fn(tmp)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.replace(tmp, dest)


def _atomic_write(path: str, write_fn, mode: str = "w") -> None:
    tmp = path + ".tmp"
    with open(tmp, mode) as f:
        write_fn(f)
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    _atomic_write(path, lambda f: json.dump(obj, f, indent=2))


def output_paths(paths: Paths) -> list[str]:
    return [paths.tabular_dataset_dir, paths.expr_npy, paths.gene_order_json,
            paths.population_qc_json, paths.context_encoder_json,
            os.path.join(paths.nuisance_dir, "compound_vocab.json"),
            os.path.join(paths.nuisance_dir, "covariate_encoder.json"),
            os.path.join(paths.nuisance_dir, "nuisance_meta.json")]


def downstream_artifacts(paths: Paths) -> list[str]:
    """Files under data_dir that build_dataset did not write (splits, expr_meta,
    nuisances, tagged nuisance dirs, ...), ignoring leftover .tmp files."""
    ours = {os.path.normpath(p) for p in output_paths(paths)}
    found = []
    for root, dirs, files in os.walk(paths.data_dir) if os.path.isdir(paths.data_dir) else ():
        if os.path.normpath(root) in ours:      # inside lincs_tabular/
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.endswith(".tmp")]
        for f in files:
            full = os.path.normpath(os.path.join(root, f))
            if full not in ours and not f.endswith(".tmp"):
                found.append(os.path.relpath(full, paths.data_dir))
    return sorted(found)


def summarize(df: pd.DataFrame, expr: np.ndarray, vocab: dict, qc: dict, cfg: CaseConfig) -> dict:
    """The Phase 0 smoke numbers; printed and stored in population_qc.json."""
    ctl = df["is_control"].values.astype(bool)
    dmso = df.loc[ctl].groupby("det_plate").size()
    arm_sizes = pd.Series(arm_keys(df)[~ctl]).value_counts()
    dl = df.loc[~ctl, "dose_level"].value_counts()
    on_grid = [float(v) for v in cfg.action.dose_levels_um]
    return {
        "n_wells": int(len(df)),
        "n_treated": int((~ctl).sum()),
        "n_vehicle": int(ctl.sum()),
        "vehicle_fraction": float(ctl.mean()),
        "n_compounds": int(df.loc[~ctl, "pert_id"].nunique()),
        "compound_vocab_size": int(len(vocab)),
        "n_plates": int(df["det_plate"].nunique()),
        "n_plate_maps": int(df["plate_map"].nunique()),
        "n_batches": int(df["batch"].nunique()),
        "expr_shape": list(expr.shape),
        "n_arms": int(arm_sizes.size),
        "arm_size_counts": {int(k): int(v) for k, v in arm_sizes.value_counts().sort_index().items()},
        "dose_level_counts": {f"{k:g}": int(v) for k, v in dl.sort_index().items() if k in on_grid},
        "dose_level_offgrid": {"n_wells": int(dl[~dl.index.isin(on_grid)].sum()),
                               "n_levels": int((~dl.index.isin(on_grid)).sum()),
                               "n_compounds": int(df.loc[~ctl & ~df["dose_level"].isin(on_grid), "pert_id"].nunique())},
        "dmso_per_plate": {"min": int(dmso.min()), "median": float(dmso.median()), "max": int(dmso.max()),
                           "by_plate": {k: int(v) for k, v in dmso.items()}},
        "plates_dropped_by_qc": [d["plate"] for d in qc["dropped_plates"]],
        "expr_range": [float(expr.min()), float(expr.max())],
        **({"by_cell_id": {c: {"n_wells": int(m.sum()), "n_treated": int((m & ~ctl).sum()),
                               "n_vehicle": int((m & ctl).sum()),
                               "n_plates": int(df.loc[m, "det_plate"].nunique()),
                               "n_compounds": int(df.loc[m & ~ctl, "pert_id"].nunique())}
                           for c in sorted(set(df["cell_id"]))
                           for m in [(df["cell_id"] == c).values]}}
           if df["cell_id"].nunique() > 1 else {}),
    }


def print_summary(s: dict) -> None:
    print("\n[build] ===== population summary =====")
    print(f"  n wells            {s['n_wells']:,}  (treated {s['n_treated']:,}, vehicle {s['n_vehicle']:,})")
    print(f"  vehicle fraction   {s['vehicle_fraction']:.4f}")
    print(f"  compounds          {s['n_compounds']:,} pert_ids (vocab {s['compound_vocab_size']:,} incl. __control__)")
    print(f"  plates             {s['n_plates']} ({s['n_plate_maps']} plate maps, {s['n_batches']} batches)")
    print(f"  Y (landmarks)      {tuple(s['expr_shape'])} float32, range [{s['expr_range'][0]:.2f}, {s['expr_range'][1]:.2f}]")
    print(f"  arms               {s['n_arms']:,}; wells/arm -> #arms {s['arm_size_counts']}")
    print(f"  dose_level counts  {s['dose_level_counts']}")
    og = s["dose_level_offgrid"]
    print(f"  off-grid doses     {og['n_wells']} wells, {og['n_levels']} levels, {og['n_compounds']} compounds")
    d = s["dmso_per_plate"]
    print(f"  DMSO per plate     min {d['min']}  median {d['median']:g}  max {d['max']}")
    if len(d["by_plate"]) <= 12:
        for p, n in d["by_plate"].items():
            print(f"      {p:<32} {n}")
    print(f"  plates dropped QC  {s['plates_dropped_by_qc'] or 'none'}")
    for c, r in (s.get("by_cell_id") or {}).items():
        print(f"  line {c:<6}        {r['n_wells']:,} wells (treated {r['n_treated']:,}, vehicle "
              f"{r['n_vehicle']:,}) on {r['n_plates']} plates, {r['n_compounds']:,} compounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: keep whole plates (sorted) until >= N wells. Writes to "
                             "data/<population>_limit<N>/ unless --data_dir is given.")
    parser.add_argument("--data_dir", default=None, help="Override cfg.paths.data_dir (all outputs go under it).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing build in the target dir.")
    parser.add_argument("--block_rows", type=int, default=1024, help="Max GCTX rows per contiguous read.")
    parser.add_argument("--population", default=None, choices=sorted(POPULATIONS),
                        help="Which declared population to build (default mcf7_24h).")
    args = parser.parse_args()

    cfg: CaseConfig = CaseConfig() if args.population is None else CaseConfig(
        population=POPULATIONS[args.population])
    pop = cfg.population
    multi_line = len(pop.cell_ids) > 1
    data_dir = args.data_dir
    if data_dir is None and args.limit is not None:
        data_dir = os.path.join(os.path.dirname(cfg.paths.data_dir), f"{pop.name}_limit{args.limit}")
    if data_dir is not None:
        cfg.paths = Paths(population=pop.name, raw_dir=cfg.paths.raw_dir, data_dir=os.path.abspath(data_dir))
    paths = cfg.paths
    print(f"[build] population {pop.key()}  ->  {paths.data_dir}")
    print(f"[build] roles: {format_role_summary(cfg)}")

    existing = [p for p in output_paths(paths) if os.path.exists(p)]
    if existing and not args.overwrite:
        raise SystemExit(f"[build] {len(existing)} outputs already exist under {paths.data_dir} "
                         f"(e.g. {existing[0]}); pass --overwrite to replace them.")
    stale = downstream_artifacts(paths)
    if stale:
        # splits / fold assignments / weights index rows of the old table; a new
        # table would silently re-point them at different wells.
        raise SystemExit(f"[build] {paths.data_dir} holds artifacts built on the current table "
                         f"({stale[:6]}{' ...' if len(stale) > 6 else ''}); move or delete them "
                         f"before rebuilding, then rebuild them from the new table.")

    # --- metadata + population ---------------------------------------------
    t0 = time.time()
    inst = read_tsv(paths.raw("inst_info"))
    gene_info = read_tsv(paths.raw("gene_info"))
    pert_info = read_tsv(paths.raw("pert_info"))
    cell_info = read_tsv(paths.raw("cell_info"))
    df, filter_counts = select_population(inst, cfg)
    print(f"[build] filters: {filter_counts}")
    check_metadata(df, cfg, pert_info, cell_info)
    df = df.sort_values("inst_id", kind="stable").reset_index(drop=True)
    if args.limit is not None:
        df = limit_whole_plates(df, args.limit, by_map=multi_line).reset_index(drop=True)
        print(f"[build] --limit {args.limit}: {len(df):,} wells on {df['det_plate'].nunique()} whole plates"
              + (f" ({df['det_plate'].str.split('_').str[0].nunique()} whole plate maps)" if multi_line else ""))

    df = encode_action(df, cfg)
    df = pd.concat([df, parse_plate(df["det_plate"])], axis=1)
    rows, cols = well_row_col(df["det_well"])
    df["well_row"], df["well_col"] = rows, cols.astype(np.int16)
    df["pert_time"] = pd.to_numeric(df["pert_time"]).astype(np.float32)

    # --- Y: 978 landmarks from the GCTX --------------------------------------
    gctx = paths.raw("gctx")
    genes = landmark_genes(gctx, gene_info, cfg.outcome.n_genes)
    df["gctx_col"] = gctx_well_positions(gctx, df["inst_id"].values)
    expr = read_gctx_rows(gctx, df["gctx_col"].values, genes["gctx_gene_pos"].values,
                          block_rows=args.block_rows)
    if not np.isfinite(expr).all():
        raise ValueError(f"{int((~np.isfinite(expr)).sum())} non-finite Level 3 values")

    # --- plate QC -------------------------------------------------------------
    keep, qc = plate_qc(df, expr, cfg.outcome.plate_qc_max_spread_ratio)
    for d in qc["dropped_plates"]:
        print(f"[build] plate QC: drop {d['plate']} (spread {d['spread']:.3f} = "
              f"{d['ratio_to_median']:.2f} x median; {d['n_wells']} wells)")
    print(f"[build] plate QC: spreads {qc['spread_range'][0]:.3f}-{qc['spread_range'][1]:.3f}, "
          f"median {qc['median_spread']}; {qc['n_wells_dropped']} wells dropped, "
          f"{len(qc['arms_emptied'])} arms emptied, {qc['n_arms_reduced_to_one_well']} arms reduced to 1 well "
          f"(single-well arms {qc['n_arms_single_well']['before_qc']} -> {qc['n_arms_single_well']['after_qc']})")
    df = df.loc[keep].reset_index(drop=True)
    expr = np.ascontiguousarray(expr[keep])

    # --- multi-line populations: compounds present in every line -----------------
    keep_c, common = common_compound_rows(df, pop.cell_ids)
    if multi_line:
        print(f"[build] every-line rule: {common['n_compounds_kept']:,}/{common['n_compounds_before']:,} "
              f"compounds have a treated well in all {len(pop.cell_ids)} lines; "
              f"{common['n_treated_wells_dropped']:,} treated wells dropped")
        df = df.loc[keep_c].reset_index(drop=True)
        expr = np.ascontiguousarray(expr[keep_c])
    elif not keep_c.all():
        raise AssertionError("the every-line rule dropped rows of a single-line population")

    # --- compound vocab --------------------------------------------------------
    ctl = df["is_control"].values.astype(bool)
    vocab = build_compound_vocab(df.loc[~ctl, "pert_id"])
    df["compound_idx"] = np.where(ctl, CONTROL_COMPOUND_IDX,
                                  df["pert_id"].map(vocab).fillna(-1).values).astype(np.int32)
    if (df["compound_idx"] < 0).any():
        raise AssertionError("treated pert_id missing from the vocab")
    print(f"[build] compound vocab size: {len(vocab)} (incl. control)")

    # --- syn_c: balanced per arm and per plate's DMSO (step C, inert in v1) ----
    groups = np.where(ctl, "dmso|" + df["det_plate"].values, "arm|" + arm_keys(df))
    df["syn_c"] = assign_syn_c(groups, df["inst_id"].values, cfg.outcome.syn_seed)
    syn_rec = {"seed": cfg.outcome.syn_seed,
               "groups": "treated: (cell_id, pert_id, dose_level) arm; vehicle: det_plate",
               "n_syn_c_1": int(df["syn_c"].sum()), "n_syn_c_0": int((df["syn_c"] == 0).sum()),
               "max_arm_imbalance": max_group_imbalance(groups[~ctl], df["syn_c"].values[~ctl]),
               "max_plate_dmso_imbalance": max_group_imbalance(groups[ctl], df["syn_c"].values[ctl])}
    if max(syn_rec["max_arm_imbalance"], syn_rec["max_plate_dmso_imbalance"]) > 1:
        raise AssertionError(f"syn_c is not balanced: {syn_rec}")

    # --- covariates -------------------------------------------------------------
    enc = build_covariate_encoder(df, cfg.covariates)
    cov_mat = apply_covariate_encoder(df, enc)
    print(f"[build] covariate vector dim: {cov_mat.shape[1]} {enc['levels']}")

    # --- context encoder, from the FILTERED table ---------------------------------
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    ctx_enc = ContextEncoder.build(df)
    ctx_enc.encode(df)  # every row encodes (raises on an unseen level)
    print(f"[build] context fields {list(CONTEXT_FIELDS)} cardinalities {ctx_enc.cardinalities}")

    # --- assemble + save ------------------------------------------------------------
    out_df = df[TABLE_COLUMNS].copy()
    out_df["cov_vec"] = list(cov_mat)
    fingerprint = hashlib.sha1(("\n".join(out_df["inst_id"]) + "\n#genes\n"
                                + ",".join(map(str, genes["pr_gene_id"]))).encode()).hexdigest()[:16]
    summary = summarize(df, expr, vocab, qc, cfg)
    print_summary(summary)

    from datasets import Dataset

    os.makedirs(paths.data_dir, exist_ok=True)
    os.makedirs(paths.nuisance_dir, exist_ok=True)
    ds = Dataset.from_pandas(out_df, preserve_index=False)
    _atomic_save_dir(ds.save_to_disk, paths.tabular_dataset_dir)
    _atomic_write(paths.expr_npy, lambda f: np.save(f, expr.astype(np.float32, copy=False)), mode="wb")
    _write_json(paths.gene_order_json, {
        "n_genes": int(len(genes)),
        "pr_gene_id": genes["pr_gene_id"].tolist(),
        "pr_gene_symbol": genes["pr_gene_symbol"].tolist(),
        "gctx_gene_pos": genes["gctx_gene_pos"].tolist(),
        "table_fingerprint": fingerprint,
    })
    ctx_enc.save(paths.context_encoder_json)
    _write_json(os.path.join(paths.nuisance_dir, "compound_vocab.json"), vocab)
    _write_json(os.path.join(paths.nuisance_dir, "covariate_encoder.json"), enc)
    # n_compounds / cov_dim here, so the trainer does not depend on fit_urr for them.
    _write_json(os.path.join(paths.nuisance_dir, "nuisance_meta.json"), {
        "n_compounds": int(len(vocab)), "cov_dim": int(cov_mat.shape[1]),
        "population": pop.name, "limit": args.limit, "table_fingerprint": fingerprint})
    _write_json(paths.population_qc_json, {
        "population": pop.name,
        "limit": args.limit,
        "table_fingerprint": fingerprint,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {k: os.path.basename(paths.raw(k)) for k in ("inst_info", "gene_info", "pert_info", "cell_info", "gctx")},
        "decisions": decisions_record(cfg),
        "filters": filter_counts,
        "plate_qc": qc,
        # Recorded only for a multi-line population, so a single-line build's
        # population_qc.json keeps the keys it has always had.
        **({"common_compounds": common, "line_groups": line_groups(pop)} if multi_line else {}),
        "syn_c": syn_rec,
        "summary": summary,
    })
    print(f"\n[build] wrote {paths.data_dir} (table {len(out_df):,} rows, fingerprint {fingerprint}) "
          f"in {time.time() - t0:.0f}s")


CONTEXT_FIELDS = _CONTEXT_FIELDS

_WELL_RE = re.compile(r"^([A-Z]+)(\d+)$")

# Table columns ContextEncoder reads.
CONTEXT_SOURCE_COLUMNS = ("det_plate", "cell_id", "syn_c", "det_well", "pert_time")


def sorted_levels(values) -> list:
    """Ordered level list for a categorical field: numeric order when every level parses as a number."""
    uniq = list(dict.fromkeys(str(v) for v in values))
    try:
        return sorted(uniq, key=lambda v: (0, float(v)))
    except (TypeError, ValueError):
        return sorted(uniq)


def context_raw(df: pd.DataFrame) -> dict[str, list[str]]:
    """Context field -> per-row level string, from the table's columns."""
    rows, cols = well_row_col(df["det_well"])
    return {
        "plate": df["det_plate"].astype(str).tolist(),
        "cell_id": df["cell_id"].astype(str).tolist(),
        "syn_c": [str(int(v)) for v in df["syn_c"]],
        "well_row": rows.tolist(),
        "well_col": [str(int(c)) for c in cols],
        "pert_time": [f"{float(v):g}" for v in df["pert_time"]],
    }


@dataclass
class ContextEncoder:
    """Maps table rows -> (F,) integer context codes. Levels are strings."""

    levels: dict[str, list]          # field -> ordered level list (categoricals)

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.levels[f]) for f in CONTEXT_FIELDS]

    # -- build / persist ----------------------------------------------------
    @classmethod
    def build(cls, df: pd.DataFrame) -> "ContextEncoder":
        """From the FILTERED table (never inst_info: 346k wells over 98 cell lines)."""
        raw = context_raw(df)
        levels = {f: sorted_levels(v) for f, v in raw.items() if f not in DECLARED_LEVELS}
        # levels: the set is fixed by definition and its ORDER is the index a checkpoint was trained on.
        levels.update({k: list(v) for k, v in DECLARED_LEVELS.items()})
        missing = [f for f in CONTEXT_FIELDS if f not in levels]
        if missing:
            raise KeyError(
                f"spec.FIELDS declares {missing}, which ContextEncoder.build does "
                f"not know how to enumerate levels for. Add it to `context_raw`.")
        return cls(levels={f: levels[f] for f in CONTEXT_FIELDS})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_json(path, {"fields": list(CONTEXT_FIELDS), "levels": self.levels})

    @classmethod
    def load(cls, path: str) -> "ContextEncoder":
        with open(path) as f:
            d = json.load(f)
        if tuple(d["fields"]) != CONTEXT_FIELDS:
            raise ValueError(
                f"context encoder at {path} was built for fields {d['fields']}, "
                f"but the code now expects {list(CONTEXT_FIELDS)}"
            )
        return cls(levels=d["levels"])

    @classmethod
    def load_or_build(cls, cfg) -> "ContextEncoder":
        """build_dataset writes the encoder eagerly; the fallback rebuilds it
        from the filtered table, never from inst_info."""
        path = context_encoder_path(cfg)
        if os.path.isfile(path):
            return cls.load(path)
        from datasets import load_from_disk
        ds = load_from_disk(cfg.paths.tabular_dataset_dir)
        enc = cls.build(pd.DataFrame({c: ds[c] for c in CONTEXT_SOURCE_COLUMNS}))
        enc.save(path)
        return enc

    # -- encode -------------------------------------------------------------
    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """(N, F) int64 context codes. An unseen level raises (the encoder is
        built from the same table it encodes)."""
        raw = context_raw(df)
        missing = [f for f in CONTEXT_FIELDS if f not in raw]
        if missing:
            raise KeyError(
                f"spec.FIELDS declares {missing}, which ContextEncoder.encode does "
                f"not know how to compute. Add it to `context_raw`.")
        out = np.zeros((len(df), len(CONTEXT_FIELDS)), dtype=np.int64)
        for j, f in enumerate(CONTEXT_FIELDS):
            lut = {str(lv): i for i, lv in enumerate(self.levels[f])}
            codes = [lut.get(v, -1) for v in raw[f]]
            out[:, j] = codes
            if (out[:, j] < 0).any():
                bad = sorted({v for v, c in zip(raw[f], codes) if c < 0})
                raise ValueError(f"context field {f!r}: unseen levels {bad[:5]}")
        return out

    def encode_row(self, row) -> np.ndarray:
        """Single HF-dataset row (a dict) -> (F,) int64."""
        return self.encode(pd.DataFrame([{c: row[c] for c in CONTEXT_SOURCE_COLUMNS}]))[0]


def _well_row_col(well: "pd.Series") -> tuple[np.ndarray, np.ndarray]:
    r, c = [], []
    for w in well.astype(str):
        m = _WELL_RE.match(w.strip())
        if not m:
            raise ValueError(f"det_well {w!r} is not of the form A01")
        r.append(m.group(1)); c.append(int(m.group(2)))
    return np.array(r, dtype=object), np.array(c, dtype=np.int64)


def well_row_col(well) -> tuple[np.ndarray, np.ndarray]:
    """(row letters, column numbers) for a well Series."""
    return _well_row_col(pd.Series(well).astype(str))


def context_encoder_path(cfg) -> str:
    return cfg.paths.context_encoder_json


def context_for_rows(cfg, meta, pick: np.ndarray) -> np.ndarray:
    """(n, F) context codes for `pick` rows of the HF tabular dataset `meta`.

    Eval uses this to generate at contexts from the empirical p(C, E) --- the marginalization
    """
    enc = ContextEncoder.load_or_build(cfg)
    df = pd.DataFrame({c: np.asarray(meta[c])[pick] for c in CONTEXT_SOURCE_COLUMNS})
    return enc.encode(df)

if __name__ == "__main__":
    main()
