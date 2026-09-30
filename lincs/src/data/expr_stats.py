"""Plate DMSO centres and per-gene z-score statistics from TRAIN rows.

New for LINCS (IMPLEMENT.md §3.2; decisions 5 and 6; P10, P11). Runs after the
split (build_tiered_split) and writes <nuisance_dir>/expr_meta.json. "TRAIN" is
the split's unthinned train pool, nu_rows.npy: the train rows in v1, and in a
thinning instance the train rows before thinning, so every gamma instance of
one seed shares one z-scale (DMSO is never thinned, so centres and mean are
the same either way).

    centre[p]  per-gene median of plate p's TRAIN DMSO wells
               (plate_center="dmso_median_train"; all zeros for the "none" ablation)
    mean       per-gene mean of the centred TRAIN DMSO wells (normalize_mean="train_dmso"),
               so z = 0 is the vehicle in every gene
    std        per-gene std of ALL centred TRAIN rows (normalize_std="train_all")
    cap_frac   per-gene fraction of TRAIN values at the Level 3 cap of 15.0
               (GAPDH ~41%); reported only, every gene is kept
    gate       the centring gate on HELD-OUT DMSO wells (P11; below)

dataset.py applies y = (x - centre[plate] - mean) / std via `normalize_expr`, and
refuses an expr_meta.json built on another split or table.

Centring gate (P11). On held-out DMSO wells, which did not set the centres:
  - per-plate means ~ 0: |held-out plate mean| / its sampling SE, where
    SE = s * sqrt(1/n_holdout + (pi/2)/n_train) (s = pooled within-plate SD of
    the plate's train DMSO; pi/2 / n is the variance of a median). The median
    over plates x genes is ~0.67 when the centres are right.
  - the plate's share of DMSO variance (one-way ANOVA per gene, median over
    genes) drops from the raw Level 3 value (~0.67) toward chance. With ~4
    held-out wells per plate, chance R^2 is (P-1)/(n-1) ~ 0.24, higher than in
    the §3.2 measurement, and R^2 after centring sits near chance plus centre
    noise (~0.3). The gate therefore uses the chance-corrected share
    omega^2 = (SSB - (P-1) MSW) / (SST + MSW) and reports R^2.
  Pass: median |z| <= 1.0, omega^2 after <= 0.20 and <= 0.5 x omega^2 before.

    python -m src.data.expr_stats [--data_dir ...] [--nuisance_dir ...] [--plate_center none]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import _write_json  # noqa: E402
from src.data.build_nu_rows import NU_ROWS_FILENAME  # noqa: E402
from src.data.splits import load_splits, population_key  # noqa: E402
from src.spec import (  # noqa: E402
    PLATE_CENTER_MODES, CaseConfig, add_adjustment_set_cli, add_paths_cli, apply_paths_args,
    config_from_args)

EXPR_META = "expr_meta.json"
LEVEL3_CAP = 15.0
MIN_TRAIN_DMSO_PER_PLATE = 3
STD_FLOOR = 1e-6

# Centring gate thresholds (P11; see module docstring).
GATE_MAX_PLATE_MEAN_Z = 1.0
GATE_MAX_OMEGA2_RATIO = 0.5
GATE_MAX_OMEGA2 = 0.20


def expr_meta_filename(plate_center: str) -> str:
    """expr_meta.json for the decision-6 default; expr_meta_<mode>.json for an ablation."""
    if plate_center not in PLATE_CENTER_MODES:
        raise ValueError(f"plate_center={plate_center!r}; expected one of {PLATE_CENTER_MODES}")
    return EXPR_META if plate_center == PLATE_CENTER_MODES[0] else f"expr_meta_{plate_center}.json"


def read_outcome_inputs(cfg: CaseConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(expr (N, G) float32 raw Level 3, det_plate (N,) str, is_control (N,) bool), table row order."""
    from datasets import load_from_disk
    meta = load_from_disk(cfg.paths.tabular_dataset_dir).select_columns(["det_plate", "is_control"])
    plates = np.asarray(meta["det_plate"], dtype=object)
    ctl = np.asarray(meta["is_control"]).astype(bool)
    expr = np.load(cfg.paths.expr_npy)
    if expr.shape != (plates.size, cfg.outcome.n_genes):
        raise ValueError(f"expr.npy {expr.shape} does not match the table ({plates.size:,} rows x "
                         f"{cfg.outcome.n_genes} genes)")
    return expr, plates, ctl


def compute_expr_stats(expr: np.ndarray, plates: np.ndarray, is_control: np.ndarray,
                       train_idx: np.ndarray, plate_center: str) -> dict:
    """Centres and z-score stats from the TRAIN rows only (float64)."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    levels = sorted(set(plates.tolist()))
    code = {p: i for i, p in enumerate(levels)}
    pc = np.array([code[p] for p in plates], dtype=np.int64)
    tr_dmso = train_idx[is_control[train_idx]]
    n_dmso = np.bincount(pc[tr_dmso], minlength=len(levels))
    G = expr.shape[1]
    centre = np.zeros((len(levels), G), dtype=np.float64)
    if plate_center == "dmso_median_train":
        short = [levels[i] for i in np.flatnonzero(n_dmso < MIN_TRAIN_DMSO_PER_PLATE)]
        if short:
            raise ValueError(f"plates with < {MIN_TRAIN_DMSO_PER_PLATE} train DMSO wells cannot be "
                             f"centred: {short[:5]}")
        for i in range(len(levels)):
            centre[i] = np.median(expr[tr_dmso[pc[tr_dmso] == i]].astype(np.float64), axis=0)
    elif plate_center != "none":
        raise ValueError(f"plate_center={plate_center!r}")
    xc = expr[train_idx].astype(np.float64) - centre[pc[train_idx]]
    mean = xc[is_control[train_idx]].mean(axis=0)
    std = xc.std(axis=0)
    n_floor = int((std < STD_FLOOR).sum())
    std = np.maximum(std, STD_FLOOR)
    cap_frac = (expr[train_idx] >= LEVEL3_CAP).mean(axis=0)
    return {"plates": levels, "n_train_dmso_per_plate": n_dmso.tolist(), "centre": centre,
            "mean": mean, "std": std, "cap_frac": cap_frac, "n_std_floored": n_floor}


def plate_codes(plates, levels: list[str]) -> np.ndarray:
    """(N,) index of each row's plate in `levels`; raises on a plate without a centre."""
    code = {p: i for i, p in enumerate(levels)}
    missing = sorted({p for p in plates if p not in code})
    if missing:
        raise ValueError(f"plates without a centre in expr_meta: {missing[:5]}")
    return np.array([code[p] for p in plates], dtype=np.int64)


def normalize_expr(x: np.ndarray, plate_code: np.ndarray, stats: dict) -> np.ndarray:
    """(n, G) raw Level 3 -> z-space: (x - centre[plate] - mean) / std, float32. Never clamped."""
    z = (np.asarray(x, dtype=np.float64) - stats["centre"][plate_code] - stats["mean"]) / stats["std"]
    return z.astype(np.float32)


def denormalize_expr(z: np.ndarray, plate_code: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of `normalize_expr`: z-space -> raw Level 3 scale."""
    return (np.asarray(z, dtype=np.float64) * stats["std"] + stats["mean"]
            + stats["centre"][plate_code]).astype(np.float32)


def _anova_share(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column of y: one-way ANOVA R^2 = SSB/SST and omega^2 by `groups`."""
    _, g = np.unique(groups, return_inverse=True)
    P, n = int(g.max()) + 1, y.shape[0]
    cnt = np.bincount(g, minlength=P).astype(np.float64)
    gm = np.zeros((P, y.shape[1]))
    np.add.at(gm, g, y)
    gm /= cnt[:, None]
    ybar = y.mean(axis=0)
    sst = ((y - ybar) ** 2).sum(axis=0)
    ssb = (cnt[:, None] * (gm - ybar) ** 2).sum(axis=0)
    msw = (sst - ssb) / max(n - P, 1)
    return ssb / sst, (ssb - (P - 1) * msw) / (sst + msw)


def centring_gate(expr: np.ndarray, plates: np.ndarray, is_control: np.ndarray,
                  train_idx: np.ndarray, holdout_idx: np.ndarray, stats: dict) -> dict:
    """P11: per-plate means ~ 0 and the plate's variance share on HELD-OUT DMSO.
    `train_idx` = the rows the stats were fit on (their DMSO wells set the centres)."""
    pcode = plate_codes(plates, stats["plates"])
    tr = np.asarray(train_idx, dtype=np.int64)
    ho = np.asarray(holdout_idx, dtype=np.int64)
    T, H = tr[is_control[tr]], ho[is_control[ho]]
    if H.size == 0:
        raise ValueError("no held-out DMSO wells: the centring gate needs them")
    x_t = expr[T].astype(np.float64)
    x_h = expr[H].astype(np.float64)
    c_h = x_h - stats["centre"][pcode[H]]
    # pooled within-plate SD of the train DMSO wells, per gene
    P = len(stats["plates"])
    cnt_t = np.bincount(pcode[T], minlength=P).astype(np.float64)
    m_t = np.zeros((P, x_t.shape[1]))
    np.add.at(m_t, pcode[T], x_t)
    m_t /= np.maximum(cnt_t, 1)[:, None]
    s = np.sqrt(((x_t - m_t[pcode[T]]) ** 2).sum(axis=0) / (T.size - int((cnt_t > 0).sum())))
    # held-out per-plate means vs their sampling SE, with and without the plate centre
    glob = np.median(x_t, axis=0)                      # one centre for all plates ("no centring")
    zs, zs_raw, n_plates = [], [], 0
    for p in np.unique(pcode[H]):
        m = pcode[H] == p
        se = s * math.sqrt(1.0 / m.sum() + (math.pi / 2) / cnt_t[p])
        zs.append(np.abs(c_h[m].mean(axis=0)) / se)
        zs_raw.append(np.abs((x_h[m] - glob).mean(axis=0)) / se)
        n_plates += 1
    zs, zs_raw = np.concatenate(zs), np.concatenate(zs_raw)
    r2_raw, om_raw = _anova_share(x_h, pcode[H])
    r2_cen, om_cen = _anova_share(c_h, pcode[H])
    z_h = c_h - stats["mean"]
    out = {
        "n_holdout_dmso": int(H.size), "n_plates": n_plates,
        "plate_mean_abs_z": {"median_centred": float(np.median(zs)), "median_uncentred": float(np.median(zs_raw)),
                             "expected_if_centred": 0.674, "p95_centred": float(np.quantile(zs, 0.95))},
        "grand_mean_z_abs_median": float(np.median(np.abs(z_h.mean(axis=0) / stats["std"]))),
        "plate_share_r2": {"before": float(np.median(r2_raw)), "after": float(np.median(r2_cen)),
                           "chance": float((n_plates - 1) / (H.size - 1))},
        "plate_share_omega2": {"before": float(np.median(om_raw)), "after": float(np.median(om_cen))},
        "thresholds": {"plate_mean_abs_z_max": GATE_MAX_PLATE_MEAN_Z, "omega2_ratio_max": GATE_MAX_OMEGA2_RATIO,
                       "omega2_after_max": GATE_MAX_OMEGA2},
    }
    om = out["plate_share_omega2"]
    out["checks"] = {
        "plate_means_zero": out["plate_mean_abs_z"]["median_centred"] <= GATE_MAX_PLATE_MEAN_Z,
        "omega2_small": om["after"] <= GATE_MAX_OMEGA2,
        "omega2_halved": om["after"] <= GATE_MAX_OMEGA2_RATIO * om["before"],
    }
    out["pass"] = all(out["checks"].values())
    return out


def load_expr_meta(cfg: CaseConfig, plate_center: str | None = None, splits: dict | None = None) -> dict:
    """expr_meta for cfg's split, arrays as float64 numpy. Raises when it was built
    on another split, table or population."""
    plate_center = plate_center or cfg.outcome.plate_center
    path = os.path.join(cfg.paths.nuisance_dir, expr_meta_filename(plate_center))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {os.path.basename(path)} in {cfg.paths.nuisance_dir}; run "
                                f"`python -m src.data.expr_stats` (plate_center={plate_center})")
    with open(path) as f:
        m = json.load(f)
    s = splits or load_splits(cfg)
    if m["split_fingerprint"] != s["split_fingerprint"] or m["population"] != s["population"]:
        raise ValueError(f"{path} was built on split {m['split_fingerprint']} / {m['population']}, not the "
                         f"current {s['split_fingerprint']} / {s['population']}; re-run src.data.expr_stats")
    for k in ("centre", "mean", "std", "cap_frac"):
        m[k] = np.asarray(m[k], dtype=np.float64)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plate_center", default=None, choices=PLATE_CENTER_MODES,
                   help="Default: cfg.outcome.plate_center (decision 6). 'none' is the ablation.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    a = p.parse_args()
    cfg = apply_paths_args(config_from_args(a), a)
    mode = a.plate_center or cfg.outcome.plate_center
    s = load_splits(cfg)
    rows = np.sort(np.load(os.path.join(cfg.paths.nuisance_dir, NU_ROWS_FILENAME)).astype(np.int64))
    if (not np.isin(s["train_idx"], rows).all() or np.isin(rows, s["holdout_idx"]).any()
            or np.isin(rows, s["reserve_idx"]).any()):
        raise ValueError(f"{NU_ROWS_FILENAME} is not this split's unthinned train pool; rebuild the split dir")
    expr, plates, ctl = read_outcome_inputs(cfg)
    st = compute_expr_stats(expr, plates, ctl, rows, mode)
    gate = centring_gate(expr, plates, ctl, rows, s["holdout_idx"], st)

    with open(cfg.paths.gene_order_json) as f:
        go = json.load(f)
    sym = np.asarray(go["pr_gene_symbol"])
    top = np.argsort(-st["cap_frac"])[:10]
    o = cfg.outcome
    summary = {
        "stats_rows": f"{NU_ROWS_FILENAME} (the unthinned train pool)",
        "n_train": int(rows.size), "n_train_dmso": int(ctl[rows].sum()),
        "train_dmso_per_plate": {"min": int(min(st["n_train_dmso_per_plate"])),
                                 "max": int(max(st["n_train_dmso_per_plate"]))},
        "std": {"min": float(st["std"].min()), "median": float(np.median(st["std"])), "max": float(st["std"].max())},
        "n_std_floored": st["n_std_floored"],
        "cap": {"n_genes_over_1pct": int((st["cap_frac"] > 0.01).sum()),
                "top": [{"gene": str(sym[i]), "pr_gene_id": int(go["pr_gene_id"][i]),
                         "cap_frac": float(st["cap_frac"][i])} for i in top]},
    }
    out = {
        "population": population_key(cfg), "split_fingerprint": s["split_fingerprint"],
        "table_fingerprint": go["table_fingerprint"],
        "plate_center": mode, "normalize_mean": o.normalize_mean, "normalize_std": o.normalize_std,
        "level3_cap": LEVEL3_CAP, "n_genes": int(expr.shape[1]), "pr_gene_id": go["pr_gene_id"],
        "summary": summary, "gate": gate,
        "plates": st["plates"], "n_train_dmso_per_plate": st["n_train_dmso_per_plate"],
        "centre": st["centre"].tolist(), "mean": st["mean"].tolist(), "std": st["std"].tolist(),
        "cap_frac": st["cap_frac"].tolist(),
    }
    dst = os.path.join(cfg.paths.nuisance_dir, expr_meta_filename(mode))
    _write_json(dst, out)

    g = gate
    print(f"[expr] {dst}  plate_center={mode}  train {summary['n_train']:,} rows "
          f"({summary['n_train_dmso']:,} DMSO; per plate {summary['train_dmso_per_plate']})")
    print(f"[expr] std median {summary['std']['median']:.3f} [{summary['std']['min']:.3f}, "
          f"{summary['std']['max']:.3f}]; {summary['cap']['n_genes_over_1pct']} genes > 1% at the "
          f"{LEVEL3_CAP} cap, top {summary['cap']['top'][0]}")
    print(f"[expr] centring gate on {g['n_holdout_dmso']} held-out DMSO wells / {g['n_plates']} plates: "
          f"plate-mean |z| median {g['plate_mean_abs_z']['median_centred']:.3f} "
          f"(uncentred {g['plate_mean_abs_z']['median_uncentred']:.2f}, expected 0.674); "
          f"plate share R2 {g['plate_share_r2']['before']:.3f} -> {g['plate_share_r2']['after']:.3f} "
          f"(chance {g['plate_share_r2']['chance']:.3f}; reported), omega2 {g['plate_share_omega2']['before']:.3f} -> "
          f"{g['plate_share_omega2']['after']:.3f}  => {'PASS' if g['pass'] else 'FAIL'} {g['checks']}")


if __name__ == "__main__":
    main()
