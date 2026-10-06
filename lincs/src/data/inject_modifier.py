"""The semi-synthetic effect modifier for policy learning (POLICY_LEARNING.md §14). No PyTorch.

Writes an INJECTED COPY of a build: the same table, splits, tier instances and
halves, with `expr.npy` changed so that, in z-space (plate-centred, z-scored),

    z_i <- z_i + beta * m[c, g] * s_c        for every treated well i of compound c in a
                                              line of group g whose dose half is the group's
                                              THIN half (G2: high, G1: low); vehicles untouched

with m[c, g] an independent sign per (compound, line group) and s_c the compound's
holdout signature on the REAL build (`learn.compound_signatures`), i.e. utility A's
axis, so the utility of an injected well moves by exactly +/- beta. The shift is
applied in raw Level 3 units, x += beta * m * (std (.) s_c), under the base build's
`expr_meta.json`, so the identity is exact wherever that file is used -- every copied
split dir carries it. beta = `--mult` x sigma_w, the per-well sd of utility A within
(compound, line, dose) cells on the real build.

Everything downstream (trainer, rollouts, policy stages, launcher) runs unchanged on
the copy with --data_dir <out>. A second beta is a NEW copy (another --out), never an
in-place rewrite: the launcher skips runs that already have checkpoints.

    python -m src.data.inject_modifier --data_dir data/core5_24h --out data/core5_24h_inj --mult 1.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import _atomic_write, _write_json  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import dose_half, load_splits  # noqa: E402
from src.policy.learn import THIN_HALF, compound_signatures  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args, line_group_sign, line_groups)

SIGNATURE_FILE = "injection_V.npy"
SIGNS_FILE = "injection_signs.npy"       # (n_compounds, 2): columns G1, G2
SWITCH_FILE = "injection_switch.npy"     # (n_rows,) int8 in {-1, 0, +1}
META_FILE = "injection.json"
COPIED_TOP = ("gene_order.json", "context_encoder.json")
SKIP_IN_SPLIT_DIRS = ("dr_weights_retarget_",)


def sha1_of(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def within_cell_sd(u: np.ndarray, key: np.ndarray) -> tuple[float, int]:
    """Pooled within-cell sd of u (cells with >= 2 rows), and the degrees of freedom."""
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    s = np.bincount(inv, weights=u)
    mean = s / n
    ss = np.bincount(inv, weights=(u - mean[inv]) ** 2)
    ok = n >= 2
    dof = int((n[ok] - 1).sum())
    return float(np.sqrt(ss[ok].sum() / dof)), dof


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="The injected build dir to write (a sibling of --data_dir).")
    p.add_argument("--mult", type=float, default=1.5, help="beta = mult x sigma_w (the ladder: 1.5, 2, 3).")
    p.add_argument("--sign_seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    t0 = time.time()
    cfg = apply_paths_args(config_from_args(args), args)
    src = os.path.normpath(cfg.paths.data_dir)
    out = os.path.normpath(os.path.abspath(args.out))
    if os.path.normpath(out) == src:
        raise SystemExit("[inject] --out must differ from --data_dir")
    runs_dir = PROJECT_ROOT / "runs" / os.path.basename(out)       # where apply_paths_args puts this build's runs
    if os.path.isdir(out):
        if not args.overwrite:
            raise SystemExit(f"[inject] {out} exists; a second beta goes to a NEW dir (the launcher skips existing runs), "
                             f"or pass --overwrite")
        trained = sorted(d.name for d in runs_dir.iterdir() if d.is_dir() and any(d.glob("checkpoint-*"))) \
            if runs_dir.is_dir() else []
        if trained:
            raise SystemExit(f"[inject] {runs_dir} holds trained runs ({trained[:3]}...); they would be paired with "
                             f"another injection. Use a new --out")
        shutil.rmtree(out)
    if line_groups(cfg.population) is None:
        raise SystemExit("[inject] the population declares no line groups")
    with open(cfg.paths.population_qc_json) as fh:
        qc = json.load(fh)
    if qc.get("injection"):
        raise SystemExit(f"[inject] {src} is itself an injected build")

    # ---- the real build ----------------------------------------------------------------
    from datasets import load_from_disk
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "cell_id", "det_plate"]).to_pandas())
    N = len(meta)
    comp = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool)
    lines = sorted(set(meta["cell_id"].values))
    line_t = np.array([lines.index(c) for c in meta["cell_id"].values], dtype=np.int64)
    g2_line = line_group_sign(np.array(lines), cfg.population) > 0
    base = load_splits(cfg)
    em = load_expr_meta(cfg, splits=base)
    if int(em.get("n_std_floored", 0)) != 0:
        raise SystemExit("[inject] expr_meta floors a gene's std; the raw-unit shift would not be exact there")
    x = np.load(cfg.paths.expr_npy)
    if x.shape != (N, em["std"].size):
        raise SystemExit(f"[inject] expr.npy {x.shape} does not match the table")
    pc = plate_codes(meta["det_plate"].values, em["plates"])
    z = normalize_expr(x, pc, em)
    hold = np.zeros(N, dtype=bool); hold[base["holdout_idx"]] = True
    train = np.zeros(N, dtype=bool); train[base["train_idx"]] = True
    n_comp = int(comp.max()) + 1
    V = compound_signatures(z, comp, ctl, line_t, dl, hold, n_comp)
    print(f"[inject] {src}: {N:,} rows, {n_comp - 1:,} compounds, signatures from the holdout wells ({time.time() - t0:.0f}s)", flush=True)

    # ---- sigma_w: the per-well sd of utility A within (compound, line, dose) cells -----
    trt = train & ~ctl
    u = np.einsum("ng,ng->n", z[trt].astype(np.float64), V[comp[trt]])
    key = np.array([f"{c}|{l}|{d:.6g}" for c, l, d in zip(comp[trt], line_t[trt], dl[trt])])
    sigma_w, dof = within_cell_sd(u, key)
    beta = float(args.mult * sigma_w)
    if dof <= 0 or not np.isfinite(sigma_w) or not np.isfinite(beta) or beta <= 0:
        raise SystemExit(f"[inject] sigma_w {sigma_w} over {dof} dof: no cell with >= 2 train wells, or not finite")
    print(f"[inject] sigma_w = {sigma_w:.3f} (within-cell sd of utility A over {int(trt.sum()):,} unthinned treated "
          f"train wells, {dof:,} dof); beta = {args.mult:g} x sigma_w = {beta:.3f}", flush=True)

    # ---- the switch: m[c, g] on the thin half of each group ------------------------------
    rng = np.random.default_rng([int(args.sign_seed), 31_337])
    signs = rng.choice([-1, 1], size=(n_comp, 2)).astype(np.int8)
    signs[0] = 0                                                   # the vehicle
    half = dose_half(comp, dl, ctl)
    gidx = np.where(g2_line[line_t], 1, 0)
    thin = np.where(g2_line[line_t], THIN_HALF["G2"], THIN_HALF["G1"])
    switch = np.where(~ctl & (half == thin), signs[comp, gidx], 0).astype(np.int8)
    rows = np.flatnonzero(switch != 0)
    print(f"[inject] {rows.size:,} of {int((~ctl).sum()):,} treated rows injected "
          f"(G1 low half {int(((switch != 0) & ~g2_line[line_t]).sum()):,}, G2 high half "
          f"{int(((switch != 0) & g2_line[line_t]).sum()):,}); signs +1 share "
          f"{float((signs[1:] > 0).mean()):.3f}", flush=True)

    # ---- the injected expression, exact in z-space under the base z-scale ---------------
    x_inj = x.astype(np.float32).copy()
    delta = (beta * switch[rows].astype(np.float64))[:, None] * (em["std"][None, :] * V[comp[rows]])
    x_inj[rows] = (x[rows].astype(np.float64) + delta).astype(np.float32)
    z_chk = normalize_expr(x_inj[rows[:2000]], pc[rows[:2000]], em).astype(np.float64)
    want = z[rows[:2000]].astype(np.float64) + beta * switch[rows[:2000]][:, None] * V[comp[rows[:2000]]]
    err = float(np.abs(z_chk - want).max())
    if not err <= 1e-3:
        raise SystemExit(f"[inject] the z-space shift is not exact (max |err| {err:.2e})")
    del z

    # ---- write the copy ---------------------------------------------------------------------
    os.makedirs(out, exist_ok=True)
    shutil.copytree(cfg.paths.tabular_dataset_dir, os.path.join(out, os.path.basename(cfg.paths.tabular_dataset_dir)))
    for fn in COPIED_TOP:
        shutil.copy2(os.path.join(src, fn), os.path.join(out, fn))
    copied_dirs = []
    for d in sorted(os.listdir(src)):
        sd = os.path.join(src, d)
        if not os.path.isdir(sd) or not (d == "nuisances" or (d.startswith("nuisances_tier_") and "_pk-half" in d)):
            continue
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for fn in sorted(os.listdir(sd)):
            if fn.startswith(SKIP_IN_SPLIT_DIRS) or fn.endswith(".tmp") or fn == "__pycache__":
                continue
            if os.path.isfile(os.path.join(sd, fn)):
                shutil.copy2(os.path.join(sd, fn), os.path.join(out, d, fn))
        copied_dirs.append(d)
    _atomic_write(os.path.join(out, "expr.npy"), lambda f: np.save(f, x_inj), mode="wb")
    _atomic_write(os.path.join(out, SIGNATURE_FILE), lambda f: np.save(f, V.astype(np.float32)), mode="wb")
    _atomic_write(os.path.join(out, SIGNS_FILE), lambda f: np.save(f, signs), mode="wb")
    _atomic_write(os.path.join(out, SWITCH_FILE), lambda f: np.save(f, switch), mode="wb")
    block = {"kind": "line_group x dose_half effect modifier (POLICY_LEARNING.md §14)",
             "rule": "z += beta * m[compound, line group] * s_compound on treated wells whose dose half is the group's "
                     "thin half; vehicles untouched; every compound",
             "thin_half": THIN_HALF, "line_groups": {k: list(v) for k, v in line_groups(cfg.population).items()},
             "beta": beta, "mult": float(args.mult), "sigma_w": sigma_w, "sigma_w_dof": dof,
             "sigma_w_definition": "pooled within-(compound, line, dose) sd of <z, s_c> over unthinned treated train wells",
             "sign_seed": int(args.sign_seed), "signature_file": SIGNATURE_FILE, "signs_file": SIGNS_FILE,
             "switch_file": SWITCH_FILE, "signature_sha1": sha1_of(V.astype(np.float32)), "signs_sha1": sha1_of(signs),
             "switch_sha1": sha1_of(switch), "n_injected_rows": int(rows.size),
             "source_data_dir": src, "source_table_fingerprint": qc.get("table_fingerprint"),
             "base_split_fingerprint": base["split_fingerprint"], "z_shift_max_err": err,
             "copied_split_dirs": copied_dirs, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    qc["injection"] = block
    _write_json(os.path.join(out, "population_qc.json"), qc)
    _write_json(os.path.join(out, META_FILE), block)
    print(f"[inject] -> {out}: expr.npy, {len(copied_dirs)} split dirs, {META_FILE} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
