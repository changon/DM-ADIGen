"""P1: post-hoc doubly-robust targeting of tau_hat (`understand.md` §3.3.1). No PyTorch.

Step C2 (IMPLEMENT.md §5) showed that ADIGen's alpha-WEIGHTED TRAINING RISK
over-corrects: the weights balance the confounder inside a positivity cell of
1-3 rows, but the generator resolves ARMS inside that cell, so at arm level the
realised weighted mix is a ratio over 1-2 wells. P1 stops weighting the risk.
It keeps the UNWEIGHTED generator as the outcome model and spends alpha only
where AIPW spends it -- on the estimand:

    delta_g       = sum_{i in g} w_i (Y_i - mu_hat(X_i, A_i)) / sum_{i in g} w_i
    tau_hat_P1(a) = tau_hat_gen(a) + delta_{g(a)}

over the KEPT TRAIN ROWS of the group g, Hajek-normalised. The vehicle side is
uncorrected: mu_hat(0) comes from real DMSO wells, which are never thinned.

**The group is derived, never hardcoded** (`splits.target_groups`): it is the
positivity cell with the confounder dropped, so

    syn_c   -> (compound, dose half)   3,497 groups on mcf7_24h  (steps C / C2)
    cell_id -> the ARM itself         10,479 groups              (step A)

Positivity holds in every cell by construction, so every level of the
confounder is represented inside a group and the weights can rebalance it.
With `counts` weights sum_{i in g} w_i = n_nu(g) exactly (measured: 3e-08
relative), so Horvitz-Thompson and Hajek coincide; with `design` weights they
differ by 7-9%, which is why the Hajek form above is the one implemented.

This is POST HOC and CPU only. It reads the generator's per-row means out of a
finished scoring run (`*_gen.npz`), never samples a model, and writes a new
eval-shaped JSON + `_tau.npz` beside the source so `step_c_report` can score it
as its own arm.

    python -m src.eval.dr_target --run_dir runs/mcf7_24h/stepc2_conditional_g1_s0 \
        --syn_meta syn_meta_compound_r1.json --weights counts --weights design
    python -m src.eval.dr_target --run_dir ... --weights ones --npz none   # control
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import _atomic_write  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import (  # noqa: E402
    CONTROL_ARM, POSITIVITY_KEYS, arm_keys, dose_half, load_splits, positivity_cells,
    target_groups)
from src.data.synthetic import directions_for, inject_meta, load_syn_meta  # noqa: E402
from src.eval import dist_metrics as dm  # noqa: E402
from src.eval.evaluate import (  # noqa: E402
    META_COLUMNS, TRUTH_GUARD, _accuracy, _cos, _cos_rows, _jsonable, _load_truth,
    _pearson_rows, _q, _split_key, bias_along_contrast, bias_along_v, corrected_ceiling)
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, add_syn_cli, apply_paths_args, config_from_args)

METHOD, VERSION = "P1", 1
WEIGHT_MODES = ("counts", "design", "ones")
# `ones` is the alpha-SENSITIVITY control, not a null one: with w = 1 the Hajek
# mean runs over the THINNED mix, so delta_g ~ beta * p_kept(g) * v and the
# correction ADDS the thinning bias. Same degrees of freedom as `counts`, wrong
# alpha, far worse -- which is the evidence that the weights are load-bearing
# and that P1 is not merely fitting the statistic with one free parameter per
# group. It writes no weights file, so it can never be a DR arm.
CONTROL_MODES = ("ones",)
# Short names for the targeted arm labels; keep in step with step_c_report.
SRC_SHORT = {"naive": "naive", "conditional": "cond"}


def _sha1_w(w: np.ndarray) -> str:
    """The trainer's convention (train_diffusion.py): sha1 of the raw float32 bytes."""
    return hashlib.sha1(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()


def _src_arm_label(arch: dict) -> str | None:
    """`naive` / `conditional` from the source run's arch.json (step_c_report's rule)."""
    if str(arch.get("dr_mode")) != "conditional":
        return None
    c = tuple(arch.get("adjustment_set") or ())
    return "naive" if not c else ("conditional" if len(c) == 1 else None)


def hajek_delta(resid: np.ndarray, w: np.ndarray, gi: np.ndarray, n_groups: int,
                gene_block: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(delta (n_groups, G), sum_w, sum_w2, n_rows) by group index `gi`.

    delta_g = sum_i w_i r_i / sum_i w_i, the self-normalised (Hajek) form. It is
    identical to Horvitz-Thompson under `counts` weights, where sum_i w_i is
    exactly n_nu(g), and is the correct general form when it is not.

    Accumulated in float64 over blocks of genes: `resid * w[:, None]` across all
    978 genes at once is a 156 MB float64 temporary, which is the single largest
    allocation in the module and is what makes a 20k-row train pool fit in well
    under a gigabyte. Rows are sorted by group once so each block is a
    `reduceat`, not a scattered `np.add.at`.
    """
    G = resid.shape[1]
    order = np.argsort(gi, kind="stable")
    gs, ws = gi[order], w[order]
    counts = np.bincount(gs, minlength=n_groups)
    present = np.flatnonzero(counts > 0)
    starts = np.concatenate([[0], np.cumsum(counts[present])[:-1]])
    num = np.zeros((n_groups, G), dtype=np.float64)
    for lo in range(0, G, gene_block):
        hi = min(lo + gene_block, G)
        blk = resid[order, lo:hi].astype(np.float64)
        blk *= ws[:, None]
        num[present, lo:hi] = np.add.reduceat(blk, starts, axis=0)
        del blk
    sw = np.bincount(gi, weights=w, minlength=n_groups)
    sw2 = np.bincount(gi, weights=w * w, minlength=n_groups)
    n = counts.astype(np.int64)
    delta = np.zeros_like(num)
    ok = sw > 0
    delta[ok] = num[ok] / sw[ok, None]
    return delta, sw, sw2, n


def clustered_contrast_se(proj: np.ndarray, halves: np.ndarray, comp: np.ndarray,
                          sel: np.ndarray, seed: int = 0, n_boot: int = 400) -> float | None:
    """SE of the high-minus-low mean contrast, resampling COMPOUNDS, not arms.

    The thinning draws independently per compound, and P1 adds one shared
    delta_g per (compound, half), so arms inside a group are perfectly
    correlated and `bias_along_v`'s arm-level SE understates the spread. The
    compound is the independent unit, so it is the one to resample.
    """
    m = sel & (halves >= 0)
    if not m.any():
        return None
    cs = np.unique(comp[m])
    if cs.size < 2:
        return None
    by = {int(c): (proj[m & (comp == c) & (halves == 1)],
                   proj[m & (comp == c) & (halves == 0)]) for c in cs}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(cs, cs.size, replace=True)
        hi = np.concatenate([by[int(c)][0] for c in pick]) if cs.size else np.empty(0)
        lo = np.concatenate([by[int(c)][1] for c in pick]) if cs.size else np.empty(0)
        vals[b] = (hi.mean() - lo.mean()) if hi.size and lo.size else np.nan
    v = vals[np.isfinite(vals)]
    return float(v.std(ddof=1)) if v.size > 1 else None


def accuracy_block(tau_est: np.ndarray, truth: dict, keys: np.ndarray, src_acc: dict,
                   syn: dict | None, half_of_arm: dict, scored_ci: set[int],
                   n_wells: np.ndarray) -> dict:
    """The gen-dependent half of `evaluate._pool_block`'s accuracy block, recomputed.

    Everything truth-side (`n_responder`, `split_half_responder`,
    `reliability_ceiling_responder`, the per-dose arm counts) is copied from
    `src_acc`: it does not depend on the estimate, and recomputing it would
    duplicate `_pool_block`'s real-side logic for no gain.
    """
    shared, i_est, i_tr = np.intersect1d(keys, truth["arm_key"], return_indices=True)
    if shared.size == 0:
        raise RuntimeError("no arm is present in both this run and the oracle")
    te, tt = tau_est[i_est], truth["tau"][i_tr]
    cos = _cos_rows(te, tt)
    pear = _pearson_rows(te, tt)
    resp = truth["responder"][i_tr].astype(bool)
    ceil_c = corrected_ceiling(truth["reliability_cos"][i_tr])
    fin = resp & np.isfinite(ceil_c)
    comp, dose = _split_key(shared)
    acc = dict(src_acc)
    acc.update({
        "n_matched": int(shared.size),
        "n_unmatched_est": int(keys.size - shared.size),
        "n_unmatched_truth": int(truth["arm_key"].size - shared.size),
        "cos_all": _q(cos), "pearson_all": _q(pear),
        "cos_responder": _q(cos[resp]), "pearson_responder": _q(pear[resp]),
        "cos_responder_with_ceiling": _q(cos[fin]),
        "cos_over_ceiling_responder": _q(
            cos[fin] / np.where(ceil_c[fin] > 0, ceil_c[fin], np.nan)),
        "tau_norm": _accuracy(np.linalg.norm(te, axis=1), np.linalg.norm(tt, axis=1)),
        "tau_norm_responder": (_accuracy(np.linalg.norm(te[resp], axis=1),
                                         np.linalg.norm(tt[resp], axis=1))
                               if resp.any() else None),
        "pooled_gene": _accuracy(te.reshape(-1), tt.reshape(-1)),
    })
    by_dose = {}
    for d in np.unique(dose):
        s = dose == d
        by_dose[str(d)] = {"n_arms": int(s.sum()),
                           "cos_mean_tau": _cos(te[s].mean(0), tt[s].mean(0)),
                           "cos_median_arm": float(np.median(cos[s]))}
    acc["by_dose_level"] = by_dose
    acc["per_compound_cos_pooled"] = _q(np.array(
        [_cos(te[comp == c].mean(0), tt[comp == c].mean(0)) for c in np.unique(comp)]))
    if syn is not None and half_of_arm is not None:
        acc["bias_along_v"] = bias_along_v(
            te, tt, shared, directions_for(syn, comp), half_of_arm,
            scored_ci or set(), n_wells=n_wells[i_est])
    # Step A (STEP_A.md §3): the readout along each arm's line-group contrast,
    # recomputed on the corrected tau exactly as `_pool_block` computes it.
    cproj = None
    if truth.get("contrast_dir") is not None and half_of_arm is not None:
        acc["bias_along_contrast"], cp = bias_along_contrast(
            te, tt, shared, truth["contrast_dir"][i_tr], half_of_arm,
            scored_ci or set(), n_wells=n_wells[i_est])
        cproj = np.full(keys.size, np.nan)
        cproj[i_est] = cp
    return acc, cos, pear, shared, comp, i_est, cproj


def _assert_same_keys(new: dict, old: dict, where: str) -> None:
    """Recursive key-set equality: a renamed `_pool_block` key must not pass silently."""
    if set(new) != set(old):
        raise RuntimeError(f"{where}: key set drifted from the source document "
                           f"(+{sorted(set(new) - set(old))} -{sorted(set(old) - set(new))})")
    for k, v in old.items():
        if isinstance(v, dict) and isinstance(new.get(k), dict):
            _assert_same_keys(new[k], v, f"{where}.{k}")


def _frame_guard(doc: dict, this: dict, where: str) -> None:
    """Every TRUTH_GUARD key of the source doc must match the frame rebuilt here."""
    bad = []
    for k in TRUTH_GUARD:
        if k not in doc:
            continue
        a, b = doc[k], this[k]
        same = (abs(float(a) - float(b)) < 1e-12 if isinstance(b, float)
                else str(a) == str(b))
        if not same:
            bad.append(f"{k}: source {a!r} != rebuilt {b!r}")
    if bad:
        raise RuntimeError(f"{where}: the frame this module rebuilt is not the one the "
                           f"source was scored under:\n  - " + "\n  - ".join(bad))


def _source_json(run_dir: str) -> str:
    """The one eval JSON of a finished scoring run (refuse 0 or >= 2)."""
    hits = [f for f in sorted(glob.glob(os.path.join(run_dir, "eval_artifacts", "gen_*.json")))
            if "_p1" not in os.path.basename(f)]
    if len(hits) != 1:
        raise SystemExit(f"[p1] {run_dir}: expected exactly one source eval JSON in "
                         f"eval_artifacts/, found {len(hits)}"
                         + (f": {[os.path.basename(h) for h in hits]}" if hits else ""))
    return hits[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", action="append", required=True,
                   help="A finished scoring run to target (repeatable).")
    p.add_argument("--weights", action="append", choices=WEIGHT_MODES, default=None,
                   help=f"Weight set(s) to target with (repeatable). Default: counts. "
                        f"{CONTROL_MODES[0]!r} is the alpha-sensitivity control.")
    p.add_argument("--truth", default=None,
                   help="Oracle JSON. Default: the source run's own recorded truth.path.")
    p.add_argument("--group_key", default="auto",
                   help="'auto' = the positivity cell minus the confounder (§3.3.1). "
                        "Override with a confounder name to coarsen to ITS group.")
    p.add_argument("--npz", default="full", choices=("full", "delta", "none"))
    p.add_argument("--baseline_tol", type=float, default=1e-6,
                   help="Max |diff| allowed when reproducing the source metrics before "
                        "the correction is applied.")
    p.add_argument("--min_group_rows", type=int, default=1)
    p.add_argument("--allow_uncorrected", action="store_true",
                   help="Write delta = 0 for groups with no kept train row instead of "
                        "refusing. A silently uncorrected arm biases the contrast.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    add_syn_cli(p)
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    modes = list(dict.fromkeys(args.weights or ["counts"]))
    t_start = time.time()

    # A wrong dgemm silently corrupts every metric (IMPLEMENT.md §5), so refuse first.
    print(f"[blas] numpy matrix products verified (rel err {dm.assert_blas_ok():.1e}, "
          f"OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE')})", flush=True)

    cfg0 = apply_paths_args(config_from_args(args), args, require_splits=False)
    meta = (load_from_disk(cfg0.paths.tabular_dataset_dir)
            .select_columns(list(META_COLUMNS)).to_pandas())
    n_total = len(meta)
    compound_idx = meta["compound_idx"].values.astype(np.int64)
    dose_level = meta["dose_level"].values.astype(np.float64)
    is_control = meta["is_control"].values.astype(np.int64)
    keys_all = arm_keys(compound_idx, dose_level, is_control)
    half_of_arm = dict(zip(keys_all, dose_half(compound_idx, dose_level, is_control)))
    expr_raw = np.load(cfg0.paths.expr_npy, mmap_mode="r")
    y_cache: dict[tuple, tuple[np.ndarray, dict | None, dict]] = {}
    written: list[str] = []

    for run_dir in args.run_dir:
        run_dir = os.path.abspath(run_dir)
        src_path = _source_json(run_dir)
        with open(src_path) as fh:
            doc = json.load(fh)
        if doc.get("targeting"):
            raise SystemExit(f"[p1] {src_path} is already a targeted document")
        if doc.get("source") != "generated":
            raise SystemExit(f"[p1] {src_path}: source={doc.get('source')!r}, expected 'generated'")
        with open(os.path.join(run_dir, "arch.json")) as fh:
            arch = json.load(fh)
        if str(arch.get("dr_mode")) == "weighted":
            raise SystemExit(
                f"[p1] {run_dir}: its training risk is ALREADY alpha-weighted "
                f"(dr_mode=weighted, {arch.get('dr_weights_file')}). Targeting its residuals "
                f"would apply alpha twice. Target the `conditional` or `naive` arm instead.")
        src_arm = _src_arm_label(arch)
        if src_arm not in SRC_SHORT:
            raise SystemExit(f"[p1] {run_dir}: not a targetable source arm "
                             f"(dr_mode={arch.get('dr_mode')!r}, C={arch.get('adjustment_set')})")
        tier = arch.get("tier") or {}
        if not tier.get("active"):
            raise SystemExit(f"[p1] {run_dir}: arch.tier.active is false; there is no "
                             f"thinning to correct")
        conf = str(tier["confounder"])
        if list(tier.get("positivity_key") or ()) != list(POSITIVITY_KEYS[conf]):
            raise SystemExit(f"[p1] {run_dir}: tier.positivity_key "
                             f"{tier.get('positivity_key')} != {list(POSITIVITY_KEYS[conf])}")
        nz = arch["nuisance_dir"]

        # ---- the frame this run was scored under ----
        class _A:   # a carrier apply_paths_args understands
            pass
        a2 = _A()
        for k, v in vars(args).items():
            setattr(a2, k, v)
        a2.nuisance_dir = nz
        a2.syn_effect = float(doc.get("syn_effect") or 0.0)
        a2.syn_seed = int(doc.get("syn_seed") or 0)
        a2.syn_meta = (doc.get("syn") or {}).get("name") or args.syn_meta
        # TWO splits are legitimately in play for a tiered arm, and conflating
        # them is the easy mistake: it TRAINS on the thinned split (the tier
        # dir, whose train_idx the weights are defined on) but is SCORED on the
        # unablated base split (§3.8.1), which is the frame the source document
        # and the oracle carry. So the eval frame and y_all come from the base
        # dir; train_idx, the weights and nu_rows come from the tier dir.
        cfg_tier = apply_paths_args(config_from_args(a2), a2)
        splits_tier = load_splits(cfg_tier)
        a2.nuisance_dir = None
        cfg = apply_paths_args(config_from_args(a2), a2)
        splits = load_splits(cfg)
        m = load_expr_meta(cfg, doc["plate_center"], splits=splits)
        if str(arch.get("split_fingerprint")) != str(splits_tier["split_fingerprint"]):
            raise SystemExit(f"[p1] {run_dir}: arch split_fingerprint "
                             f"{arch.get('split_fingerprint')} != its tier dir's "
                             f"{splits_tier['split_fingerprint']}")
        # The tier dir carries its own expr_meta; expr_stats fits it on nu_rows, so
        # the z-scale must be identical to the base build's or Y_i and mu_hat would
        # live in different spaces (scripts/phase5_cpu.sub asserts this at build time).
        m_tier = load_expr_meta(cfg_tier, doc["plate_center"], splits=splits_tier)
        for k in ("centre", "mean", "std"):
            if not np.array_equal(np.asarray(m[k], dtype=np.float64),
                                  np.asarray(m_tier[k], dtype=np.float64)):
                raise SystemExit(f"[p1] {run_dir}: the tier dir's expr_meta {k!r} differs "
                                 f"from the base build's; the z-scale is not shared")

        key = (str(m.get("split_fingerprint")), doc["plate_center"], str(a2.syn_meta),
               float(a2.syn_effect))
        if key not in y_cache:
            y = normalize_expr(np.asarray(expr_raw), plate_codes(meta["det_plate"].values,
                                                                 m["plates"]), m)
            s = None
            if cfg.outcome.syn_effect != 0:
                s = load_syn_meta(cfg, n_genes=cfg.outcome.n_genes)
                y = inject_meta(y, meta["syn_c"].values.astype(np.int64), compound_idx, s)
            y_cache[key] = (y, s, m)
            print(f"[p1] y_all built for {a2.syn_meta} "
                  f"({'injected beta %.4f' % s['beta'] if s else 'no injection'})", flush=True)
        y_all, syn, m = y_cache[key]
        this = {"population": cfg.population.name,
                "table_fingerprint": m.get("table_fingerprint"),
                "split_fingerprint": splits.get("split_fingerprint"),
                "gene_order_sha1": hashlib.sha1(
                    json.dumps([int(g) for g in m["pr_gene_id"]]).encode()).hexdigest(),
                "plate_center": doc["plate_center"],
                "normalize_mean": m.get("normalize_mean"),
                "normalize_std": m.get("normalize_std"),
                "syn_effect": float(cfg.outcome.syn_effect),
                "syn_seed": int(cfg.outcome.syn_seed),
                "syn_beta": (float(syn["beta"]) if syn else 0.0),
                "syn_v_sha1": (str(syn["v_sha1"]) if syn else None),
                "min_dose_n": int(doc["min_dose_n"]),
                "adjustment_set": list(cfg.adjustment_set)}
        _frame_guard(doc, this, src_path)

        # ---- the generator's per-row means ----
        gen_path = (doc.get("artifacts") or {}).get("gen_npz")
        if not gen_path or not os.path.isfile(gen_path):
            gen_path = src_path[:-5] + "_gen.npz"
        if not os.path.isfile(gen_path):
            raise SystemExit(f"[p1] {run_dir}: no _gen.npz; P1 needs the generator's per-row "
                             f"means. Re-score without --no_save_gen.")
        gz = np.load(gen_path)
        row_id = gz["row_id"]
        row_mean = gz["row_mean"]
        train_idx = np.asarray(splits_tier["train_idx"], dtype=np.int64)
        pos = np.full(n_total, -1, dtype=np.int64)
        pos[row_id] = np.arange(row_id.size)
        if (pos[train_idx] < 0).any():
            raise SystemExit(f"[p1] {run_dir}: it was scored on a pool that excludes train "
                             f"rows (pool={doc.get('pool')!r}), so mu_hat is unavailable "
                             f"there and P1 cannot be computed.")
        if int(gz["n_per_row"]) != int((doc["generator"] or {}).get("n_per_row", -1)):
            raise SystemExit(f"[p1] {run_dir}: _gen.npz n_per_row != the JSON's")

        # ---- groups ----
        gconf = conf if args.group_key == "auto" else args.group_key
        groups = target_groups(gconf, compound_idx, dose_level, is_control)
        cells = positivity_cells(conf, compound_idx, dose_level, is_control,
                                 meta[conf].values)
        trt = is_control.astype(bool) == False  # noqa: E712
        if len(set(zip(cells[trt].tolist(), groups[trt].tolist()))) != len(set(cells[trt].tolist())):
            raise SystemExit(f"[p1] positivity cells for {conf!r} do not refine the group key "
                             f"{gconf!r}; the counts identity would not hold")
        gnames = np.array(sorted({g for g in groups[trt]}), dtype=object)
        gindex = {g: i for i, g in enumerate(gnames)}
        nu = np.load(os.path.join(nz, "nu_rows.npy"))
        nu_t = nu[trt[nu]]
        n_nu = np.bincount(np.array([gindex[g] for g in groups[nu_t]], dtype=np.int64),
                           minlength=gnames.size)

        tr_t = train_idx[trt[train_idx]]
        gi = np.array([gindex[g] for g in groups[tr_t]], dtype=np.int64)
        # float32: both operands are, and hajek_delta accumulates in float64
        # per gene block. In float64 this array alone is 156 MB.
        resid = y_all[tr_t] - row_mean[pos[tr_t]]

        # ---- per-run, mode-independent: the oracle, the pristine arm tables,
        # the arm -> group map, and the BASELINE reproduction of the source
        # metrics. None of these depend on the weights, so computing them once
        # per run instead of once per mode is most of the runtime.
        tz0 = dict(np.load(src_path[:-5] + "_tau.npz", allow_pickle=False))
        truth_path = args.truth or (doc.get("truth") or {})["path"]
        scored_ci = {int(v) for v in (tier.get("scored_compounds") or {}).values()}
        # arm -> group, resolved on the WHOLE TABLE. `dose_half` ranks a
        # compound's DISTINCT dose levels, so deriving it from a pool's arm
        # table would shift the halves of any compound whose arms the pool
        # dropped (`--min_dose_n`, or the holdout's one-well arms): measured, 12
        # arms in `all` and 150 in `holdout` would land in the other half and
        # receive the wrong group's delta, while `bias_along_v` still classifies
        # them by the table's halves. evaluate.py:902 builds `half_of_arm` the
        # same way and for the same reason.
        group_of_arm = dict(zip(keys_all, groups))
        per_pool: dict[str, dict] = {}
        for pname, blk in doc["pools"].items():
            keys = tz0[f"{pname}/arm_key"]
            gof = np.array([gindex.get(group_of_arm.get(k), -1) for k in keys],
                           dtype=np.int64)
            if (gof < 0).any():
                missing = [k for k, i in zip(keys, gof) if i < 0][:3]
                raise SystemExit(f"[p1] {pname}: {int((gof < 0).sum())} arms map to no "
                                 f"group, e.g. {missing}")
            truth = _load_truth(truth_path, this, pname, int(blk["min_dose_n"]))
            tau_src = tz0[f"{pname}/tau_gen"].astype(np.float64)
            n_wells = tz0[f"{pname}/n_wells"]
            # The decisive guard: reproduce the source's own metrics on the
            # UNCORRECTED tau before applying a single correction. It proves
            # y_all, the oracle, the arm intersection and the metric path all
            # reproduce the original scoring.
            base_acc, *_ = accuracy_block(tau_src, truth, keys, blk["accuracy"], syn,
                                          half_of_arm, scored_ci, n_wells)
            checks = {}
            for path_ in (("cos_all", "median"), ("cos_responder", "median"),
                          ("pooled_gene", "mse"),
                          ("bias_along_v", "scored_high_minus_low_mean"),
                          ("bias_along_contrast", "scored_high_minus_low_mean")):
                aa = (base_acc.get(path_[0]) or {}).get(path_[1])
                bb = (blk["accuracy"].get(path_[0]) or {}).get(path_[1])
                if aa is not None and bb is not None:
                    checks[".".join(path_)] = abs(float(aa) - float(bb))
            worst = max(checks.values()) if checks else 0.0
            if worst > args.baseline_tol:
                raise RuntimeError(
                    f"[p1] {run_dir} {pname}: could not reproduce the source metrics "
                    f"before correcting (max |diff| {worst:.3g} > {args.baseline_tol}): "
                    f"{checks}")
            per_pool[pname] = {"keys": keys, "gof": gof, "truth": truth,
                               "tau_src": tau_src, "n_wells": n_wells,
                               "checks": checks, "worst": worst}
            del base_acc
        print(f"[p1] {os.path.basename(run_dir)}: baseline reproduced, max |diff| "
              f"{max(v['worst'] for v in per_pool.values()):.2e}", flush=True)

        for mode in modes:
            out_json = src_path[:-5] + f"_p1{mode}.json"
            if os.path.isfile(out_json) and not args.overwrite:
                with open(out_json) as fh:
                    old = (json.load(fh).get("targeting") or {}).get("weights") or {}
                print(f"[p1] {os.path.basename(out_json)} exists (weights sha1 "
                      f"{str(old.get('sha1'))[:12]}); pass --overwrite to replace", flush=True)
                continue
            # ---- the weights ----
            if mode in CONTROL_MODES:
                w_all = np.ones(train_idx.size, dtype=np.float32)
                wpath, wsha, wmode_rec, wfp = None, None, mode, None
            else:
                wpath = os.path.join(nz, f"dr_weights_{mode}.npz")
                wz = np.load(wpath)
                if not np.array_equal(wz["row_id"], train_idx):
                    raise SystemExit(f"[p1] {wpath}: row_id != this split's train_idx")
                if "split_fingerprint" in wz.files and \
                        str(wz["split_fingerprint"]) != str(splits_tier["split_fingerprint"]):
                    raise SystemExit(f"[p1] {wpath}: built for split "
                                     f"{str(wz['split_fingerprint'])}, not this arm's "
                                     f"{splits_tier['split_fingerprint']}")
                w_all = wz["w"].astype(np.float32)
                if not np.all(np.isfinite(w_all)) or (w_all <= 0).any():
                    raise SystemExit(f"[p1] {wpath}: weights must be finite and > 0")
                wsha = _sha1_w(w_all)
                # The `design` npz carries no `mode` key; detect by presence.
                wmode_rec = str(wz["mode"]) if "mode" in wz.files else mode
                wfp = str(wz["split_fingerprint"]) if "split_fingerprint" in wz.files else None
            w = w_all[trt[train_idx]].astype(np.float64)

            delta, sw, sw2, nrows = hajek_delta(resid, w, gi, gnames.size)
            # counts weights restore the unthinned group size EXACTLY; design only
            # in expectation. Assert the first, measure the second.
            have = (n_nu > 0) & (sw > 0)
            rel = np.abs(sw[have] - n_nu[have]) / np.maximum(n_nu[have], 1)
            max_rel = float(rel.max()) if rel.size else 0.0
            if wmode_rec == "counts" and max_rel > 1e-6:
                raise SystemExit(f"[p1] counts identity violated: max |sum_w - n_nu|/n_nu "
                                 f"= {max_rel:.3g} (> 1e-6)")
            uncorrected = int(((n_nu > 0) & (sw <= 0)).sum())
            if uncorrected and not args.allow_uncorrected:
                raise SystemExit(f"[p1] {uncorrected} groups have eval rows but no kept train "
                                 f"row; pass --allow_uncorrected to write delta = 0 for them")
            n_eff = np.where(sw2 > 0, sw ** 2 / np.maximum(sw2, 1e-12), 0.0)

            # ---- apply, per sub-pool ----
            tz = dict(tz0)
            new = copy.deepcopy(doc)
            tgt_delta_rec, pool_recs = {}, {}
            for pname, blk in doc["pools"].items():
                pp = per_pool[pname]
                keys, gof, truth = pp["keys"], pp["gof"], pp["truth"]
                n_wells, checks, worst = pp["n_wells"], pp["checks"], pp["worst"]
                tau_new = pp["tau_src"] + delta[gof]
                acc, cos, pear, shared, comp_sh, i_est, cproj = accuracy_block(
                    tau_new, truth, keys, blk["accuracy"], syn, half_of_arm,
                    scored_ci, n_wells)
                nb = copy.deepcopy(blk)
                nb["accuracy"] = acc
                tau_norm_new = np.linalg.norm(tau_new, axis=1)
                nb["tau_norm_est"] = _q(tau_norm_new)
                dn = tz[f"{pname}/tau_norm_denoised"]
                nb["amplitude_ratio_est_over_denoised"] = _q(
                    tau_norm_new / np.where(dn > 0, dn, np.nan))
                arm_cos = np.full(keys.size, np.nan); arm_cos[i_est] = cos
                arm_pear = np.full(keys.size, np.nan); arm_pear[i_est] = pear
                tau_real = tz[f"{pname}/tau"].astype(np.float64)
                for row in nb.get("effects") or []:
                    sel = np.flatnonzero(tz[f"{pname}/compound_idx"] == row["compound_idx"])
                    if sel.size == 0 or "tau_norm_est_max" not in row:
                        continue
                    best = sel[np.argmax(tz[f"{pname}/tau_norm"][sel])]
                    row["tau_norm_est_max"] = float(tau_norm_new[best])
                    row["cos_best_arm"] = (float(arm_cos[best])
                                           if np.isfinite(arm_cos[best]) else None)
                    row["cos_pooled"] = _cos(tau_new[sel].mean(0), tau_real[sel].mean(0))
                _assert_same_keys(nb, blk, f"pools.{pname}")
                del tau_real
                new["pools"][pname] = nb
                tz[f"{pname}/tau_gen"] = tau_new.astype(np.float32)
                tz[f"{pname}/tau_norm_gen"] = tau_norm_new
                tz[f"{pname}/arm_cos"] = arm_cos
                tz[f"{pname}/arm_pearson"] = arm_pear
                tz[f"{pname}/group_of_arm"] = gof.astype(np.int32)
                if cproj is not None:
                    tz[f"{pname}/contrast_proj"] = cproj

                rec = {"baseline_check": {"tol": args.baseline_tol, "max_abs_diff": worst,
                                          "fields": checks}}
                bv = acc.get("bias_along_v")
                if bv is not None and syn is not None:
                    err = tau_new[i_est] - truth["tau"][np.intersect1d(
                        keys, truth["arm_key"], return_indices=True)[2]]
                    proj = np.einsum("ag,ag->a", err, directions_for(syn, comp_sh))
                    halves = np.array([int(half_of_arm.get(k, -1)) for k in shared])
                    sc = np.isin(comp_sh, list(scored_ci))
                    rec["scored_high_minus_low_mean"] = bv["scored_high_minus_low_mean"]
                    rec["scored_contrast_se_compound_cluster"] = clustered_contrast_se(
                        proj, halves, comp_sh, sc, seed=args.seed)
                bc = acc.get("bias_along_contrast")
                if bc is not None and cproj is not None:
                    okc = np.isfinite(cproj[i_est])
                    halves = np.array([int(half_of_arm.get(k, -1)) for k in shared])
                    sc = np.isin(comp_sh, list(scored_ci))
                    rec["line_contrast_scored_high_minus_low_mean"] = bc.get(
                        "scored_high_minus_low_mean")
                    rec["line_contrast_se_compound_cluster"] = clustered_contrast_se(
                        np.where(okc, cproj[i_est], 0.0), np.where(okc, halves, -1),
                        comp_sh, sc, seed=args.seed)
                pool_recs[pname] = rec

            tgt_delta_rec = {
                "norm": _q(np.linalg.norm(delta, axis=1)),
                "n_eff": _q(n_eff),
                "n_eff_below_2": int((n_eff < 2).sum()),
                "sum_w_equals_n_nu": {"checked": wmode_rec == "counts",
                                      "max_rel_err": max_rel, "tol": 1e-6},
            }
            if syn is not None:
                gcomp = np.array([int(str(g).split("|")[0]) for g in gnames], dtype=np.int64)
                tgt_delta_rec["proj_on_v"] = _q(
                    np.einsum("ag,ag->a", delta, directions_for(syn, gcomp)))

            lbl = f"p1_{SRC_SHORT[src_arm]}_{mode}"
            new["targeting"] = {
                "method": METHOD, "version": VERSION, "arm": lbl, "source_arm": src_arm,
                "source_json": src_path, "source_tau_npz": src_path[:-5] + "_tau.npz",
                "source_gen_npz": gen_path,
                "estimand": ("tau_hat_P1(a) = tau_hat_gen(a) + delta_{g(a)}; delta_g = "
                             "sum_{i in g} w_i (Y_i - mu_hat_i) / sum_{i in g} w_i (Hajek, "
                             "over kept train rows). The vehicle side is uncorrected."),
                "confounder": conf, "group_key": list(POSITIVITY_KEYS[gconf])[:-1],
                "group_source": ("positivity_cell_minus_confounder" if args.group_key == "auto"
                                 else f"positivity_cell_minus_confounder({gconf})"),
                "nuisance_dir": nz,
                "weights": {"mode": wmode_rec, "is_control": mode in CONTROL_MODES,
                            "path": wpath, "sha1": wsha, "split_fingerprint": wfp,
                            "n_rows": int(w_all.size), "w_mean": float(w_all.mean()),
                            "w_min": float(w_all.min()), "w_max": float(w_all.max())},
                "rows": {"n_train": int(train_idx.size), "n_train_treated": int(tr_t.size),
                         "n_groups": int(gnames.size),
                         "min_rows_per_group": int(nrows[nrows > 0].min()) if (nrows > 0).any() else 0,
                         "n_groups_uncorrected": uncorrected},
                "delta": tgt_delta_rec, "pools": pool_recs,
                "vehicle": {"corrected": False,
                            "note": "real DMSO wells are never thinned"},
                "recomputed": ["accuracy", "tau_norm_est",
                               "amplitude_ratio_est_over_denoised", "effects.*_est"],
                "inherited": ["learned_syn_effect", "quality", "per_compound", "reference",
                              "mu0_*", "within_row_sd_gen", "tau_norm_real*", "floor_*"],
                "code_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest(),
            }
            new["generator"] = dict(new["generator"], targeted_as=lbl)
            new["artifacts"] = dict(new.get("artifacts") or {},
                                    tau_npz=(out_json[:-5] + "_tau.npz"
                                             if args.npz != "none" else None),
                                    gen_npz=gen_path)
            new["elapsed_sec"] = round(time.time() - t_start, 1)

            if args.dry_run:
                print(f"[p1] DRY {lbl}: ||delta|| median "
                      f"{tgt_delta_rec['norm']['median']:.3f}; would write "
                      f"{os.path.basename(out_json)}", flush=True)
                continue
            for guard in (src_path, (doc.get("truth") or {}).get("path")):
                if guard and os.path.abspath(out_json) == os.path.abspath(guard):
                    raise SystemExit(f"[p1] refusing to write onto {guard}")
            _atomic_write(out_json, lambda f: json.dump(new, f, indent=2, default=_jsonable))
            if args.npz != "none":
                arrs = dict(tz) if args.npz == "full" else {
                    k: v for k, v in tz.items() if k.endswith(("/arm_key", "/group_of_arm"))}
                arrs.update({"targeting/group_name": gnames.astype(str),
                             "targeting/delta": delta.astype(np.float32),
                             "targeting/sum_w": sw, "targeting/n_nu": n_nu,
                             "targeting/n_rows": nrows, "targeting/n_eff": n_eff})
                np.savez_compressed(out_json[:-5] + "_tau.npz", **arrs)
            written.append(out_json)
            bvm = ((pool_recs.get("all") or {}).get("scored_high_minus_low_mean")
                   if syn is not None else
                   (pool_recs.get("all") or {}).get("line_contrast_scored_high_minus_low_mean"))
            print(f"[p1] {lbl:18s} {os.path.basename(run_dir):26s} "
                  f"||delta|| med {tgt_delta_rec['norm']['median']:7.3f}  "
                  f"n_eff med {tgt_delta_rec['n_eff']['median']:5.2f}  "
                  + (f"scored high-low {bvm:+.3f}" if bvm is not None else ""), flush=True)
        del resid

    print(f"\n[p1] wrote {len(written)} targeted document(s) in "
          f"{time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
