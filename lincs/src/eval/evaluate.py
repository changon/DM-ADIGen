"""LINCS gene-space evaluation: the dose-specific ATE, real and generated.

    tau_hat(c, d) = mu_hat(c, d) - mu_hat(0)   in R^978, on plate-centred,
                                               z-scored Level 3 (IMPLEMENT.md §3.7)

`--source real` writes the oracle; `--source generated --truth <oracle.json>`
scores one trained arm against it. Arms are keyed on `splits.arm_keys`, i.e.
(compound_idx, dose_level) with every vehicle row in `CONTROL_ARM` (P2).

This is a rewrite, not a port: RxRx's `evaluate.py` is a COVID rescue panel on
image embeddings (IMPLEMENT.md §2.2, §3.7). What is copied from it: the pool /
anchor mask algebra, `_rng` / `_accuracy` / `_rankdata` / `_corr` / `_cos`, the
`--truth` frame guard, and the output-path anti-clobber tagging. What is gone:
PANEL_HITS and the literature AUROC, the Mock <-> untreated-infected axis and
`rescue`, every encoder (domain ResNet18 / OpenPhenom / Inception), TVN, the
image VAE roundtrip, and the pixel FID suite.

Phase 4 settings are fixed in IMPLEMENT.md §3.14 (E1-E12) and are the defaults
here: `checkpoint-0499` EMA weights, no guidance, 16 samples per real row,
`--min_dose_n 2`, 100 sampler steps, both Frechet variants plus MMD.

    # oracle (CPU job: numpy only, torch is never imported on this path)
    python -m src.eval.evaluate --source real --pool all

    # one arm, scored against it
    python -m src.eval.evaluate --source generated \
        --run_dir runs/mcf7_24h/mlp-B_conditional_ddpm_s0 \
        --truth runs/mcf7_24h/eval_artifacts/oracle_mcf7_24h_poolall.json

Writes `<out>.json` (aggregates + per-arm scalars), `<out>_tau.npz` (the
(K, 978) tau matrix, which `--truth` reads), and for a generated arm
`<out>_gen.npz` (per-row means, so metrics can be recomputed without
re-sampling).
"""
from __future__ import annotations

import argparse
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

from src.data.build_dataset import CONTEXT_SOURCE_COLUMNS, _atomic_write  # noqa: E402
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes  # noqa: E402
from src.data.splits import CONTROL_ARM, arm_keys, dose_half, load_splits  # noqa: E402
from src.data.synthetic import inject, load_syn_meta  # noqa: E402
from src.eval import dist_metrics as dm  # noqa: E402
from src.nuisances.precompute_cmean import group_means  # noqa: E402
from src.spec import (  # noqa: E402
    PLATE_CENTER_MODES, add_adjustment_set_cli, add_paths_cli, add_syn_cli,
    apply_paths_args, config_from_args, format_role_summary)

# Table columns the estimand needs. `dose_level` is NOT in LincsDataset's
# TABLE_COLUMNS, and the curated panel needs the names, so read the table
# directly (the precompute_cmean pattern). CONTEXT_SOURCE_COLUMNS must be in
# here too: `generation.targets_from_rows` calls `context_for_rows`, which reads
# them off this same frame.
META_COLUMNS = tuple(dict.fromkeys(
    ("compound_idx", "dose_level", "is_control", "pert_id", "pert_iname",
     "log10_conc") + tuple(CONTEXT_SOURCE_COLUMNS)))

# The reference compound list of IMPLEMENT.md §3.7, by `pert_iname`, with the
# well counts the plan quotes. These are MCF7 24 h compounds that actually have
# wells in this table -- NOT RxRx's remdesivir panel.
PANEL = (
    ("proteasome", ("bortezomib", "MG-132")),
    ("hsp90", ("geldanamycin", "NVP-AUY922", "alvespimycin")),
    ("hdac", ("entinostat", "belinostat", "vorinostat")),
    ("mtor", ("sirolimus", "torin-1")),
    ("mek", ("PD-0325901", "selumetinib", "trametinib")),
    ("er", ("estradiol", "tamoxifen", "fulvestrant")),
)

ORACLE_SOURCE = "real"


# ---------------------------------------------------------------------------
# Numeric helpers (copied from RxRx19a/src/eval/evaluate.py)
# ---------------------------------------------------------------------------

def _rng(seed: int, *key) -> np.random.Generator:
    """An independent stream per (seed, purpose). Every draw site names itself,
    so a draw does not depend on how many draws ran before it."""
    h = hashlib.blake2b("|".join(str(k) for k in key).encode(), digest_size=8)
    return np.random.default_rng([seed, int.from_bytes(h.digest(), "little")])


def _rankdata(v: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (the Spearman convention)."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(1, len(v) + 1)
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq)); np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _accuracy(est: np.ndarray, truth: np.ndarray) -> dict:
    """How close, and how correctly ordered, is `est` relative to `truth`?"""
    est = np.asarray(est, dtype=np.float64); truth = np.asarray(truth, dtype=np.float64)
    d = est - truth
    return {"n": int(len(est)), "mse": float(np.mean(d ** 2)),
            "rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))),
            "bias": float(np.mean(d)),
            "spearman_rho": _corr(_rankdata(est), _rankdata(truth)),
            "pearson_r": _corr(est, truth), "truth_sd": float(np.std(truth))}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two (K, G) matrices."""
    na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
    return (A * B).sum(1) / (na * nb + 1e-12)


def _pearson_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r between two (K, G) matrices (cosine after centring)."""
    return _cos_rows(A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True))


def _q(v: np.ndarray, name: str = "") -> dict:
    """Summary of a scalar array, for the JSON."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": None, "mean": None, "p10": None, "p90": None}
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


# ---------------------------------------------------------------------------
# Arm table
# ---------------------------------------------------------------------------

def arm_table(y: np.ndarray, keys: np.ndarray, rows: np.ndarray) -> dict:
    """Per-arm mean of `y` over `rows`, and tau against the vehicle arm.

    `y` and `keys` are already restricted to `rows` (same order).
    """
    uniq, mu, counts = group_means(y, keys)
    uniq = uniq.astype(str)          # object -> <U, so the npz needs no pickle
    ctl = np.flatnonzero(uniq == CONTROL_ARM)
    if ctl.size != 1:
        raise RuntimeError(
            f"expected exactly one {CONTROL_ARM!r} arm among {uniq.size} arms, found "
            f"{ctl.size}. Without vehicle wells in the pool there is no mu_hat(0).")
    i0 = int(ctl[0])
    mu0 = mu[i0].copy()
    trt = np.ones(uniq.size, dtype=bool); trt[i0] = False
    return {"arm_key": uniq[trt], "mu": mu[trt], "tau": mu[trt] - mu0,
            "n_wells": counts[trt].astype(np.int64),
            "mu0": mu0, "n_dmso": int(counts[i0]), "rows": rows}


def _split_key(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """'<compound_idx>|<dose>' -> (compound_idx, dose string)."""
    if len(k) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=str)
    parts = np.array([str(s).split("|", 1) for s in k], dtype=object)
    return parts[:, 0].astype(np.int64), parts[:, 1].astype(str)


# ---------------------------------------------------------------------------
# Reference scale (E11): what an arm's tau looks like when there is no effect,
# and how well a real arm agrees with itself.
# ---------------------------------------------------------------------------

def noise_floor(y_dmso: np.ndarray, sizes: np.ndarray, n_draws: int,
                seed: int) -> dict:
    """||tau_hat|| for pseudo-arms of n real DMSO wells, per arm size n.

    §3.8.1's responder rule is "||tau_hat|| > 1.5x the noise floor" but never
    defines the floor (E11 does). Each draw takes n vehicle wells as the
    "treated" side and a DISJOINT vehicle subsample as the mu_hat(0) side, so the
    two sides are independent exactly as a real arm's are.
    """
    n_dmso = int(y_dmso.shape[0])
    out: dict[str, dict] = {}
    for n in np.unique(np.asarray(sizes, dtype=np.int64)):
        n = int(n)
        # Hold the same number of wells back for the anchor side as the real
        # mu_hat(0) uses, but never more than leaves n wells for the arm side.
        n_ref = min(n_dmso - n, n_dmso // 2)
        if n < 1 or n_ref < 1:
            continue
        rng = _rng(seed, "floor", n)
        vals = np.empty(int(n_draws), dtype=np.float64)
        for i in range(int(n_draws)):
            pick = rng.choice(n_dmso, n + n_ref, replace=False)
            a = y_dmso[pick[:n]].mean(0)
            b = y_dmso[pick[n:]].mean(0)
            vals[i] = np.linalg.norm(a - b)
        out[str(n)] = {"n_draws": int(n_draws), "n_ref": int(n_ref),
                       "median": float(np.median(vals)),
                       "p90": float(np.percentile(vals, 90))}
    return out


def floor_for_sizes(floor: dict, sizes: np.ndarray) -> np.ndarray:
    """The floor median for each arm size, nearest available size if exact is
    missing (so a truth JSON stays usable on a pool with other arm sizes)."""
    have = np.array(sorted(int(k) for k in floor), dtype=np.int64)
    if have.size == 0:
        return np.full(len(sizes), np.nan)
    med = np.array([floor[str(int(h))]["median"] for h in have], dtype=np.float64)
    idx = np.abs(have[None, :] - np.asarray(sizes, dtype=np.int64)[:, None]).argmin(1)
    return med[idx]


def split_half_reliability(y: np.ndarray, keys: np.ndarray, mu0: np.ndarray,
                           seed: int) -> dict:
    """cos(tau_A, tau_B) from two disjoint halves of each arm's real wells.

    This is the ceiling: the best cosine ANY generator can score against a
    2-or-3-well oracle. Without it a median per-arm cosine cannot be read -- with
    ~3 wells per arm the oracle tau is itself mostly noise (E11).
    """
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    bounds = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1], [True]]))
    per_n: dict[int, list[float]] = {}
    arm_cos: dict[str, float] = {}
    rng = _rng(seed, "reliability")
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        key = str(ks[lo])
        if key == CONTROL_ARM:
            continue
        idx = order[lo:hi]
        n = idx.size
        if n < 2:
            continue
        perm = rng.permutation(n)
        h = n // 2
        a = y[idx[perm[:h]]].mean(0) - mu0
        b = y[idx[perm[h:]]].mean(0) - mu0
        c = _cos(a, b)
        arm_cos[key] = c
        per_n.setdefault(n, []).append(c)
    return {"by_n": {str(n): _q(np.array(v)) for n, v in sorted(per_n.items())},
            "overall": _q(np.array(list(arm_cos.values()))),
            "per_arm": arm_cos}   # per_arm is written to the npz, not the JSON


# ---------------------------------------------------------------------------
# Quality grouping
# ---------------------------------------------------------------------------


def corrected_ceiling(r_half: np.ndarray) -> np.ndarray:
    """The largest cos(tau_gen, tau_oracle) a perfect generator can reach.

    `split_half_reliability` measures cos(tau_A, tau_B) between two HALVES of an
    arm's wells, so it is the reliability of a half-sized estimate, not of the
    oracle. Write s = ||tau||^2 and v = sigma^2 / n for the full arm; then

        r_half = s / (s + 2v)                      (each half has noise 2v)
        r_full = s / (s + v) = 2 r_half / (1 + r_half)      (Spearman-Brown)

    and a noiseless generator correlates with the oracle at
    cos = sqrt(r_full). Comparing a generated cosine with the raw split-half
    value understates the model: on `mcf7_24h` the raw median is 0.451 while the
    attainable bound is 0.788.

    A non-positive r_half means the arm is indistinguishable from noise, so no
    bound is defined and the entry is NaN.
    """
    r = np.asarray(r_half, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_full = 2.0 * r / (1.0 + r)
        out = np.sqrt(np.clip(r_full, 0.0, 1.0))
    return np.where(r > 0, out, np.nan)



def bias_along_v(tau_est: np.ndarray, tau_truth: np.ndarray, arm_key: np.ndarray,
                 v: np.ndarray, half_of_arm: dict, scored_ci: set[int]) -> dict:
    """Signed projection of the generator's error onto the injected direction v.

    §3.8.2: "The bias lives along v. Report the signed projection
    <tau_gen(a) - tau_oracle(a), v> for scored arms by dose half, rare vs
    non-rare." The thinning over-keeps one syn_c level in one dose half and the
    other level in the other, so a confounded (naive) generator should show a
    projection of OPPOSITE SIGN in the two halves, while `dr` should show none.
    That sign flip is the signature; a two-sided magnitude would hide it.
    """
    comp, _ = _split_key(arm_key)
    proj = (np.asarray(tau_est, dtype=np.float64)
            - np.asarray(tau_truth, dtype=np.float64)) @ np.asarray(v, dtype=np.float64)
    halves = np.array([int(half_of_arm.get(k, -1)) for k in arm_key])
    scored = np.isin(comp, list(scored_ci)) if scored_ci else np.zeros(comp.size, bool)
    out = {"n_scored_arms": int(scored.sum()), "n_unscored_arms": int((~scored).sum())}
    for name, sel in (("scored", scored), ("unscored", ~scored)):
        for hname, hsel in (("low", halves == 0), ("high", halves == 1)):
            out[f"{name}_{hname}"] = _q(proj[sel & hsel])
        out[name] = _q(proj[sel])
    # The contrast the lever creates: high minus low, within each group.
    for name, sel in (("scored", scored), ("unscored", ~scored)):
        hi, lo = proj[sel & (halves == 1)], proj[sel & (halves == 0)]
        out[f"{name}_high_minus_low_median"] = (
            float(np.median(hi) - np.median(lo)) if hi.size and lo.size else None)
    return out


def quality_groups(dose_level: np.ndarray, is_control: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Finest partition for the distribution metrics: one group per treated
    `dose_level` plus one for the vehicles. Coarser pools (all treated) are
    concatenations of these, so no sample is stored twice."""
    lab = np.where(np.asarray(is_control).astype(bool), "dmso",
                   np.array([f"dose_{v:g}" for v in np.asarray(dose_level, dtype=float)]))
    names = ["dmso"] + sorted({s for s in lab if s != "dmso"},
                              key=lambda s: float(s.split("_", 1)[1]))
    code = {nm: i for i, nm in enumerate(names)}
    return np.array([code[s] for s in lab], dtype=np.int64), names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    add_syn_cli(p)
    p.add_argument("--source", default="real", choices=("real", "generated"))
    p.add_argument("--pool", default="all", choices=("all", "train", "holdout"),
                   help="Real rows the estimand scores (and, for --source generated, "
                        "the rows sampled at). E12: 'all' for the DDPM headline arms, "
                        "'holdout' for the FM cmean ablation arms.")
    p.add_argument("--min_dose_n", type=int, default=2,
                   help="Skip arms with fewer than this many wells in the pool (E4). "
                        "Applies to every sub-pool except holdout.")
    p.add_argument("--min_dose_n_holdout", type=int, default=1,
                   help="The same threshold for the holdout sub-pool. It must default to 1: "
                        "under the P3 split a 3-well arm puts exactly ONE well in the holdout, "
                        "so --min_dose_n 2 would empty the pool. §3.7 reads holdout in "
                        "aggregate (per dose level, per compound) for that reason.")
    p.add_argument("--plate_center", default=None, choices=PLATE_CENTER_MODES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Output JSON (default: derived, see --help_paths).")
    # generator
    p.add_argument("--run_dir", default=None, help="Arm run dir (required for --source generated).")
    p.add_argument("--gen_epoch", type=int, default=499, help="-> checkpoint-NNNN (E1).")
    p.add_argument("--which_wgt", default="ema", choices=("ema", "train"),
                   help="ema = model_1.safetensors (E1).")
    p.add_argument("--n_per_row", type=int, default=16, help="Generated wells per real row (E3).")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG weight. E2 fixes the headline at 1.0 (no guidance).")
    p.add_argument("--sampler", default="ddim", choices=("ddim", "ddpm", "dpm"),
                   help="DDPM arms only; FM arms always use the flow Euler sampler.")
    p.add_argument("--num_inference_steps", type=int, default=100, help="E10.")
    p.add_argument("--gen_batch_size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gen_anchors", action="store_true",
                   help="mu_hat(0) from GENERATED vehicle wells instead of real ones. "
                        "§3.7 asks for both: real isolates treated-arm error, generated tests Y(0).")
    p.add_argument("--save_gen", action="store_true", default=True,
                   help="Write <out>_gen.npz (per-row generated means) so metrics can be "
                        "recomputed without re-sampling.")
    p.add_argument("--no_save_gen", dest="save_gen", action="store_false")
    # scoring
    p.add_argument("--truth", default=None, help="Oracle JSON from a --source real run.")
    p.add_argument("--n_floor_draws", type=int, default=200, help="Noise-floor draws per arm size (E11).")
    p.add_argument("--responder_mult", type=float, default=1.5,
                   help="||tau|| > mult x floor marks a responder (§3.8.1).")
    p.add_argument("--quality_n", type=int, default=4096,
                   help="Samples kept per dose group for MMD / Frechet. 0 disables QUALITY.")
    p.add_argument("--mmd_max_samples", type=int, default=4000)
    p.add_argument("--pca_var", type=float, default=0.90,
                   help="Explained-variance target for the top-k PC Frechet (E5).")
    p.add_argument("--extra_metrics", action="store_true", help="Also report KID and PRDC.")
    p.add_argument("--probe", action="store_true",
                   help="Optional TRTS: a linear probe on dose_level, trained on real, tested on generated.")
    return p.parse_args()


def _out_paths(cfg, args, arch: dict | None,
               syn_meta: dict | None = None) -> tuple[str, str]:
    """(out_json, oracle_json). Copied from RxRx's tagging, which refuses to
    write a non-oracle run onto the oracle's path."""
    pop = cfg.population.name
    if args.source == ORACLE_SOURCE:
        base_dir = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    else:
        base_dir = os.path.join(args.run_dir, "eval_artifacts")
    oracle_dir = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    ptag = f"_pool{args.pool}"
    # The step-C injection changes the ground truth, so an injected oracle is a
    # DIFFERENT oracle and must not land on the v1 oracle's path (§3.8.2).
    stag_syn = ""
    if float(cfg.outcome.syn_effect) != 0:
        stag_syn = f"_syn{float(cfg.outcome.syn_effect):g}"
        if int((syn_meta or {}).get("vec_seed", 0)):
            stag_syn += f"-v{int(syn_meta['vec_seed'])}"
    otag = f"oracle_{pop}{ptag}{stag_syn}"
    if args.source == ORACLE_SOURCE:
        tag = otag
    else:
        tag = f"gen_{os.path.basename(os.path.normpath(args.run_dir))}_ep{args.gen_epoch:04d}{ptag}"
        tag += f"_{args.which_wgt}{stag_syn}"
        if abs(args.guidance_scale - 1.0) > 1e-6:
            tag += f"_g{args.guidance_scale:g}"
        if args.n_per_row != 16:
            tag += f"_m{args.n_per_row}"
        if args.num_inference_steps != 100:
            tag += f"_t{args.num_inference_steps}"
        if args.gen_anchors:
            tag += "_genanchor"
        if args.seed:
            tag += f"_s{args.seed}"
    out = args.out or os.path.join(base_dir, f"{tag}.json")
    oracle = os.path.join(oracle_dir, f"{otag}.json")
    if args.source != ORACLE_SOURCE and os.path.abspath(out) == os.path.abspath(oracle):
        raise RuntimeError(
            f"refusing to write a --source {args.source} run onto the oracle path "
            f"{oracle}. Pass a different --out.")
    return out, oracle


# arch.json / oracle fields that must agree before a generated arm may be
# compared with an oracle. A mismatch means the two tau's are not on one scale.
# What must match between a generated run and its oracle: everything that
# defines the OUTCOME. `adjustment_set` is deliberately NOT here -- the oracle is
# a mean of real wells and never builds a cond spec, so its tau is bit-identical
# with and without a C (verified: max |diff| 0.0). Requiring it would force a
# redundant oracle per C holding the same numbers, and would block step C, whose
# arms condition on syn_c while the oracle marginalises over it (§3.8.1). The C
# that does matter -- the arm's own, which it must be sampled under -- is checked
# by `generation.check_arm_against_data`.
TRUTH_GUARD = ("population", "table_fingerprint", "split_fingerprint",
               "gene_order_sha1", "plate_center", "normalize_mean", "normalize_std",
               "syn_effect", "syn_seed", "syn_beta", "syn_v_sha1")

# 'all' contains both of the others, so an oracle on 'all' can score any pool.
_POOL_CONTAINS = {"all": ("all", "train", "holdout"), "train": ("train",),
                  "holdout": ("holdout",)}


def _min_n_for(pname: str, args) -> int:
    """The per-arm well threshold for one sub-pool (see --min_dose_n_holdout)."""
    return int(args.min_dose_n_holdout if pname == "holdout" else args.min_dose_n)


def _floor_sizes(sizes: np.ndarray, small_max: int = 20, n_large: int = 12) -> np.ndarray:
    """Arm sizes to evaluate the floor at: every small size exactly, then a
    geometric ladder. `floor_for_sizes` maps the rest to the nearest.

    Arm sizes run 1..~980 on mcf7_24h, but 90% of arms have 2 or 3 wells, so
    evaluating every distinct size would spend most of the time on singletons in
    the tail.
    """
    u = np.unique(np.asarray(sizes, dtype=np.int64))
    small = u[u <= small_max]
    large = u[u > small_max]
    if large.size > n_large:
        pick = np.unique(np.geomspace(large.min(), large.max(), n_large).round().astype(np.int64))
        large = large[np.abs(large[None, :] - pick[:, None]).argmin(1)]
    return np.unique(np.concatenate([small, large]))


def _pool_block(real: dict, gen: dict | None, *, meta_by_arm: dict, floor: dict,
                reliability_cos: np.ndarray, reliability_summary: dict, args,
                panel_idx: dict, mu0_real: np.ndarray, mu0_scale: float,
                quality: dict | None, truth: dict | None,
                syn: dict | None = None, half_of_arm: dict | None = None,
                scored_ci: set[int] | None = None,
                mu0_offset: np.ndarray | None = None) -> tuple[dict, dict]:
    """Everything reported for one sub-pool: (json_block, npz_arrays)."""
    keys = real["arm_key"]
    tau_real = real["tau"]
    n_wells = real["n_wells"]
    comp, dose = _split_key(keys)
    floor_med = floor_for_sizes(floor, n_wells)
    tau_norm_real = np.linalg.norm(tau_real, axis=1)
    ratio = tau_norm_real / np.where(floor_med > 0, floor_med, np.nan)
    tau_norm_real_dn = np.sqrt(np.clip(tau_norm_real ** 2 - floor_med ** 2, 0.0, None))
    responder_arm = ratio > float(args.responder_mult)

    est = gen if gen is not None else real
    tau_est = est["tau"]
    tau_norm_est = np.linalg.norm(tau_est, axis=1)

    block: dict = {
        "n_arms": int(keys.size),
        "n_rows": int(len(real["rows"])),
        "n_dmso": int(real["n_dmso"]),
        "mu0_norm_real": float(np.linalg.norm(mu0_real)),
        # E[||mean of n independent DMSO wells||] under the pool's own DMSO
        # spread: the scale ||mu_hat(0)|| should sit at when z = 0 is the
        # vehicle (§3.2). The ratio, not the norm, is the check.
        "mu0_norm_expected": float(mu0_scale),
        # Under the step-C injection the vehicles move too, by the KNOWN offset
        # beta * mean(syn_c over this pool's DMSO) * v (§3.8.2), so the ratio is
        # taken after removing it -- otherwise it reads ~beta/2 / scale (19.6x on
        # mcf7_24h) and says nothing about the centring.
        "mu0_offset_norm": (None if mu0_offset is None
                            else float(np.linalg.norm(mu0_offset))),
        "mu0_norm_over_expected": (
            float(np.linalg.norm(mu0_real if mu0_offset is None
                                 else mu0_real - mu0_offset) / mu0_scale)
            if mu0_scale > 0 else None),
        "tau_norm_real": _q(tau_norm_real),
        "tau_norm_est": _q(tau_norm_est),
        # E[||tau_hat||^2] = ||tau||^2 + E[||noise||^2], and at n = 3 the floor
        # (14.5) is most of the measured median (15.2). So the amplitude a
        # generator should match is the noise-corrected one, not the raw median.
        "tau_norm_real_denoised": _q(tau_norm_real_dn),
        "amplitude_ratio_est_over_denoised": _q(
            tau_norm_est / np.where(tau_norm_real_dn > 0, tau_norm_real_dn, np.nan)),
        "floor_ratio_real": _q(ratio),
        "n_responder_arms": int(responder_arm.sum()),
        "responder_arm_frac": float(responder_arm.mean()) if keys.size else None,
        "reference": {"floor": floor, "reliability": reliability_summary},
    }
    if gen is not None:
        block["mu0_norm_gen"] = float(np.linalg.norm(gen["mu0"]))
        _g0 = gen["mu0"] if mu0_offset is None else gen["mu0"] - mu0_offset
        block["mu0_norm_gen_over_expected"] = (float(np.linalg.norm(_g0) / mu0_scale)
                                               if mu0_scale > 0 else None)
        block["within_row_sd_gen"] = est.get("within_row_sd")

    # ---- ACCURACY against the oracle ----
    if truth is not None:
        shared, i_est, i_tr = np.intersect1d(keys, truth["arm_key"],
                                            return_indices=True)
        if shared.size == 0:
            raise RuntimeError("no arm is present in both this run and the oracle")
        te, tt = tau_est[i_est], truth["tau"][i_tr]
        cos = _cos_rows(te, tt)
        pear = _pearson_rows(te, tt)
        resp = truth["responder"][i_tr].astype(bool)
        # The ceiling: how well the ORACLE agrees with itself on this arm, so
        # it comes from the truth side, never from the generated one. The raw
        # split-half value needs the Spearman-Brown step to become a bound on
        # cos(generated, oracle) -- see `corrected_ceiling`.
        ceil = truth["reliability_cos"][i_tr]
        ceil_c = corrected_ceiling(ceil)
        acc = {
            "n_matched": int(shared.size),
            "n_unmatched_est": int(keys.size - shared.size),
            "n_unmatched_truth": int(truth["arm_key"].size - shared.size),
            "cos_all": _q(cos), "pearson_all": _q(pear),
            "cos_responder": _q(cos[resp]), "pearson_responder": _q(pear[resp]),
            "n_responder": int(resp.sum()),
            # The ceiling exists only for arms with >= 2 oracle wells, which on
            # the holdout is a small minority. Summarise the cosine over exactly
            # those arms too, so "cos vs ceiling" compares one arm set with
            # itself rather than 1-well arms against multi-well ones.
            "n_responder_with_ceiling": int((resp & np.isfinite(ceil)).sum()),
            "split_half_responder": _q(ceil[resp]),
            "reliability_ceiling_responder": _q(ceil_c[resp]),
            "cos_responder_with_ceiling": _q(cos[resp & np.isfinite(ceil_c)]),
            "cos_over_ceiling_responder": _q(
                cos[resp & np.isfinite(ceil_c)]
                / np.where(ceil_c[resp & np.isfinite(ceil_c)] > 0,
                           ceil_c[resp & np.isfinite(ceil_c)], np.nan)),
            "tau_norm": _accuracy(np.linalg.norm(te, axis=1), np.linalg.norm(tt, axis=1)),
            "tau_norm_responder": _accuracy(np.linalg.norm(te[resp], axis=1),
                                           np.linalg.norm(tt[resp], axis=1)) if resp.any() else None,
            # Pooled over every gene of every matched arm.
            "pooled_gene": _accuracy(te.reshape(-1), tt.reshape(-1)),
        }
        # Per dose level and per compound, averaged over the arms in each.
        by_dose = {}
        for d in np.unique(dose[i_est]):
            sel = dose[i_est] == d
            by_dose[str(d)] = {"n_arms": int(sel.sum()),
                               "cos_mean_tau": _cos(te[sel].mean(0), tt[sel].mean(0)),
                               "cos_median_arm": float(np.median(cos[sel]))}
        acc["by_dose_level"] = by_dose
        cmp_cos: list[float] = []
        for c in np.unique(comp[i_est]):
            sel = comp[i_est] == c
            cmp_cos.append(_cos(te[sel].mean(0), tt[sel].mean(0)))
        acc["per_compound_cos_pooled"] = _q(np.array(cmp_cos))
        # Step C's readout, alongside the aggregate metrics (§3.8.2).
        if syn is not None and half_of_arm is not None:
            acc["bias_along_v"] = bias_along_v(te, tt, shared, syn["v"],
                                              half_of_arm, scored_ci or set())
        block["accuracy"] = acc
        arm_cos, arm_pear = np.full(keys.size, np.nan), np.full(keys.size, np.nan)
        arm_cos[i_est] = cos; arm_pear[i_est] = pear
    else:
        block["accuracy"] = None
        arm_cos = arm_pear = np.full(keys.size, np.nan)

    # ---- EFFECTS: the §3.7 curated list ----
    eff = []
    for family, names in PANEL:
        for nm in names:
            ci = panel_idx.get(nm)
            if ci is None:
                continue
            sel = np.flatnonzero(comp == ci)
            if sel.size == 0:
                continue
            best = sel[np.argmax(tau_norm_real[sel])]
            row = {"family": family, "pert_iname": nm, "compound_idx": int(ci),
                   "n_arms": int(sel.size), "n_wells_total": int(n_wells[sel].sum()),
                   "best_dose_level": dose[best],
                   "tau_norm_real_max": float(tau_norm_real[best]),
                   "floor_ratio_max": float(ratio[best]) if np.isfinite(ratio[best]) else None,
                   "responder": bool(responder_arm[sel].any())}
            if gen is not None:
                row["tau_norm_est_max"] = float(tau_norm_est[best])
                row["cos_best_arm"] = float(arm_cos[best]) if np.isfinite(arm_cos[best]) else None
                row["cos_pooled"] = _cos(tau_est[sel].mean(0), tau_real[sel].mean(0))
            eff.append(row)
    block["effects"] = eff

    # ---- per compound (Phase 5 reads this to build responders.json) ----
    per_comp = []
    for c in np.unique(comp):
        sel = comp == c
        b = sel & (tau_norm_real == tau_norm_real[sel].max())
        per_comp.append({
            "compound_idx": int(c),
            "pert_id": meta_by_arm["pert_id"].get(int(c)),
            "pert_iname": meta_by_arm["pert_iname"].get(int(c)),
            "n_arms": int(sel.sum()),
            "max_tau_norm_real": float(tau_norm_real[sel].max()),
            "max_floor_ratio": (float(np.nanmax(ratio[sel]))
                                if np.isfinite(ratio[sel]).any() else None),
            "responder": bool(responder_arm[sel].any()),
            "best_dose_level": str(dose[np.flatnonzero(b)[0]]),
        })
    block["per_compound"] = per_comp
    block["n_responder_compounds"] = int(sum(r["responder"] for r in per_comp))
    block["responder_compound_frac"] = (block["n_responder_compounds"] / len(per_comp)
                                        if per_comp else None)
    block["quality"] = quality

    npz = {"arm_key": keys, "compound_idx": comp, "dose_level": dose,
           "n_wells": n_wells, "tau": tau_real.astype(np.float32),
           "mu0": mu0_real.astype(np.float32), "tau_norm": tau_norm_real,
           "floor_median": floor_med, "floor_ratio": ratio,
           "tau_norm_denoised": tau_norm_real_dn,
           "responder": responder_arm,
           "reliability_cos": np.asarray(reliability_cos, dtype=np.float64)}
    if gen is not None:
        npz.update({"tau_gen": tau_est.astype(np.float32),
                    "tau_norm_gen": tau_norm_est,
                    "mu0_gen": gen["mu0"].astype(np.float32),
                    "arm_cos": arm_cos, "arm_pearson": arm_pear})
    return block, npz


def _quality_block(y_real: np.ndarray, groups_real: np.ndarray, group_names: list[str],
                   reservoir: dict, pca: dict | None, args) -> dict:
    """MMD and both Frechet variants per dose group, plus the pooled treated and
    vehicle pools (E5)."""
    rng = _rng(args.seed, "quality_real")
    out: dict = {"group_names": group_names, "per_group": {}, "pooled": {}}
    if pca is not None:
        out["pca"] = {"k": pca["k"], "var_target": pca["var_target"],
                      "explained": pca["explained"], "n_fit": pca["n_fit"]}
    real_by_g, gen_by_g = {}, {}
    for gi, nm in enumerate(group_names):
        r = y_real[groups_real == gi]
        g = reservoir.get(gi)
        if r.shape[0] and args.quality_n and r.shape[0] > args.quality_n:
            r = r[rng.choice(r.shape[0], args.quality_n, replace=False)]
        real_by_g[nm], gen_by_g[nm] = r, (g if g is not None else np.zeros((0, y_real.shape[1]), np.float32))
        out["per_group"][nm] = [m.to_dict() for m in dm.quality_metrics(
            real_by_g[nm], gen_by_g[nm], pca=pca,
            mmd_max_samples=args.mmd_max_samples, extra=args.extra_metrics,
            seed=args.seed)]
    for pooled, members in (("treated", [n for n in group_names if n != "dmso"]),
                            ("dmso", ["dmso"])):
        r = np.concatenate([real_by_g[n] for n in members], axis=0) if members else None
        g = np.concatenate([gen_by_g[n] for n in members], axis=0) if members else None
        if r is None or r.shape[0] == 0 or g is None or g.shape[0] == 0:
            out["pooled"][pooled] = [{"name": "skipped", "n_real": int(0 if r is None else r.shape[0]),
                                      "n_gen": int(0 if g is None else g.shape[0])}]
            continue
        out["pooled"][pooled] = [m.to_dict() for m in dm.quality_metrics(
            r, g, pca=pca, mmd_max_samples=args.mmd_max_samples,
            extra=args.extra_metrics, seed=args.seed)]
    return out


def _probe_block(y_real: np.ndarray, groups_real: np.ndarray, reservoir: dict,
                 group_names: list[str], seed: int) -> dict:
    """TRTS: a linear probe on the dose group, trained on real Y, tested on
    generated. A conditioning check, not part of the estimand."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        return {"skipped": f"sklearn unavailable ({e})"}
    gi = sorted(reservoir)
    if len(gi) < 2:
        return {"skipped": "fewer than 2 generated groups"}
    Xg = np.concatenate([reservoir[g] for g in gi], 0)
    yg = np.concatenate([np.full(reservoir[g].shape[0], g) for g in gi])
    keep = np.isin(groups_real, gi)
    Xr, yr = y_real[keep], groups_real[keep]
    sc = StandardScaler().fit(Xr)
    clf = LogisticRegression(max_iter=300, multi_class="auto", random_state=seed)
    clf.fit(sc.transform(Xr), yr)
    _, cnt = np.unique(yg, return_counts=True)
    return {"train_acc_real": float(clf.score(sc.transform(Xr), yr)),
            "test_acc_generated": float(clf.score(sc.transform(Xg), yg)),
            "chance": float(cnt.max() / cnt.sum()),
            "n_classes": int(len(gi)), "n_real": int(Xr.shape[0]), "n_gen": int(Xg.shape[0]),
            "labels": [group_names[g] for g in gi]}


def _load_truth(path: str, this: dict, pool: str, min_n: int) -> dict:
    """The oracle tables, after refusing any frame the comparison would corrupt."""
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("source") != ORACLE_SOURCE:
        raise RuntimeError(f"--truth {path} has source={doc.get('source')!r}; "
                           f"the oracle must be a --source real run")
    bad = []
    for k in TRUTH_GUARD:
        if k not in doc:
            print(f"[truth] WARNING: the oracle predates {k!r}; not checked", file=sys.stderr)
            continue
        a, b = doc[k], this[k]
        same = (abs(float(a) - float(b)) < 1e-12 if isinstance(b, float)
                else list(a) == list(b) if isinstance(b, (list, tuple))
                else str(a) == str(b))
        if not same:
            bad.append(f"{k}: oracle {a!r} != this run {b!r}")
    if bad:
        raise RuntimeError("--truth frame mismatch; rebuild the oracle under the "
                           "same frame:\n  - " + "\n  - ".join(bad))
    if list(doc.get("adjustment_set") or []) != list(this.get("adjustment_set") or []):
        print(f"[truth] note: the oracle carries adjustment_set "
              f"{doc.get('adjustment_set')} and this run {this.get('adjustment_set')}. "
              f"That is expected for step C and does not affect tau: the oracle "
              f"marginalises over C.", file=sys.stderr)
    tmn = (doc.get("pools", {}).get(pool) or {}).get("min_dose_n")
    if tmn is not None and int(tmn) != int(min_n):
        raise RuntimeError(
            f"oracle sub-pool {pool!r} kept arms with >= {tmn} wells but this run uses "
            f">= {min_n}; the two arm sets differ, so matched medians would not be "
            f"comparable. Rerun with the same threshold.")
    if pool not in _POOL_CONTAINS.get(str(doc.get("pool")), ()):
        raise RuntimeError(
            f"oracle pool {doc.get('pool')!r} does not contain this run's pool "
            f"{pool!r}; score against an oracle built on 'all' or on {pool!r}")
    npz_path = doc.get("artifacts", {}).get("tau_npz")
    if not npz_path or not os.path.isfile(npz_path):
        npz_path = path[:-5] + "_tau.npz"
    z = np.load(npz_path, allow_pickle=False)
    if f"{pool}/arm_key" not in z:
        raise RuntimeError(f"{npz_path} has no '{pool}' sub-pool table "
                           f"(has {sorted({k.split('/')[0] for k in z.files})})")
    return {"doc": doc, "path": path, "npz": npz_path,
            "arm_key": z[f"{pool}/arm_key"], "tau": z[f"{pool}/tau"],
            "n_wells": z[f"{pool}/n_wells"], "responder": z[f"{pool}/responder"],
            "reliability_cos": z[f"{pool}/reliability_cos"],
            "mu0": z[f"{pool}/mu0"]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    args = _parse_args()
    if args.source == "generated" and not args.run_dir:
        raise SystemExit("--source generated needs --run_dir <runs/.../arm>")
    if args.source == ORACLE_SOURCE and args.truth:
        print("[truth] ignored: --source real IS the oracle", file=sys.stderr)
        args.truth = None
    cfg = apply_paths_args(config_from_args(args), args)
    print(f"[init] roles: {format_role_summary(cfg)}", flush=True)
    plate_center = args.plate_center or cfg.outcome.plate_center

    # ---- real data (numpy only; torch is imported lazily, below) ----
    splits = load_splits(cfg)
    m = load_expr_meta(cfg, plate_center, splits=splits)
    with open(os.path.join(cfg.paths.nuisance_dir, "nuisance_meta.json")) as fh:
        nmeta = json.load(fh)
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(list(META_COLUMNS)).to_pandas())
    n_total = len(meta)
    compound_idx = meta["compound_idx"].values.astype(np.int64)
    dose_level = meta["dose_level"].values.astype(np.float64)
    is_control = meta["is_control"].values.astype(np.int64)
    keys_all = arm_keys(compound_idx, dose_level, is_control)
    # The dose half is computed over the WHOLE table, as the tiered-split builder
    # does, so --min_dose_n dropping an arm cannot shift a compound's halves.
    half_all = dose_half(compound_idx, dose_level, is_control)
    half_of_arm = dict(zip(keys_all, half_all))

    pool_rows = {"all": np.arange(n_total, dtype=np.int64),
                 "train": splits["train_idx"],
                 "holdout": splits["holdout_idx"]}[args.pool]
    pool_rows = np.sort(np.asarray(pool_rows, dtype=np.int64))

    expr = np.load(cfg.paths.expr_npy, mmap_mode="r")
    if expr.shape != (n_total, cfg.outcome.n_genes):
        raise ValueError(f"expr.npy {expr.shape} != table ({n_total}, {cfg.outcome.n_genes})")
    pc_all = plate_codes(meta["det_plate"].values, m["plates"])
    # One pass over every row: the PCA basis is fit on TRAIN rows (E5) even when
    # the pool is the holdout, and 37,340 x 978 float32 is only 146 MB.
    y_all = normalize_expr(np.asarray(expr), pc_all, m)
    # Step C (§3.8.2): the oracle must carry the SAME injection the arms trained
    # on, from the same resolved syn_meta.json, or `--truth` would compare two
    # different ground truths.
    syn = None
    if cfg.outcome.syn_effect != 0:
        syn = load_syn_meta(cfg, n_genes=cfg.outcome.n_genes)
        y_all = inject(y_all, meta["syn_c"].values.astype(np.int64),
                       syn["beta"], syn["v"])
        print(f"[syn] injected beta {syn['beta']:.4f} along v (vec_seed "
              f"{syn['vec_seed']}, sha1 {syn['v_sha1'][:12]}) on "
              f"{int((meta['syn_c'].values.astype(int) == 1).sum()):,} syn_c=1 rows",
              flush=True)
    print(f"[data] table {n_total:,} rows, pool {args.pool} {pool_rows.size:,} rows, "
          f"{cfg.outcome.n_genes} genes, plate_center={plate_center}", flush=True)

    panel_names = {nm for _, names in PANEL for nm in names}
    iname = meta["pert_iname"].astype(str).values
    pid = meta["pert_id"].astype(str).values
    panel_idx: dict[str, int] = {}
    for nm in sorted(panel_names):
        hit = np.flatnonzero(iname == nm)
        if hit.size:
            panel_idx[nm] = int(compound_idx[hit[0]])
    missing = sorted(panel_names - set(panel_idx))
    if missing:
        print(f"[panel] WARNING: not in this table, dropped: {missing}", file=sys.stderr)
    meta_by_arm = {"pert_id": {}, "pert_iname": {}}
    for c in np.unique(compound_idx):
        j = int(np.flatnonzero(compound_idx == c)[0])
        meta_by_arm["pert_id"][int(c)] = str(pid[j])
        meta_by_arm["pert_iname"][int(c)] = str(iname[j])

    this = {"population": cfg.population.name,
            "table_fingerprint": m.get("table_fingerprint"),
            "split_fingerprint": splits.get("split_fingerprint"),
            "gene_order_sha1": hashlib.sha1(
                json.dumps([int(g) for g in m["pr_gene_id"]]).encode()).hexdigest(),
            "plate_center": plate_center,
            "normalize_mean": m.get("normalize_mean"),
            "normalize_std": m.get("normalize_std"),
            "syn_effect": float(cfg.outcome.syn_effect),
            "syn_seed": int(cfg.outcome.syn_seed),
            "syn_beta": (float(syn["beta"]) if syn else 0.0),
            "syn_v_sha1": (str(syn["v_sha1"]) if syn else None),
            "min_dose_n": int(args.min_dose_n),
            "adjustment_set": list(cfg.adjustment_set)}

    # ---- the generator ----
    gen_out, arch, checked, gen_pos = None, None, None, None
    groups_pool, group_names = quality_groups(dose_level[pool_rows], is_control[pool_rows])
    if args.source == "generated":
        import torch  # noqa: E402  (lazy: the oracle path must not need torch)

        from src.eval import generation as gn  # noqa: E402
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda but CUDA is unavailable; this node's "
                               "CUDA environment is broken (see the job script's preflight)")
        model, scheduler, arch, ckpt_dir = gn.load_arm(
            args.run_dir, epoch=args.gen_epoch, which=args.which_wgt, device=dev,
            sampler=args.sampler)
        checked = gn.check_arm_against_data(arch, cfg, splits, m,
                                           n_compounds=int(nmeta["n_compounds"]),
                                           syn=syn)
        if abs(args.guidance_scale - 1.0) > 1e-6 and not gn.supports_cfg_null(model):
            raise RuntimeError("--guidance_scale != 1 needs an arm trained with "
                               "class_dropout_prob > 0")
        print(f"[gen] {os.path.basename(os.path.normpath(args.run_dir))} "
              f"{arch['arch']}-{arch['size']} {arch['diffusion_method']} "
              f"ckpt={os.path.basename(ckpt_dir)} wgt={args.which_wgt} "
              f"sampler={'fm-euler' if arch['diffusion_method'] == 'fm' else args.sampler} "
              f"steps={args.num_inference_steps} w={args.guidance_scale} "
              f"m={args.n_per_row}", flush=True)
        gen_out = gn.generate_for_rows(
            model, scheduler, cfg, meta, pool_rows, n_per_row=args.n_per_row,
            n_inference_steps=args.num_inference_steps, device=dev, seed=args.seed,
            guidance_scale=args.guidance_scale, batch_size=args.gen_batch_size,
            group_of_row=groups_pool, reservoir_cap=args.quality_n)
        gen_pos = np.full(n_total, -1, dtype=np.int64)
        gen_pos[pool_rows] = np.arange(pool_rows.size)

    # ---- sub-pools: one generation pass, both readings ----
    # Which compounds the arm's training split thinned (§3.8.1). Only a tiered
    # arm has them; the oracle itself scores nothing.
    scored_ci: set[int] = set()
    if arch is not None and (arch.get("tier") or {}).get("active"):
        scored_ci = {int(v) for v in (arch["tier"].get("scored_compounds") or {}).values()}
        print(f"[tier] the arm was trained on a thinned split: "
              f"{len(scored_ci):,} scored compounds, confounder "
              f"{arch['tier'].get('confounder')}, gamma {arch['tier'].get('gamma')}",
              flush=True)

    sub: dict[str, np.ndarray] = {args.pool: pool_rows}
    if args.pool == "all":
        sub["holdout"] = np.sort(np.asarray(splits["holdout_idx"], dtype=np.int64))

    pools_json: dict[str, dict] = {}
    npz: dict[str, np.ndarray] = {}
    pca_basis: list = [None]          # fit lazily, once, on real TRAIN Y (E5)
    truth_doc = None
    for pname, rows in sub.items():
        min_n = _min_n_for(pname, args)
        real = arm_table(y_all[rows], keys_all[rows], rows)
        keep = real["n_wells"] >= min_n
        n_drop = int((~keep).sum())
        real = dict(real, arm_key=real["arm_key"][keep], mu=real["mu"][keep],
                    tau=real["tau"][keep], n_wells=real["n_wells"][keep])
        mu0_real = real["mu0"]

        dmso_rows = rows[is_control[rows] == 1]
        floor = noise_floor(y_all[dmso_rows], _floor_sizes(real["n_wells"]),
                            args.n_floor_draws, args.seed)
        # The injection's known shift of this pool's vehicle mean (§3.8.2).
        mu0_offset = None
        if syn is not None:
            p1 = float(meta["syn_c"].values.astype(np.float64)[dmso_rows].mean())
            mu0_offset = syn["beta"] * p1 * syn["v"]
        y_dmso = y_all[dmso_rows]
        mu0_scale = float(np.sqrt((y_dmso.astype(np.float64).var(0, ddof=1)
                                   / max(y_dmso.shape[0], 1)).sum())) if y_dmso.shape[0] > 1 else 0.0
        rel = split_half_reliability(y_all[rows], keys_all[rows], mu0_real, args.seed)
        rel_cos = np.array([rel["per_arm"].get(str(k), np.nan) for k in real["arm_key"]])
        rel_summary = {k: rel[k] for k in ("by_n", "overall")}
        print(f"[pool {pname}] {real['arm_key'].size:,} arms "
              f"({n_drop:,} below min_dose_n {min_n}), "
              f"{real['n_dmso']:,} DMSO, reliability median "
              f"{rel['overall']['median']}", flush=True)

        gen_tbl, quality = None, None
        if gen_out is not None:
            pos = gen_pos[rows]
            if (pos < 0).any():
                raise RuntimeError(f"sub-pool {pname} has rows outside the generation pool")
            g = arm_table(gen_out["row_mean"][pos], keys_all[rows], rows)
            gk = g["n_wells"] >= min_n
            gen_tbl = dict(g, arm_key=g["arm_key"][gk], mu=g["mu"][gk],
                           tau=g["mu"][gk] - (g["mu0"] if args.gen_anchors else mu0_real),
                           n_wells=g["n_wells"][gk])
            gen_tbl["within_row_sd"] = float(np.sqrt(gen_out["row_var"][pos].mean()))
            if not np.array_equal(gen_tbl["arm_key"], real["arm_key"]):
                raise RuntimeError("generated and real arm tables disagree on arms")

        if args.truth:
            truth = _load_truth(args.truth, this, pname, min_n)
            truth_doc = truth
        else:
            truth = None

        if gen_out is not None and args.quality_n:
            if pname != args.pool:
                # The reservoir is drawn over the generation pool, so a sub-pool
                # has no sample of its own, and per-row means are not draws from
                # p(Y | a). QUALITY is reported on the generation pool only.
                quality = {"skipped": "reservoir is drawn over the generation pool"}
            else:
                if pca_basis[0] is None:
                    pca_basis[0] = dm.fit_pca(y_all[splits["train_idx"]], var=args.pca_var)
                    print(f"[pca] k={pca_basis[0]['k']} explains "
                          f"{pca_basis[0]['explained']:.3f} of train variance", flush=True)
                gq, _ = quality_groups(dose_level[rows], is_control[rows])
                quality = _quality_block(y_all[rows], gq, group_names,
                                        gen_out["reservoir"], pca_basis[0], args)
                if args.probe:
                    quality["probe"] = _probe_block(y_all[rows], gq, gen_out["reservoir"],
                                                   group_names, args.seed)

        block, arrs = _pool_block(
            real, gen_tbl, meta_by_arm=meta_by_arm, floor=floor,
            reliability_cos=rel_cos, reliability_summary=rel_summary, args=args,
            panel_idx=panel_idx, mu0_real=mu0_real, mu0_scale=mu0_scale,
            quality=quality, truth=truth, syn=syn, half_of_arm=half_of_arm,
            scored_ci=scored_ci, mu0_offset=mu0_offset)
        block["n_arms_below_min_dose_n"] = n_drop
        block["min_dose_n"] = min_n
        pools_json[pname] = block
        for k, v in arrs.items():
            npz[f"{pname}/{k}"] = v

    # ---- write ----
    out_json, _ = _out_paths(cfg, args, arch, syn_meta=syn)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    tau_npz = out_json[:-5] + "_tau.npz"
    gen_npz = out_json[:-5] + "_gen.npz"
    doc = dict(this)
    doc.update({
        "estimand": "tau_hat(c, d) = mu_hat(c, d) - mu_hat(0) on plate-centred, "
                    "z-scored Level 3 landmark expression (IMPLEMENT.md §3.7)",
        "source": args.source, "pool": args.pool, "seed": args.seed,
        "n_genes": int(cfg.outcome.n_genes), "n_rows_table": int(n_total),
        "n_rows_pool": int(pool_rows.size),
        "responder_mult": float(args.responder_mult),
        "n_floor_draws": int(args.n_floor_draws),
        "panel_missing": missing,
        "generator": None if args.source == ORACLE_SOURCE else {
            "run_dir": os.path.abspath(args.run_dir),
            "run": os.path.basename(os.path.normpath(args.run_dir)),
            "arch": arch["arch"], "size": arch["size"],
            "diffusion_method": arch["diffusion_method"],
            "zero_snr": arch.get("zero_snr"),
            "dr_mode": arch.get("dr_mode"), "cmean_lambda": arch.get("cmean_lambda"),
            "gen_epoch": int(args.gen_epoch), "which_wgt": args.which_wgt,
            "sampler": "fm-euler" if arch["diffusion_method"] == "fm" else args.sampler,
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "n_per_row": int(args.n_per_row), "n_samples": int(gen_out["n_samples"]),
            "gen_anchors": bool(args.gen_anchors),
            "checked_against_data": checked},
        "truth": None if truth_doc is None else {"path": truth_doc["path"],
                                                 "npz": truth_doc["npz"]},
        "syn": None if syn is None else {k: syn[k] for k in
                                        ("syn_effect", "syn_seed", "vec_seed", "scale",
                                         "scale_source", "beta", "v_sha1")},
        "tier_scored_compounds": len(scored_ci) or None,
        "pools": pools_json,
        "artifacts": {"tau_npz": tau_npz,
                      "gen_npz": gen_npz if (gen_out is not None and args.save_gen) else None},
        "elapsed_sec": round(time.time() - t_start, 1),
    })
    _atomic_write(out_json, lambda f: json.dump(doc, f, indent=2, default=_jsonable))
    np.savez_compressed(tau_npz, **npz)
    if gen_out is not None and args.save_gen:
        np.savez(gen_npz, row_id=pool_rows, row_mean=gen_out["row_mean"],
                 row_var=gen_out["row_var"], n_per_row=np.int64(gen_out["n_per_row"]),
                 group_of_row=groups_pool,
                 group_names=np.array(group_names),
                 **{f"reservoir/{g}": v for g, v in gen_out["reservoir"].items()})
    print(f"[out] {out_json}\n[out] {tau_npz}"
          + (f"\n[out] {gen_npz}" if (gen_out is not None and args.save_gen) else ""), flush=True)
    _summarise(doc)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def _summarise(doc: dict) -> None:
    for pname, b in doc["pools"].items():
        print(f"\n=== POOL {pname} ({b['n_arms']:,} arms, {b['n_dmso']:,} DMSO) ===")
        off = b.get("mu0_offset_norm")
        print(f"  ||mu_hat(0)|| real {b['mu0_norm_real']:.3f}"
              + (f" (minus the known injection offset {off:.3f})" if off else "")
              + f" = {b['mu0_norm_over_expected']:.2f}x its DMSO sampling scale "
              f"({b['mu0_norm_expected']:.3f}); ~1x means z = 0 is the vehicle (§3.2)")
        if "mu0_norm_gen" in b:
            print(f"  ||mu_hat(0)|| gen  {b['mu0_norm_gen']:.3f} "
                  f"= {b['mu0_norm_gen_over_expected']:.2f}x that scale")
        print(f"  ||tau|| real   median {b['tau_norm_real']['median']}  "
              f"(noise-corrected {b['tau_norm_real_denoised']['median']})")
        print(f"  ||tau|| est    median {b['tau_norm_est']['median']}  "
              f"(est / corrected = {b['amplitude_ratio_est_over_denoised']['median']})")
        print(f"  floor ratio    median {b['floor_ratio_real']['median']}   "
              f"responder arms {b['n_responder_arms']:,}/{b['n_arms']:,} "
              f"({100 * (b['responder_arm_frac'] or 0):.1f}%)   "
              f"compounds {b['n_responder_compounds']:,}")
        rel = b["reference"]["reliability"]["overall"]
        print(f"  split-half cos (half-sized estimate): median {rel['median']} "
              f"(n={rel['n']}); the bound on cos(gen, oracle) is its "
              f"Spearman-Brown lift, reported per arm below")
        a = b.get("accuracy")
        if a:
            print(f"  ACCURACY vs oracle, {a['n_matched']:,} matched arms")
            print(f"    per-arm cos   all {a['cos_all']['median']}   "
                  f"responder {a['cos_responder']['median']} "
                  f"(n={a['n_responder']:,})")
            print(f"    on the {a['n_responder_with_ceiling']:,} responder arms with a "
                  f"ceiling: cos {a['cos_responder_with_ceiling']['median']} vs the "
                  f"attainable {a['reliability_ceiling_responder']['median']} "
                  f"(split-half {a['split_half_responder']['median']}) "
                  f"-> {a['cos_over_ceiling_responder']['median']} of the ceiling")
            print(f"    ||tau|| spearman {a['tau_norm']['spearman_rho']}   "
                  f"bias {a['tau_norm']['bias']}   truth_sd {a['tau_norm']['truth_sd']}")
            print(f"    pooled gene MSE {a['pooled_gene']['mse']}   "
                  f"bias {a['pooled_gene']['bias']}")
            bv = a.get("bias_along_v")
            if bv:
                print(f"    BIAS ALONG v (signed <tau_gen - tau_oracle, v>), "
                      f"{bv['n_scored_arms']:,} scored / {bv['n_unscored_arms']:,} unscored arms")
                for nm in ("scored", "unscored"):
                    print(f"      {nm:8s} low {bv[nm + '_low']['median']}  "
                          f"high {bv[nm + '_high']['median']}  "
                          f"high-low {bv[nm + '_high_minus_low_median']}")
        q = b.get("quality")
        if q and "pooled" in q:
            for nm, rows in q["pooled"].items():
                vals = "  ".join(f"{r['name']}={r.get('value')}" for r in rows)
                print(f"  QUALITY {nm}: {vals}")
            if "probe" in q:
                print(f"  TRTS probe: {q['probe']}")
    print(f"\n[done] {doc['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
