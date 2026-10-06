"""ADIGen-PO on the step-A data, numpy side (POLICY_LEARNING.md §7-9). No PyTorch.

Contexts are (compound, line); the intervention is the dose; the utility is a
scalar per well. For a context x and dose a:

    mu_h(x, a)      the truth: the holdout well(s) of (x, a), minus the line's holdout vehicle
    mu_hat(x, a)    a generator: the mean of L rollouts (src.policy.rollouts), minus its vehicle
    pi_b(a | x)     the logger: the kept train wells' share of a's dose half, uniform within
                    the half (the thinning's own form)
    pi_beta(a | x)  the tilt  pi_b(a | x) exp(mu_hat(x, a) / beta) / Z(x)      (Eq. 3)
    V(pi)           mean over contexts of sum_a pi(a | x) mu_h(x, a), the compound as the
                    cluster (a compound-jackknife error through decision_task.Est)

Stages (one JSON each, under <runs>/eval_artifacts/policy/):
    gate     L0 from real wells only, before any generator is read
    round1   every rollouts file of the instance: the tilt over B, beta chosen on the
             validation compounds, metrics on the test compounds; the real-data policies
    pilot    for each half j, the pilot policy of its `dr` generator at its chosen beta,
             written as retargeting weights for the OTHER half's train rows at every lambda
    round2   the retargeted generators at the fixed beta, lambda chosen on validation
    report   L0-L7 across the two instances (gamma = 0 and gamma > 0)

    python -m src.policy.learn --data_dir data/core5_24h --tier <tier dir> --stage gate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import _atomic_write, _write_json  # noqa: E402
from src.data.split_halves import N_HALVES, half_dir  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.eval.decision_task import Est, average, fmt_z, gene_axis  # noqa: E402
from src.policy.targets import ROLLOUTS_NAME, decision_targets  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args, line_group_sign)

STAGES = ("gate", "round1", "pilot", "round2", "report")
UTILITIES = ("A", "B")                  # A: the compound's own signature (headline); B: proliferation
HEADLINE = "A"
BETAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
BETA_GATE = 0.2                         # L0's mid-grid temperature
GATE_MIN_THIN_MASS = 1.0 / 3.0
GATE_MIN_SE = 3.0
Z = 2.0
VAL_FRAC = 0.4                          # of the thinned compounds; the rest are the test
GEN_ARMS = ("naive", "conditional", "dr")
RT_ARM = "rt"                           # the retargeted `dr`
REAL_POLICIES = ("kept_pooled", "kept_strat", "unthinned", "logger", "random", "oracle_h")
OUT_SUBDIR = "policy"
WEIGHTS_RT = "dr_weights_retarget_l{lam:g}.npz"
# The dose half the gamma > 0 PL logger keeps 1 well of 6 in, per line group (the
# design of the PL tier, POLICY_LEARNING.md §5). The one definition: the injection
# and the checks import it from here.
THIN_HALF = {"G1": 0, "G2": 1}
RATIO_FLOOR = 1e-6                      # of pi_tilde / pi_b in the retargeting weights


def pkey(arm: str, half: int, extra: str = "") -> str:
    return f"{arm}|h{half}" + (f"|{extra}" if extra else "")


# ---------------------------------------------------------------------------
# the frame: wells, utilities, logger
# ---------------------------------------------------------------------------
def compound_signatures(y: np.ndarray, comp: np.ndarray, ctl: np.ndarray, line_t: np.ndarray,
                        dl: np.ndarray, hold: np.ndarray, n_compounds: int) -> np.ndarray:
    """(n_compounds, G) the unit direction of each compound's holdout effect: the
    mean over its (line, dose) cells that have a holdout well of [the cell's
    holdout mean minus the line's holdout vehicle mean]. Row 0 (the vehicle) is
    zero. This is utility A's axis; the semi-synthetic modifier is planted along
    it (`src.data.inject_modifier`), so the two can never drift."""
    comp = np.asarray(comp, dtype=np.int64); ctl = np.asarray(ctl).astype(bool)
    line_t = np.asarray(line_t, dtype=np.int64); hold = np.asarray(hold).astype(bool)
    L = int(line_t.max()) + 1
    G = y.shape[1]
    hv = np.zeros((L, G)); np.add.at(hv, line_t[hold & ctl], y[hold & ctl].astype(np.float64))
    hv /= np.maximum(np.bincount(line_t[hold & ctl], minlength=L), 1)[:, None]
    tr = hold & ~ctl
    key = np.array([f"{c}|{l}|{d:.6g}" for c, l, d in zip(comp[tr], line_t[tr], np.asarray(dl, dtype=np.float64)[tr])])
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    s = np.zeros((uniq.size, G)); np.add.at(s, inv, y[tr].astype(np.float64))
    n = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    cell_comp, cell_line = comp[tr][first], line_t[tr][first]
    eff = s / n[:, None] - hv[cell_line]
    V = np.zeros((n_compounds, G))
    for c in np.unique(cell_comp):
        m = eff[cell_comp == c].mean(axis=0)
        nn = float(np.sqrt((m ** 2).sum()))
        if not np.isfinite(nn) or nn == 0:
            raise RuntimeError(f"compound {c}: no holdout direction")
        V[c] = m / nn
    return V


def injection_block(cfg) -> dict | None:
    """The build's `injection` record (population_qc.json), or None on a real build."""
    with open(cfg.paths.population_qc_json) as fh:
        return json.load(fh).get("injection")


def load_frame(cfg, tier: str, val_seed: int = 0, signature_file: str | None = None) -> dict:
    """Everything the policies rest on that no generator enters. `signature_file`
    (an (n_compounds, G) .npy keyed by compound_idx) replaces the holdout-derived
    utility axis; on an injected build it defaults to the build's own."""
    from datasets import load_from_disk

    from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes
    from src.data.splits import dose_half

    t0 = time.time()
    base = load_splits(cfg)                                   # the base split: holdout and unthinned train
    tz = os.path.normpath(tier)
    dd = os.path.normpath(os.path.abspath(cfg.paths.data_dir))
    if not os.path.abspath(tz).startswith(dd + os.sep):
        raise SystemExit(f"[policy] the tier {tz} is not under the build {dd}: a stage would read "
                         f"another build's expression")
    inj = injection_block(cfg)
    if signature_file is None and inj is not None:
        signature_file = os.path.join(dd, inj["signature_file"])
    with open(os.path.join(tz, "splits.json")) as fh:
        ts = json.load(fh)
    if not ts["tier"].get("active") or ts["tier"].get("half"):
        raise SystemExit(f"[policy] {tz} is not a (parent) thinning instance")
    if sorted(ts["holdout_idx"]) != sorted(base["holdout_idx"].tolist()):
        raise SystemExit(f"[policy] {tz} does not share the base split's holdout")
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "log10_conc", "is_control", "cell_id", "det_plate",
                             "pert_iname"]).to_pandas())
    N = len(meta)
    comp = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool)
    line = meta["cell_id"].values.astype(str)
    T = decision_targets(meta)
    lines = sorted(set(line.tolist()))
    L = len(lines)
    ctx_line = np.array([lines.index(l) for l in T["ctx_line"]], dtype=np.int64)
    n_ctx = T["ctx_compound"].size
    # (context, dose) layout: the compound's doses ascending, padded to D
    nd = np.bincount(T["t_ctx"], minlength=n_ctx)
    D = int(nd.max())
    valid = np.arange(D)[None, :] < nd[:, None]
    dose_level = np.full((n_ctx, D), np.nan)
    dose_level[T["t_ctx"], T["t_dose_index"]] = T["t_dose_level"]
    # the dose half, as splits.dose_half defines it: the top n - n // 2 of the
    # compound's distinct levels are high, so it is the dose's rank alone
    half = np.where(valid, (np.arange(D)[None, :] >= (nd // 2)[:, None]).astype(np.int8), -1).astype(np.int8)
    half_t = dose_half(comp, dl, ctl)
    # every row of (context, dose) -> target index
    key_of_target = {(int(c), int(l), float(d)): i for i, (c, l, d) in
                     enumerate(zip(T["ctx_compound"][T["t_ctx"]], ctx_line[T["t_ctx"]], T["t_dose_level"]))}
    row_t = np.full(N, -1, dtype=np.int64)
    for i in np.flatnonzero(~ctl):
        row_t[i] = key_of_target[(int(comp[i]), lines.index(line[i]), float(dl[i]))]
    t_ctx, t_di = T["t_ctx"], T["t_dose_index"]
    if not np.array_equal(half[t_ctx[row_t[~ctl]], t_di[row_t[~ctl]]], half_t[~ctl]):
        raise RuntimeError("the (context, dose) halves disagree with splits.dose_half on the table's rows")
    g2 = line_group_sign(np.array(lines), cfg.population) > 0
    thin_half = np.where(g2[ctx_line], THIN_HALF["G2"], THIN_HALF["G1"]).astype(np.int8)
    expr_meta = load_expr_meta(cfg, splits=base)
    y = normalize_expr(np.asarray(np.load(cfg.paths.expr_npy, mmap_mode="r")),
                       plate_codes(meta["det_plate"].values, expr_meta["plates"]), expr_meta)
    G = y.shape[1]
    line_t = np.array([lines.index(l) for l in line], dtype=np.int64)

    def mask(idx):
        m = np.zeros(N, dtype=bool)
        m[np.asarray(idx, dtype=np.int64)] = True
        return m
    sets = {"holdout": mask(base["holdout_idx"]), "unthinned": mask(base["train_idx"]), "kept": mask(ts["train_idx"])}
    for j in range(1, N_HALVES + 1):
        hd = half_dir(tz, j)
        if os.path.isfile(os.path.join(hd, "splits.json")):
            with open(os.path.join(hd, "splits.json")) as fh:
                sets[f"kept_h{j}"] = mask(json.load(fh)["train_idx"])
    if (sets["kept"] & ~sets["unthinned"]).any() or (sets["kept"] & sets["holdout"]).any():
        raise SystemExit("[policy] the tier's kept rows are not a subset of the base train rows")

    # ---- cell counts, holdout cell means -> the signature axis per compound --------------
    def cell_counts(m):
        c = np.zeros((n_ctx, D), dtype=np.int64)
        np.add.at(c, (t_ctx[row_t[m & ~ctl]], t_di[row_t[m & ~ctl]]), 1)
        return c
    cnt = {k: cell_counts(m) for k, m in sets.items()}
    veh_cnt = {k: np.bincount(line_t[m & ctl], minlength=L) for k, m in sets.items()}
    if signature_file is None:
        V = compound_signatures(y, comp, ctl, line_t, dl, sets["holdout"], int(comp.max()) + 1)
    else:
        V = np.load(signature_file, allow_pickle=False).astype(np.float64)
        if V.ndim != 2 or V.shape[1] != G or V.shape[0] <= int(comp.max()):
            raise SystemExit(f"[policy] {signature_file}: shape {V.shape} does not cover the table's compounds")
        nrm = np.sqrt((V[np.unique(T["ctx_compound"])] ** 2).sum(axis=1))
        if not np.allclose(nrm, 1.0, atol=1e-5):
            raise SystemExit(f"[policy] {signature_file}: not unit vectors on the table's compounds")
    S_ctx = V[T["ctx_compound"]]                                           # (n_ctx, G)
    s_b = gene_axis([str(s) for s in expr_meta_symbols(cfg)])

    # ---- per-well utilities, then every well set's cell means are scalar sums -----------
    u = {"A": np.einsum("ng,ng->n", y, S_ctx[np.where(ctl, 0, t_ctx[np.maximum(row_t, 0)])]).astype(np.float64),
         "B": np.einsum("ng,g->n", y, s_b).astype(np.float64)}
    # a vehicle well has no compound: its A-utility is taken against the context it is
    # compared with, so the vehicle mean is kept per (line) as a vector and projected later
    mu = {}                                                            # mu[set][utility]: (n_ctx, D), NaN where empty
    for k, m in sets.items():
        tr = m & ~ctl
        vm = np.zeros((L, G)); np.add.at(vm, line_t[m & ctl], y[m & ctl].astype(np.float64))
        vm /= np.maximum(veh_cnt[k], 1)[:, None]
        mu[k] = {}
        for ut in UTILITIES:
            s = np.zeros((n_ctx, D)); np.add.at(s, (t_ctx[row_t[tr]], t_di[row_t[tr]]), u[ut][tr])
            with np.errstate(invalid="ignore", divide="ignore"):
                cell = s / cnt[k]
            veh = (np.einsum("cg,cg->c", vm[ctx_line], S_ctx) if ut == "A" else vm[ctx_line] @ s_b)
            mu[k][ut] = cell - veh[:, None]
    # the pooled kept mean: lines ignored (what a line-blind estimator sees)
    pooled = {}
    for k in ("kept",):
        tr = sets[k] & ~ctl
        vm_all = y[sets[k] & ctl].astype(np.float64).mean(axis=0)
        pooled[k] = {}
        for ut in UTILITIES:
            s = np.zeros((n_ctx, D)); n = np.zeros((n_ctx, D))
            # pool over the lines of the compound: sum per (compound, dose) then broadcast
            sc = {}; nc = {}
            for i in np.flatnonzero(tr):
                kk = (int(comp[i]), float(dl[i]))
                sc[kk] = sc.get(kk, 0.0) + u[ut][i]; nc[kk] = nc.get(kk, 0) + 1
            for ci in range(n_ctx):
                for di in range(nd[ci]):
                    kk = (int(T["ctx_compound"][ci]), float(dose_level[ci, di]))
                    s[ci, di], n[ci, di] = sc.get(kk, np.nan), nc.get(kk, 0)
            veh = np.einsum("cg,g->c", S_ctx, vm_all) if ut == "A" else np.full(n_ctx, float(vm_all @ s_b))
            with np.errstate(invalid="ignore", divide="ignore"):
                pooled[k][ut] = s / n - veh[:, None]

    # ---- the logger from the kept counts, at the dose half --------------------------------
    pi_b = logger_from_counts(cnt["kept"], half, valid)
    # the design's own logger (tier_meta: each train well's realised inclusion probability)
    with open(os.path.join(tz, "tier_meta.json")) as fh:
        tm = json.load(fh)
    incl = np.ones(N); 
    for a in tm["compounds"]:
        incl[np.asarray(a["rows"], dtype=np.int64)] = np.asarray(a["incl"], dtype=np.float64)
    um = sets["unthinned"] & ~ctl
    w_or = np.zeros((n_ctx, D)); np.add.at(w_or, (t_ctx[row_t[um]], t_di[row_t[um]]), incl[um])
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_oracle = np.where(valid, w_or / w_or.sum(axis=1, keepdims=True), 0.0)

    # ---- eligibility, the thinned set, the validation / test split --------------------------
    scored = np.array(sorted(int(v) for v in ts["tier"]["scored_compounds"].values()), dtype=np.int64)
    thinned = np.isin(T["ctx_compound"], scored)
    eligible = (nd >= 2) & ((cnt["holdout"] > 0) | ~valid).all(axis=1)
    rng = np.random.default_rng([int(val_seed), 9_091])
    sc_perm = rng.permutation(scored)
    n_val = int(round(VAL_FRAC * scored.size))
    val_comp = np.sort(sc_perm[:n_val]); test_comp = np.sort(sc_perm[n_val:])
    role = np.where(np.isin(T["ctx_compound"], val_comp), "val",
                    np.where(np.isin(T["ctx_compound"], test_comp), "test", "control"))
    print(f"[policy] frame: {n_ctx:,} contexts ({L} lines, up to {D} doses), {int(eligible.sum()):,} with a holdout "
          f"well at every dose; thinned {int((thinned & eligible).sum()):,} (val {int(((role == 'val') & eligible).sum()):,}, "
          f"test {int(((role == 'test') & eligible).sum()):,}), control {int((~thinned & eligible).sum()):,}; "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"T": T, "n_ctx": n_ctx, "D": D, "nd": nd, "valid": valid, "dose_level": dose_level, "half": half,
            "thin_half": thin_half, "ctx_line": ctx_line, "lines": lines, "g2": g2, "comp": T["ctx_compound"],
            "name_of": dict(zip(comp.tolist(), meta["pert_iname"].astype(str).tolist())),
            "cnt": cnt, "mu": mu, "pooled": pooled, "pi_b": pi_b, "pi_oracle": pi_oracle, "S_ctx": S_ctx, "s_b": s_b,
            "thinned": thinned, "eligible": eligible, "role": role, "val_comp": val_comp, "test_comp": test_comp,
            "tier": tz, "tier_fingerprint": ts["split_fingerprint"], "gamma": float(ts["tier"]["gamma"]),
            "halves": {j: half_dir(tz, j) for j in range(1, N_HALVES + 1) if f"kept_h{j}" in sets},
            "injection": inj, "signature_file": signature_file,
            "signs": (np.load(os.path.join(dd, inj["signs_file"]), allow_pickle=False) if inj else None)}


def expr_meta_symbols(cfg) -> list:
    with open(cfg.paths.gene_order_json) as fh:
        return [str(s) for s in json.load(fh)["pr_gene_symbol"]]


def logger_from_counts(cnt: np.ndarray, half: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(n_ctx, D) pi_b(a | x): each half's share of the context's kept wells, spread
    uniformly over the half's doses. A context with no kept well is uniform."""
    n_ctx, D = cnt.shape
    out = np.zeros((n_ctx, D))
    for h in (0, 1):
        m = valid & (half == h)
        n_h = (cnt * m).sum(axis=1).astype(np.float64)
        d_h = m.sum(axis=1)
        tot = (cnt * valid).sum(axis=1).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(tot > 0, n_h / tot, d_h / np.maximum(valid.sum(axis=1), 1))
            out += np.where(m, (share / np.maximum(d_h, 1))[:, None], 0.0)
    out = np.where(valid, out, 0.0)
    s = out.sum(axis=1, keepdims=True)
    return np.where(s > 0, out / np.where(s > 0, s, 1), np.where(valid, 1.0 / np.maximum(valid.sum(axis=1), 1)[:, None], 0.0))


# ---------------------------------------------------------------------------
# tilt, value, metrics
# ---------------------------------------------------------------------------
def tilt(mu: np.ndarray, pi_b: np.ndarray, beta: float, valid: np.ndarray) -> np.ndarray:
    """(n_ctx, D) pi_b exp(mu / beta) / Z, over the valid doses; NaN mu rows give NaN."""
    m = np.where(valid, mu, -np.inf)
    z = (m - np.nanmax(np.where(valid, mu, np.nan), axis=1, keepdims=True)) / beta
    w = np.where(valid, pi_b * np.exp(z), 0.0)
    s = w.sum(axis=1, keepdims=True)
    return np.where(s > 0, w / np.where(s > 0, s, 1), np.nan)


def value_per_context(pi: np.ndarray, mu_h: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, pi * mu_h, 0.0).sum(axis=1)


def by_compound(v: np.ndarray, comp: np.ndarray, sel: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """(n_comps,) the mean of v over the selected contexts of each compound, in the
    order `comps`; NaN for a compound with no selected context."""
    out = np.full(comps.size, np.nan)
    pos = {int(c): i for i, c in enumerate(comps)}
    s = np.zeros(comps.size); n = np.zeros(comps.size)
    for i in np.flatnonzero(sel):
        j = pos.get(int(comp[i]))
        if j is not None and np.isfinite(v[i]):
            s[j] += v[i]; n[j] += 1
    out[n > 0] = s[n > 0] / n[n > 0]
    return out


def est(v: np.ndarray, comp: np.ndarray, sel: np.ndarray, comps: np.ndarray) -> Est:
    """A compound-clustered statistic (jackknife) of the context values v on sel."""
    pc = by_compound(v, comp, sel, comps)
    ok = np.isfinite(pc)
    if ok.sum() < 2:
        raise ValueError("fewer than two compounds with a value")
    # fixed order over comps: missing compounds are dropped consistently by the caller's sel
    return Est.mean(pc[ok]) if ok.all() else Est(pc[ok].mean(), np.full(comps.size, np.nan), "jk")


class Sets:
    """The context sets a policy is scored on and the compound order of each."""

    def __init__(self, F: dict):
        el, th, role = F["eligible"], F["thinned"], F["role"]
        self.sel = {"test": el & th & (role == "test"), "val": el & th & (role == "val"),
                    "thinned": el & th, "control": el & ~th}
        self.comps = {k: np.unique(F["comp"][m]) for k, m in self.sel.items()}
        self.comp = F["comp"]

    def est(self, v: np.ndarray, which: str) -> Est:
        return est(v, self.comp, self.sel[which], self.comps[which])


def policy_metrics(pi: np.ndarray, mu_hat: np.ndarray | None, F: dict, sets: Sets, ut: str, which: str) -> dict:
    """Value and diagnostics of one policy on one context set."""
    mu_h = F["mu"]["holdout"][ut]
    v = value_per_context(pi, mu_h, F["valid"])
    out = {"value": sets.est(v, which).doc()}
    thin = (F["half"] == F["thin_half"][:, None]) & F["valid"]
    out["mass_on_thin_half"] = float(np.where(thin, pi, 0.0).sum(axis=1)[sets.sel[which]].mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(F["valid"] & (F["pi_b"] > 0), pi / F["pi_b"], np.nan)
    r = ratio[sets.sel[which]]
    out["transfer_factor"] = {"max": float(np.nanmax(r)), "p99": float(np.nanpercentile(r, 99)),
                              "mean_max_per_context": float(np.nanmean(np.nanmax(r, axis=1)))}
    if mu_hat is not None:
        vm = value_per_context(pi, mu_hat, F["valid"])
        out["self_evaluation_error"] = sets.est(vm - v, which).doc()          # optimism: model minus truth
        err = np.where(F["valid"], (mu_hat - mu_h) ** 2, 0.0)
        out["model_error_at_policy"] = sets.est((pi * err).sum(axis=1), which).doc()
        out["model_error_global"] = sets.est(err.sum(axis=1) / np.maximum(F["nd"], 1), which).doc()
    return out


def select_beta_with_logger(mu_hat: np.ndarray, lg: np.ndarray, F: dict, sets: Sets, ut: str,
                            fixed: float | None = None, betas=BETAS):
    """The temperature with the largest validation value of the tilt of `lg` by
    mu_hat (or `fixed`), and the validation curve."""
    mu_h = F["mu"]["holdout"][ut]
    curve = {}
    for b in betas:
        pi = tilt(mu_hat, lg, b, F["valid"])
        curve[f"{b:g}"] = sets.est(value_per_context(pi, mu_h, F["valid"]), "val").doc()
    if fixed is not None:
        return float(fixed), curve
    return float(max(curve, key=lambda k: curve[k]["value"])), curve


def select_beta(mu_hat: np.ndarray, F: dict, sets: Sets, ut: str, betas=BETAS, fixed: float | None = None):
    """`select_beta_with_logger` over the estimated logger."""
    return select_beta_with_logger(mu_hat, F["pi_b"], F, sets, ut, fixed=fixed, betas=betas)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------
def mu_from_rollouts(path: str, F: dict) -> tuple[dict, dict]:
    """mu_hat per utility (n_ctx, D) from one rollouts file, aligned to the frame."""
    z = np.load(path, allow_pickle=False)
    T = F["T"]
    if (not np.array_equal(z["ctx_compound"], T["ctx_compound"]) or not np.array_equal(z["ctx_line"], T["ctx_line"])
            or not np.array_equal(z["t_ctx"], T["t_ctx"]) or not np.array_equal(z["t_dose_index"], T["t_dose_index"])):
        raise SystemExit(f"[policy] {path}: its targets differ from the frame's")
    lines = [str(l) for l in z["lines"]]
    if lines != F["lines"]:
        raise SystemExit(f"[policy] {path}: lines {lines} differ from the frame's")
    mean = z["mean"].astype(np.float64)                                   # (n_t, G)
    veh = z["vehicle_mean"].astype(np.float64)[F["ctx_line"]]             # (n_ctx, G)
    out = {}
    for ut in UTILITIES:
        if ut == "A":
            u_t = np.einsum("tg,tg->t", mean, F["S_ctx"][T["t_ctx"]])
            u_v = np.einsum("cg,cg->c", veh, F["S_ctx"])
        else:
            u_t = mean @ F["s_b"]; u_v = veh @ F["s_b"]
        m = np.full((F["n_ctx"], F["D"]), np.nan)
        m[T["t_ctx"], T["t_dose_index"]] = u_t - u_v[T["t_ctx"]]
        out[ut] = m
    return out, json.loads(str(z["label"]))


def planted_share(F: dict, mu_inj: np.ndarray, uninjected_path: str, label: dict) -> dict:
    """How much of the planted modifier a generator reproduces: the mean over the
    THIN-half cells of m[c, g] (mu_hat_injected - mu_hat_uninjected) / beta, and the
    same over the favoured half (expected 0). The uninjected run has the same rows,
    seeds and noise seeds, so the difference isolates the modifier's effect on the
    generator (plus run-to-run noise). Per thinned / control compounds."""
    inj = F["injection"]
    mu_un, lab_un = mu_from_rollouts(uninjected_path, F)
    mu_un = mu_un["A"]
    for k in ("L", "num_inference_steps", "seed", "gen_epoch"):
        if lab_un.get(k) != label.get(k):
            raise SystemExit(f"[policy] {uninjected_path}: {k} {lab_un.get(k)} differs from the injected run's {label.get(k)}")
    g2 = F["g2"][F["ctx_line"]]
    m = F["signs"][F["comp"], np.where(g2, 1, 0)]                          # (n_ctx,) the sign of each context's cell
    d = (mu_inj - mu_un) * m[:, None] / float(inj["beta"])
    thin = (F["half"] == F["thin_half"][:, None]) & F["valid"]
    fav = F["valid"] & ~thin
    out = {"uninjected_run": uninjected_path}
    for sname, sel in (("thinned", F["thinned"] & F["eligible"]), ("control", ~F["thinned"] & F["eligible"])):
        comps = np.unique(F["comp"][sel])
        th = np.where(thin, d, 0).sum(axis=1) / np.maximum(thin.sum(axis=1), 1)
        fv = np.where(fav, d, 0).sum(axis=1) / np.maximum(fav.sum(axis=1), 1)
        out[sname] = {"thin_half": est(th, F["comp"], sel, comps).doc(), "favoured_half": est(fv, F["comp"], sel, comps).doc()}
    return out


def find_rollouts(runs_dir: str, pattern: str) -> dict[str, str]:
    """{run name: rollouts path} for run dirs matching the glob pattern."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, pattern))):
        p = os.path.join(d, "eval_artifacts", ROLLOUTS_NAME)
        if os.path.isfile(p):
            out[os.path.basename(d)] = p
    return out


def parse_run(name: str, prefix: str) -> dict | None:
    """pl_<arm>_g<gamma>_h<half>[_l<lambda>]  ->  fields."""
    if not name.startswith(prefix + "_"):
        return None
    parts = name[len(prefix) + 1:].split("_")
    try:
        arm = parts[0]
        g = float(parts[1][1:]); h = int(parts[2][1:])
        lam = float(parts[3][1:]) if len(parts) > 3 else None
    except (IndexError, ValueError):
        return None
    return {"arm": arm, "gamma": g, "half": h, "lam": lam}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def real_policies(F: dict, sets: Sets, ut: str) -> dict:
    """The non-generative anchors, each at its validation-chosen beta where it has one."""
    mu_h = F["mu"]["holdout"][ut]
    uniform = np.where(F["valid"], 1.0 / np.maximum(F["nd"], 1)[:, None], 0.0)
    out = {}
    for name in REAL_POLICIES:
        if name == "logger":
            pi, beta, curve, mu_hat = F["pi_b"], None, None, None
        elif name == "random":
            pi, beta, curve, mu_hat = uniform, None, None, None
        elif name == "oracle_h":
            # the tilt of the truth itself (uniform logger): optimistic by the holdout noise
            beta, curve = select_beta_with_logger(mu_h, uniform, F, sets, ut)
            pi = tilt(mu_h, uniform, beta, F["valid"]); mu_hat = None
        else:
            mu_hat = {"kept_pooled": F["pooled"]["kept"][ut], "kept_strat": F["mu"]["kept"][ut],
                      "unthinned": F["mu"]["unthinned"][ut]}[name]
            lg = uniform if name == "unthinned" else F["pi_b"]
            # a context without a kept well at some dose has no mean there: it is left to
            # the logger's share (mu = the context's mean over its other doses)
            mu_hat = np.where(np.isfinite(mu_hat), mu_hat, np.nanmean(mu_hat, axis=1, keepdims=True))
            beta, curve = select_beta_with_logger(mu_hat, lg, F, sets, ut)
            pi = tilt(mu_hat, lg, beta, F["valid"])
        rec = {"beta": beta, "val_curve": curve}
        for which in ("test", "control", "val"):
            rec[which] = policy_metrics(pi, None if name in ("logger", "random", "oracle_h") else mu_hat,
                                        F, sets, ut, which)
        rec["_v"] = value_per_context(pi, mu_h, F["valid"])
        out[name] = rec
    return out


def gains(block: dict, sets: Sets, comp, which: str) -> None:
    """Add gain over random and captured share (vs `unthinned`) to every policy of a
    stage block, paired by compound."""
    rnd, ref = block["real"]["random"]["_v"], block["real"]["unthinned"]["_v"]
    e_r, e_ref = sets.est(rnd, which), sets.est(ref, which)
    span = e_ref - e_r
    for grp in ("real", "generators"):
        for name, rec in block.get(grp, {}).items():
            e = sets.est(rec["_v"], which)
            rec[which]["gain_over_random"] = (e - e_r).doc()
            stable = span.z is not None and span.z >= 3 and span.value > 0
            rec[which]["captured_share"] = ((e - e_r) / span).doc() if stable else {
                "value": (e.value - e_r.value) / span.value if span.value else None, "se": None, "unstable_denominator": True}


def stage_gate(F: dict, sets: Sets) -> dict:
    """L0, from real wells only."""
    out = {"beta": BETA_GATE, "utilities": {}}
    uniform = np.where(F["valid"], 1.0 / np.maximum(F["nd"], 1)[:, None], 0.0)
    thin = (F["half"] == F["thin_half"][:, None]) & F["valid"]
    for ut in UTILITIES:
        mu_h = F["mu"]["holdout"][ut]
        ref = F["mu"]["unthinned"][ut]
        ref = np.where(np.isfinite(ref), ref, np.nanmean(ref, axis=1, keepdims=True))
        pi_ref = tilt(ref, uniform, BETA_GATE, F["valid"])
        pooled = F["pooled"]["kept"][ut]
        pooled = np.where(np.isfinite(pooled), pooled, np.nanmean(pooled, axis=1, keepdims=True))
        pi_pool = tilt(pooled, F["pi_b"], BETA_GATE, F["valid"])
        sel = sets.sel["thinned"]
        mass = float(np.where(thin, pi_ref, 0.0).sum(axis=1)[sel].mean())
        loss = sets.est(value_per_context(pi_ref, mu_h, F["valid"]), "thinned") - \
            sets.est(value_per_context(pi_pool, mu_h, F["valid"]), "thinned")
        strat = F["mu"]["kept"][ut]
        strat = np.where(np.isfinite(strat), strat, np.nanmean(strat, axis=1, keepdims=True))
        pi_strat = tilt(strat, F["pi_b"], BETA_GATE, F["valid"])
        # (c) the thin wells carry usable signal: the per-line memoriser of the kept
        # wells beats the line-blind one (POLICY_LEARNING.md §14)
        signal = sets.est(value_per_context(pi_strat, mu_h, F["valid"]), "thinned") - \
            sets.est(value_per_context(pi_pool, mu_h, F["valid"]), "thinned")
        rec = {"thin_half_mass_of_reference_tilt": mass, "needs_mass_at_least": GATE_MIN_THIN_MASS,
               "reference_minus_kept_pooled": {**loss.doc(), "z": loss.z}, "needs_z_at_least": GATE_MIN_SE,
               "kept_strat_minus_kept_pooled": {**signal.doc(), "z": signal.z},
               "n_thinned_contexts": int(sel.sum()),
               "kept_wells_on_thin_half_share": float(
                   (np.where(thin, F["cnt"]["kept"], 0).sum(axis=1) / np.maximum(F["cnt"]["kept"].sum(axis=1), 1))[sel].mean())}
        rec["pass_a"] = bool(mass >= GATE_MIN_THIN_MASS)
        rec["pass_b"] = bool(loss.z is not None and loss.z >= GATE_MIN_SE)
        rec["pass_c"] = bool(signal.z is not None and signal.z >= GATE_MIN_SE)
        # (c) is part of the gate on an injected build only (it was declared with the modifier)
        rec["pass"] = bool(rec["pass_a"] and rec["pass_b"] and (rec["pass_c"] or F["injection"] is None))
        out["utilities"][ut] = rec
    out["pass"] = out["utilities"][HEADLINE]["pass"]
    return out


def stage_round(F: dict, sets: Sets, rollouts: dict[str, str], prefix: str, fixed_beta: dict | None,
                uninjected: dict[str, str] | None = None) -> dict:
    """round1 (fixed_beta None: beta chosen on validation) or round2 (beta fixed per half).
    `uninjected` maps run names to the rollouts of the same runs on the real build
    (the planted-share diagnostic of an injected build)."""
    out = {"injection": F["injection"]}
    if F["injection"] is not None and uninjected is not None and not uninjected:
        out["planted_share_skipped"] = "no uninjected counterparts were found; see --uninjected_runs_dir"
        print("[policy] WARNING: injected build but no uninjected rollouts of the same runs were found; "
              "the planted-share diagnostic is skipped", flush=True)
    for ut in UTILITIES:
        blk = {"real": real_policies(F, sets, ut), "generators": {}}
        for run, path in sorted(rollouts.items()):
            info = parse_run(run, prefix)
            if info is None or abs(info["gamma"] - F["gamma"]) > 1e-9:
                continue
            mu_all, label = mu_from_rollouts(path, F)
            mu_hat = mu_all[ut]
            if not np.isfinite(mu_hat[F["valid"]]).all():
                raise SystemExit(f"[policy] {path}: a decision has no finite mu_hat")
            fb = None if fixed_beta is None else fixed_beta[str(info["half"])]
            beta, curve = select_beta(mu_hat, F, sets, ut, fixed=fb)
            pi = tilt(mu_hat, F["pi_b"], beta, F["valid"])
            rec = {"run": run, "arm": info["arm"], "half": info["half"], "lam": info["lam"], "beta": beta,
                   "beta_fixed": fb is not None, "val_curve": curve, "label": {k: label.get(k) for k in
                   ("dr_mode", "dr_weights_file", "dr_weight_norm", "nuisance_dir", "L", "num_inference_steps")}}
            for which in ("test", "control", "val"):
                rec[which] = policy_metrics(pi, mu_hat, F, sets, ut, which)
            rec["_v"] = value_per_context(pi, F["mu"]["holdout"][ut], F["valid"])
            rec["_mu"] = mu_hat
            if ut == HEADLINE and F["injection"] is not None and uninjected:
                if run not in uninjected:
                    raise SystemExit(f"[policy] no uninjected rollouts for {run} (pass --uninjected_runs_dir, or '' to skip)")
                rec["planted_share"] = planted_share(F, mu_all["A"], uninjected[run], label)
            blk["generators"][run] = rec
        for which in ("test", "control"):
            gains(blk, sets, F["comp"], which)
        out[ut] = blk
    return out


def strip(doc):
    """Drop the per-context arrays before writing JSON."""
    if isinstance(doc, dict):
        return {k: strip(v) for k, v in doc.items() if not k.startswith("_")}
    if isinstance(doc, list):
        return [strip(v) for v in doc]
    if isinstance(doc, (np.floating, np.integer)):
        return doc.item()
    return doc


def stage_pilot(F: dict, r1: dict, rollouts: dict[str, str], lambdas=LAMBDAS) -> dict:
    """Retargeting weights (POLICY_LEARNING.md §8): for half j, the pilot policy is its
    `dr` generator's tilt at the beta round 1 chose; the weights go on the OTHER half's
    rows: w_i = lam * exp(mu_hat(x_i, a_i) / beta) / Z(x_i) + (1 - lam), vehicles 1.
    `r1` is the round-1 stage's record (the betas); mu_hat is read from the rollouts."""
    from datasets import load_from_disk
    ut = HEADLINE
    gens = r1[ut]["generators"]
    meta = (load_from_disk(F["cfg"].paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "cell_id"]).to_pandas())
    comp = meta["compound_idx"].values.astype(np.int64); dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool); line = meta["cell_id"].values.astype(str)
    key = {(int(c), int(l), float(d)): (ci, di) for ci, di, c, l, d in zip(
        F["T"]["t_ctx"], F["T"]["t_dose_index"], F["comp"][F["T"]["t_ctx"]], F["ctx_line"][F["T"]["t_ctx"]],
        F["T"]["t_dose_level"])}
    out = {"utility": ut, "lambdas": list(lambdas), "halves": {}}
    for j, hd in F["halves"].items():
        other = [k for k in F["halves"] if k != j]
        if len(other) != 1:
            raise SystemExit("[policy] the pilot needs exactly two halves")
        src = [r for r in gens.values() if r["arm"] == "dr" and r["half"] == other[0] and r["lam"] is None]
        if len(src) != 1:
            raise SystemExit(f"[policy] need exactly one `dr` round-1 run of half {other[0]}, found {len(src)}")
        src = src[0]
        beta = float(src["beta"])
        mu_hat = mu_from_rollouts(rollouts[src["run"]], F)[0][ut]
        z = (mu_hat - np.nanmax(np.where(F["valid"], mu_hat, np.nan), axis=1, keepdims=True)) / beta
        ratio = np.where(F["valid"], np.exp(z), 0.0)
        ratio = ratio / np.where(F["valid"], F["pi_b"] * ratio, 0.0).sum(axis=1, keepdims=True)   # pi_tilde / pi_b
        # a row far below the pilot's preferred dose at a small beta has a ratio below
        # float32's range; the floor keeps every context's weights positive (a group of
        # zeros would stop the trainer's context normalisation)
        ratio = np.maximum(ratio, RATIO_FLOOR)
        with open(os.path.join(hd, "splits.json")) as fh:
            hs = json.load(fh)
        tr = np.asarray(hs["train_idx"], dtype=np.int64)
        base_ratio = np.ones(tr.size)
        for n, i in enumerate(tr):
            if not ctl[i]:
                ci, di = key[(int(comp[i]), F["lines"].index(line[i]), float(dl[i]))]
                base_ratio[n] = ratio[ci, di]
        files = {}
        n_floor = int((base_ratio[~ctl[tr]] <= RATIO_FLOOR).sum())
        for lam in lambdas:
            w = np.maximum(lam * base_ratio + (1.0 - lam), RATIO_FLOOR).astype(np.float32)
            w[ctl[tr]] = 1.0
            fn = WEIGHTS_RT.format(lam=lam)
            _atomic_write(os.path.join(hd, fn), lambda f, w=w: np.savez(
                f, row_id=tr, w=w, mode=np.array("retarget"), lam=np.array(lam), beta=np.array(beta),
                pilot_run=np.array(src["run"]), split_fingerprint=np.array(hs["split_fingerprint"])), mode="wb")
            files[f"{lam:g}"] = {"file": fn, "ess_over_n": float(w.sum() ** 2 / (w ** 2).sum() / w.size),
                                 "max": float(w.max()), "treated_mean": float(w[~ctl[tr]].mean()),
                                 "n_rows_at_floor": n_floor}
        out["halves"][str(j)] = {"dir": hd, "pilot_run": src["run"], "pilot_half": other[0], "beta": beta,
                                 "weights": files, "n_rows": int(tr.size)}
        print(f"[policy] half {j}: pilot from {src['run']} at beta {beta:g}; weights "
              + ", ".join(f"lambda {k}: ESS/n {v['ess_over_n']:.3f} max {v['max']:.2f}" for k, v in files.items()), flush=True)
    return out


def stage_round2(F: dict, sets: Sets, rollouts: dict, prefix: str, pilot: dict,
                 uninjected: dict[str, str] | None = None) -> dict:
    fixed = {h: float(v["beta"]) for h, v in pilot["halves"].items()}
    r2 = stage_round(F, sets, rollouts, prefix, fixed_beta=fixed, uninjected=uninjected)
    # L7's quantity: the retargeted policy's transfer factor against its TRAINING law
    # pi_lambda = lam * pi_tilde + (1 - lam) * pi_b (bounded by 1 / lam), next to the
    # one against the logger that every policy carries
    pilot_pi = {}
    for h, v in pilot["halves"].items():
        src = find_rollouts(F["cfg"].paths.train_output_dir, v["pilot_run"]).get(v["pilot_run"])
        if src is None:
            raise SystemExit(f"[policy] the pilot run {v['pilot_run']} has no rollouts")
        mu_p = mu_from_rollouts(src, F)[0][HEADLINE]
        pilot_pi[int(h)] = tilt(mu_p, F["pi_b"], float(v["beta"]), F["valid"])
    for ut in UTILITIES:
        for run, rec in r2[ut]["generators"].items():
            if rec["arm"] != RT_ARM or rec["half"] not in pilot_pi:
                continue
            lam = float(rec["lam"])
            pi_l = lam * pilot_pi[rec["half"]] + (1 - lam) * F["pi_b"]
            pi = tilt(rec["_mu"], F["pi_b"], rec["beta"], F["valid"])
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(F["valid"] & (pi_l > 0), pi / pi_l, np.nan)
            for which in ("test", "control"):
                r = ratio[sets.sel[which]]
                rec[which]["transfer_factor_vs_training_law"] = {
                    "mean_max_per_context": float(np.nanmean(np.nanmax(r, axis=1))), "max": float(np.nanmax(r)),
                    "bound_1_over_lambda": 1.0 / lam}
    # lambda chosen on validation, per half, for the headline utility; the others follow it
    for ut in UTILITIES:
        gens = r2[ut]["generators"]
        choice = {}
        for h in fixed:
            cands = {run: rec for run, rec in gens.items() if rec["arm"] == RT_ARM and rec["half"] == int(h)}
            if not cands:
                continue
            best = max(cands, key=lambda r: cands[r]["val"]["value"]["value"])
            choice[h] = {"run": best, "lam": cands[best]["lam"],
                         "val_curve": {f"{rec['lam']:g}": rec["val"]["value"] for rec in cands.values()}}
        r2[ut]["selected"] = choice
    return r2


def load_stage(ea: str, name: str) -> dict | None:
    p = os.path.join(ea, OUT_SUBDIR, name)
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def stage_report(ea: str, prefix: str, Fs: dict[str, dict], Ss: dict[str, Sets]) -> dict:
    """L0-L7 from the per-instance stage files and per-context values; "g0" / "g1"
    are the control and the confounded instance."""
    from src.policy.values import load_values
    out = {"criteria": {}, "tables": {}}
    g0, g1 = Fs["g0"], Fs["g1"]
    if not np.array_equal(g0["comp"], g1["comp"]) or not np.array_equal(g0["role"], g1["role"]):
        raise SystemExit("[policy] the two instances do not share contexts and roles")
    sets = Ss["g1"]
    tag = {k: f"g{Fs[k]['gamma']:g}" for k in ("g0", "g1")}       # the stage files are tagged by gamma
    if tag["g0"] == tag["g1"]:
        raise SystemExit("[policy] the two instances have the same gamma")
    V = {k: load_values(os.path.join(ea, OUT_SUBDIR, f"values_{tag[k]}.npz")) for k in ("g0", "g1")}
    gate = load_stage(ea, f"gate_{tag['g1']}.json")
    ut = HEADLINE

    def val(inst, name, which):
        return sets.est(V[inst][f"{ut}/{name}"], which)

    def arm_avg(inst, arm, which, lam_sel=None):
        names = [n for n in V[inst] if n.startswith(f"{ut}/") and parse_run(n.split("/", 1)[1], prefix)
                 and parse_run(n.split("/", 1)[1], prefix)["arm"] == arm]
        if lam_sel is not None:
            names = [n for n in names if n.split("/", 1)[1] in lam_sel]
        if not names:
            return None
        return average([sets.est(V[inst][n], which) for n in sorted(names)])

    def crit(e, thr, **extra):
        if e is None:
            return {"pass": None, "note": "arm not present", **extra}
        if e.z is None:
            return {"pass": None, "note": "no error could be estimated", **e.doc(), "z": None, **extra}
        return {"pass": bool(e.z > thr), **e.doc(), "z": e.z, "needs_z_above": thr, **extra}
    C = out["criteria"]
    C["L0"] = {"pass": gate["pass"] if gate else None, **(gate["utilities"][ut] if gate else {})}
    kp = val("g0", "kept_pooled", "test") - val("g1", "kept_pooled", "test")
    C["L0"]["kept_pooled_gamma0_minus_gamma"] = {**kp.doc(), "z": kp.z,
                                                 "note": "the D0-style difference of the same real-data policy; reported"}
    loss = {arm: ((arm_avg("g0", arm, "test") - arm_avg("g1", arm, "test"))
                  if arm_avg("g0", arm, "test") is not None and arm_avg("g1", arm, "test") is not None else None)
            for arm in GEN_ARMS}
    C["L1_naive_loses"] = crit(loss["naive"], Z, quantity="V(naive, 0) - V(naive, g), test")
    C["L2_conditional_loses_testability"] = crit(loss["conditional"], Z, quantity="V(cond, 0) - V(cond, g), test")
    gain = (loss["conditional"] - loss["dr"]) if loss["conditional"] is not None and loss["dr"] is not None else None
    C["L3_gain_dr"] = crit(gain, Z, quantity="[V(dr, g) - V(dr, 0)] - [V(cond, g) - V(cond, 0)], test")
    r2 = load_stage(ea, f"round2_{tag['g1']}.json")
    sel1 = {h: v["run"] for h, v in (r2 or {}).get(ut, {}).get("selected", {}).items()}
    rt1 = arm_avg("g1", RT_ARM, "test", lam_sel=set(sel1.values())) if sel1 else None
    dr1 = arm_avg("g1", "dr", "test")
    C["L4_retarget_gain"] = crit((rt1 - dr1) if rt1 is not None and dr1 is not None else None, Z,
                                 quantity="V(rt, g) - V(dr, g) at the validation lambda, test", selected=sel1)
    cost = {}
    for arm in ("dr",):
        a0, c0 = arm_avg("g0", arm, "test"), arm_avg("g0", "conditional", "test")
        cost[arm] = (a0 - c0) if a0 is not None and c0 is not None else None
    r20 = load_stage(ea, f"round2_{tag['g0']}.json")
    sel0 = {h: v["run"] for h, v in (r20 or {}).get(ut, {}).get("selected", {}).items()}
    rt0 = arm_avg("g0", RT_ARM, "test", lam_sel=set(sel0.values())) if sel0 else None
    c0 = arm_avg("g0", "conditional", "test")
    cost["rt"] = (rt0 - c0) if rt0 is not None and c0 is not None else None
    C["L5_no_cost_at_gamma0"] = {arm: ({"pass": (None if e.z is None else bool(e.z >= -Z)), **e.doc(), "z": e.z,
                                        "needs_z_at_least": -Z} if e is not None else {"pass": None})
                                 for arm, e in cost.items()}
    parts = [v.get("pass") for v in C["L5_no_cost_at_gamma0"].values() if isinstance(v, dict)]
    C["L5_no_cost_at_gamma0"]["pass"] = (None if any(x is None for x in parts) or not parts else all(parts))
    moved = {}
    for arm in GEN_ARMS + (RT_ARM,):
        a0, a1 = arm_avg("g0", arm, "control"), arm_avg("g1", arm, "control")
        if a0 is not None and a1 is not None:
            e = a0 - a1
            moved[arm] = {**e.doc(), "z": e.z}
    C["L6_control_unchanged"] = {"pass": (None if not moved or any(m["z"] is None for m in moved.values())
                                          else all(abs(m["z"]) <= Z for m in moved.values())),
                                 "needs_abs_z_at_most": Z, "arms": moved}
    r1 = load_stage(ea, f"round1_{tag['g1']}.json")
    tf = {}
    if r1:
        for run, rec in r1[ut]["generators"].items():
            tf[run] = rec["test"]["transfer_factor"]["mean_max_per_context"]
    if r2:
        for run, rec in r2[ut]["generators"].items():
            tf[run] = rec["test"]["transfer_factor"]["mean_max_per_context"]
    cond_tf = [v for k, v in tf.items() if parse_run(k, prefix) and parse_run(k, prefix)["arm"] == "conditional"]
    rt_tf = [v for k, v in tf.items() if k in sel1.values()]
    # the retargeted policy against its own training law (POLICY_LEARNING.md §14: the
    # theorem's quantity), against the conditional policy against ITS training law, the logger
    rt_tl = [r2[ut]["generators"][k]["test"]["transfer_factor_vs_training_law"]["mean_max_per_context"]
             for k in sel1.values() if r2 and "transfer_factor_vs_training_law" in r2[ut]["generators"][k]["test"]]
    C["L7_lambda_and_transfer"] = {"selected_lambda": {h: parse_run(r, prefix)["lam"] for h, r in sel1.items()},
                                   "transfer_factor_conditional_vs_logger": cond_tf,
                                   "transfer_factor_retargeted_vs_logger": rt_tf,
                                   "transfer_factor_retargeted_vs_training_law": rt_tl,
                                   "pass": bool(sel1 and cond_tf and rt_tl and np.mean(cond_tf) > np.mean(rt_tl))}
    C["testable"] = {"task": C["L0"]["pass"], "dr_vs_conditional": C["L2_conditional_loses_testability"]["pass"]}
    if C["L2_conditional_loses_testability"]["pass"] is False:
        for k in ("L3_gain_dr", "L4_retarget_gain"):
            C[k] = {**C[k], "pass": None, "would_pass": C[k].get("pass"), "note": "not testable: L2 did not pass"}
    if C["L0"]["pass"] is False:
        for k in ("L1_naive_loses", "L2_conditional_loses_testability", "L3_gain_dr", "L4_retarget_gain"):
            if isinstance(C[k].get("pass"), bool):
                C[k] = {**C[k], "pass": None, "would_pass": C[k]["pass"], "note": "descriptive: L0 did not pass"}
    # the value table, both instances, test and control
    tab = {}
    for inst in ("g0", "g1"):
        for name in sorted(V[inst]):
            if not name.startswith(f"{ut}/"):
                continue
            for which in ("test", "control"):
                tab.setdefault(name.split("/", 1)[1], {})[f"{inst}/{which}"] = sets.est(V[inst][name], which).doc()
    out["tables"]["values"] = tab
    return out


def print_report(rep: dict, g1: float) -> None:
    print(f"\n## Values on the holdout truth (utility {HEADLINE}; test = thinned test compounds, control = unthinned)\n")
    print(f"| policy | γ = 0, test | γ = {g1:g}, test | γ = 0, control | γ = {g1:g}, control |\n|---|---|---|---|---|")
    for name, row in rep["tables"]["values"].items():
        f = lambda k: ("n/a" if k not in row or row[k]["value"] is None else
                       f"{row[k]['value']:+.3f}" + (f" ± {row[k]['se']:.3f}" if row[k].get("se") else ""))
        print(f"| `{name}` | {f('g0/test')} | {f('g1/test')} | {f('g0/control')} | {f('g1/control')} |")
    print("\n## Criteria\n\n| # | value ± SE | z | result |\n|---|---|---|---|")
    for k, c in rep["criteria"].items():
        if not isinstance(c, dict) or "pass" not in c:
            continue
        v = ("n/a" if c.get("value") is None else f"{c['value']:+.3f}" + (f" ± {c['se']:.3f}" if c.get("se") else ""))
        z = "n/a" if c.get("z") is None else f"{c['z']:+.2f}"
        res = "PASS" if c["pass"] else ("FAIL" if c["pass"] is False else "not judged")
        print(f"| {k} | {v} | {z} | {res} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--tier", default=None, help="The (parent) thinning instance; gate/round1/pilot/round2.")
    p.add_argument("--tier_g0", default=None, help="report: the gamma = 0 instance.")
    p.add_argument("--tier_g1", default=None, help="report: the confounded instance.")
    p.add_argument("--prefix", default="pl", help="Run-name prefix: <prefix>_<arm>_g<gamma>_h<half>[_l<lambda>].")
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--tag", default=None, help="Suffix of the stage files (default g<gamma>).")
    p.add_argument("--signature_file", default=None, help="Utility A's axis per compound (.npy, keyed by compound_idx). "
                   "Default: the holdout signature, or on an injected build its own injection_V.npy.")
    p.add_argument("--uninjected_runs_dir", default=None, help="Injected build: the runs dir of the REAL build holding the "
                   "same runs, for the planted-share diagnostic (default runs/<source build>; '' skips it).")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    t0 = time.time()
    from src.eval import dist_metrics as dm
    print(f"[blas] numpy matrix products verified (rel err {dm.assert_blas_ok():.1e})", flush=True)
    cfg = apply_paths_args(config_from_args(args), args)
    ea = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    od = os.path.join(ea, OUT_SUBDIR)
    os.makedirs(od, exist_ok=True)
    from src.policy.values import save_values

    if args.stage == "report":
        if not (args.tier_g0 and args.tier_g1):
            raise SystemExit("report needs --tier_g0 and --tier_g1")
        Fs, Ss = {}, {}
        for k, t in (("g0", args.tier_g0), ("g1", args.tier_g1)):
            Fs[k] = load_frame(cfg, t, args.val_seed, signature_file=args.signature_file); Fs[k]["cfg"] = cfg
            Ss[k] = Sets(Fs[k])
        rep = stage_report(ea, args.prefix, Fs, Ss)
        rep["elapsed_sec"] = round(time.time() - t0, 1)
        _write_json(os.path.join(od, "policy_verdict.json"), strip(rep))
        print_report(rep, Fs["g1"]["gamma"])
        print(f"\n[policy] -> {od}/policy_verdict.json ({rep['elapsed_sec']}s)")
        return
    if not args.tier:
        raise SystemExit("--tier is required")
    F = load_frame(cfg, args.tier, args.val_seed, signature_file=args.signature_file); F["cfg"] = cfg
    sets = Sets(F)
    tag = args.tag or f"g{F['gamma']:g}"
    uninjected = None
    if F["injection"] is not None and args.uninjected_runs_dir != "":
        ud = args.uninjected_runs_dir or os.path.join(os.path.dirname(os.path.normpath(cfg.paths.train_output_dir)),
                                                      os.path.basename(F["injection"]["source_data_dir"]))
        uninjected = find_rollouts(ud, f"{args.prefix}_*")
        print(f"[policy] injected build (beta {F['injection']['beta']:.3f}); uninjected counterparts from {ud}: "
              f"{len(uninjected)} rollouts", flush=True)
    if args.stage == "gate":
        rec = stage_gate(F, sets)
        rec.update({"tier": F["tier"], "tier_fingerprint": F["tier_fingerprint"], "gamma": F["gamma"],
                    "injection": F["injection"],
                    "val_compounds": F["val_comp"].tolist(), "test_compounds": F["test_comp"].tolist()})
        _write_json(os.path.join(od, f"gate_{tag}.json"), strip(rec))
        for ut, r in rec["utilities"].items():
            print(f"[gate] utility {ut}: (a) the reference tilt puts {r['thin_half_mass_of_reference_tilt']:.3f} of its mass on "
                  f"the thin half (needs >= {GATE_MIN_THIN_MASS:.3f}); (b) reference minus kept-pooled value "
                  f"{r['reference_minus_kept_pooled']['value']:+.4f} ± {r['reference_minus_kept_pooled']['se']:.4f} "
                  f"(z {fmt_z(r['reference_minus_kept_pooled']['z'])}, needs >= {GATE_MIN_SE:g}); (c) kept per-line minus "
                  f"kept-pooled {r['kept_strat_minus_kept_pooled']['value']:+.4f} ± {r['kept_strat_minus_kept_pooled']['se']:.4f} "
                  f"(z {fmt_z(r['kept_strat_minus_kept_pooled']['z'])}); kept wells on the thin half "
                  f"{r['kept_wells_on_thin_half_share']:.3f}: {'PASS' if r['pass'] else 'FAIL'}", flush=True)
        print(f"[gate] L0 ({HEADLINE}): {'PASS' if rec['pass'] else 'FAIL -- the task is not testable (POLICY_LEARNING.md §6)'}")
        return
    rolls = find_rollouts(cfg.paths.train_output_dir, f"{args.prefix}_*")
    if args.stage == "round1":
        r1 = stage_round(F, sets, {k: v for k, v in rolls.items() if parse_run(k, args.prefix) and parse_run(k, args.prefix)["lam"] is None},
                         args.prefix, None, uninjected=uninjected)
        _write_json(os.path.join(od, f"round1_{tag}.json"), strip(r1))
        save_values(os.path.join(od, f"values_{tag}.npz"), F, r1, merge=True)
        for ut in UTILITIES:
            print(f"\n[round1] utility {ut} (γ = {F['gamma']:g}; thinned test compounds)")
            for grp in ("real", "generators"):
                for name, rec in r1[ut][grp].items():
                    t = rec["test"]
                    ps = rec.get("planted_share")
                    print(f"  {name:28s} beta {rec['beta']!s:>5} value {t['value']['value']:+.3f} ± {t['value']['se']:.3f}  "
                          f"gain {t['gain_over_random']['value']:+.3f}  thin-mass {t['mass_on_thin_half']:.3f}  "
                          f"TF {t['transfer_factor']['mean_max_per_context']:.2f}"
                          + (f"  self-eval {t['self_evaluation_error']['value']:+.3f}" if "self_evaluation_error" in t else "")
                          + (f"  planted share thin {ps['thinned']['thin_half']['value']:+.3f}±{ps['thinned']['thin_half']['se']:.3f} "
                             f"fav {ps['thinned']['favoured_half']['value']:+.3f} (control thin {ps['control']['thin_half']['value']:+.3f})"
                             if ps else ""))
    elif args.stage == "pilot":
        r1 = load_stage(ea, f"round1_{tag}.json")
        if r1 is None:
            raise SystemExit("pilot needs the round-1 stage's file")
        rec = stage_pilot(F, r1, rolls)
        _write_json(os.path.join(od, f"pilot_{tag}.json"), strip(rec))
    elif args.stage == "round2":
        pilot = load_stage(ea, f"pilot_{tag}.json")
        if pilot is None:
            raise SystemExit("round2 needs the pilot stage's file")
        r2 = stage_round2(F, sets, {k: v for k, v in rolls.items() if parse_run(k, args.prefix) and parse_run(k, args.prefix)["lam"] is not None},
                          args.prefix, pilot, uninjected=uninjected)
        _write_json(os.path.join(od, f"round2_{tag}.json"), strip(r2))
        save_values(os.path.join(od, f"values_{tag}.npz"), F, r2, merge=True)
        for ut in UTILITIES:
            print(f"\n[round2] utility {ut} (γ = {F['gamma']:g}; thinned test compounds); selected {r2[ut].get('selected')}")
            for name, rec in r2[ut]["generators"].items():
                t = rec["test"]
                ps = rec.get("planted_share"); tl = t.get("transfer_factor_vs_training_law")
                print(f"  {name:28s} beta {rec['beta']:g} value {t['value']['value']:+.3f} ± {t['value']['se']:.3f}  "
                      f"gain {t['gain_over_random']['value']:+.3f}  thin-mass {t['mass_on_thin_half']:.3f}  "
                      f"TF {t['transfer_factor']['mean_max_per_context']:.2f}"
                      + (f" (vs training law {tl['mean_max_per_context']:.2f})" if tl else "")
                      + f"  self-eval {t['self_evaluation_error']['value']:+.3f}"
                      + (f"  planted share thin {ps['thinned']['thin_half']['value']:+.3f}±{ps['thinned']['thin_half']['se']:.3f}" if ps else ""))
    print(f"[policy] stage {args.stage} done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
