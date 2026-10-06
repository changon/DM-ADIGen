"""Decisions from the step-A generators (`DECISION.md`). No PyTorch.

Step A scored each generator on the error of its effect estimates. This scores
what the estimates are for: choosing a dose, or choosing compounds. Nothing is
trained or sampled: it reads the per-row generated means the step-A scoring jobs
saved (`*_gen.npz`, `row_mean`).

One effect for every decision (DECISION.md §3), for the truth and every estimator:

    tau(c, d)   = mean over the L lines of [ mu(c, d, line) - mu(0, line) ]     (equal line weights)
    ATE_s(c, d) = <tau(c, d), s>

    task A   s = s_c, the direction of compound c's effect averaged over its
             eligible doses, from HOLDOUT wells only. Decision: the dose with the
             largest ATE.
    task B   s = minus the mean of spec.PROLIFERATION_GENES (a unit vector, fixed
             in spec, never fitted). Decision: the top-k compounds, each at its
             best dose.

A policy's VALUE is the mean true effect of what it chose. The truth is each
arm's holdout wells (no generator trained on one, so the value is unbiased for
any policy); the all-wells effect is the secondary truth, and it shares two
thirds of its wells with the training set. An arm is eligible when it has a
holdout well in every line; a compound enters with >= spec.DECISION_MIN_DOSES
eligible doses, and every policy chooses among those same doses.

Policies: the four generators at gamma = 0 and gamma > 0, and three built from
real train wells -- `pooled_real` (the kept wells, lines ignored: what `naive`
imitates), `stratified_real` (the kept wells per line, equal weights: what the
adjusted arms imitate) and `full_data` (stratified, over the UNTHINNED train
wells: the reference). `random` is a uniformly random eligible dose (or compound).

    net at gamma > 0  =  V_arm - V_conditional
                      =  [V_arm - V_conditional at gamma = 0]                       the weights' noise cost
                       + [(V_arm(g) - V_arm(0)) - (V_cond(g) - V_cond(0))]          the gain from removing the confounding

Errors: task A is a mean over compounds, so a delete-one-compound jackknife; a
top-k mean is not smooth, so task B uses a bootstrap over compounds. Both keep
the replicates, so every difference and ratio below gets a paired error.

Criteria D0-D5 are DECISION.md §6. D0 (does the planted confounding move the
decision in the raw data at all) is printed before any generator is read.

    python -m src.eval.decision_task --data_dir data/core5_24h
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.contrast_stats import jackknife_se  # noqa: E402
from src.spec import (  # noqa: E402
    DECISION_K, DECISION_K_CONTROL, DECISION_K_CURVE, DECISION_MIN_DOSES, PROLIFERATION_GENES,
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args, line_group_sign,
    line_groups)

GEN_ARMS = ("naive", "conditional", "dr", "dr_p2")
REAL_ARMS = ("pooled_real", "stratified_real")
ADIGEN_ARMS = ("dr", "dr_p2")
BASELINE = "conditional"
REFERENCE, RANDOM = "full_data", "random"
SETS = ("thinned", "unthinned")
TRUTHS = ("holdout", "all_wells")          # primary first
D0_MIN_SE = 3.0                            # the power gate
Z = 2.0                                    # D1-D5
# A captured share is a ratio; its error is reported only when the reference's own
# gain over a random choice (the denominator) is at least this many SE from zero.
SHARE_MIN_Z = 3.0
# The task-B axis is read as anti-proliferative; these panel families (evaluate.PANEL)
# should then sit near the top of the oracle's ranking. A diagnostic, not a gate.
AXIS_SANITY_FAMILIES = ("proteasome", "hsp90", "hdac")
AXIS_SANITY_MIN_PCT = 0.8


def pkey(arm: str, gamma: float) -> str:
    return f"{arm}|g{gamma:g}"


# ---------------------------------------------------------------------------
# effects from rows
# ---------------------------------------------------------------------------
def _group_sum(y: np.ndarray, g: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_groups, G) float64 sums of y's rows by group id, and the group sizes."""
    out = np.zeros((n_groups, y.shape[1]), dtype=np.float64)
    if g.size:
        order = np.argsort(g, kind="stable")
        gs = g[order]
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        out[gs[starts]] = np.add.reduceat(y[order], starts, axis=0, dtype=np.float64)
    return out, np.bincount(g, minlength=n_groups).astype(np.float64)


def effects_from_rows(y: np.ndarray, arm: np.ndarray, line: np.ndarray, ctl: np.ndarray,
                      n_arms: int, n_lines: int, g2_lines: np.ndarray | None = None) -> dict:
    """Per-arm effects from a set of rows (real wells, or one run's generated rows).

    `arm` is each row's arm index (-1: not in the arm table; ignored on vehicle
    rows), `line` its line index and `ctl` marks the vehicles. Returns

      cnt      (K, L) treated rows per (arm, line) cell
      eq       (K, G) the EQUAL-WEIGHT effect, mean over lines of [cell mean - that
               line's vehicle mean]; NaN unless the arm has a row in every line
      pooled   (K, G) the arm's mean over all its rows minus the vehicle mean over
               all vehicle rows: the arm's realised line mix, lines ignored
      contrast (K, G) tau_G2 - tau_G1, each an equal-weight mean over the group's
               lines (only with `g2_lines`, a bool per line); NaN as `eq`
    """
    ctl = np.asarray(ctl).astype(bool)
    trt = ~ctl & (arm >= 0)
    K, L = int(n_arms), int(n_lines)
    csum, ccnt = _group_sum(y[trt], arm[trt] * L + line[trt], K * L)
    vsum, vcnt = _group_sum(y[ctl], line[ctl], L)
    if (vcnt == 0).any():
        raise RuntimeError(f"these rows hold no vehicle in line(s) {np.flatnonzero(vcnt == 0).tolist()}")
    G = y.shape[1]
    csum, ccnt = csum.reshape(K, L, G), ccnt.reshape(K, L)
    veh = vsum / vcnt[:, None]
    full = (ccnt > 0).all(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = csum / ccnt[:, :, None] - veh[None]                  # (K, L, G); NaN where the cell is empty
        n_arm = ccnt.sum(axis=1)
        pooled = csum.sum(axis=1) / n_arm[:, None] - vsum.sum(axis=0) / vcnt.sum()
    out = {"cnt": ccnt, "eq": np.where(full[:, None], dev.mean(axis=1), np.nan), "pooled": pooled}
    if g2_lines is not None:
        g2 = np.asarray(g2_lines, dtype=bool)
        out["contrast"] = np.where(full[:, None], dev[:, g2].mean(axis=1) - dev[:, ~g2].mean(axis=1), np.nan)
    return out


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise <a, b>, without BLAS (IMPLEMENT.md §5, the Sapphire Rapids dgemm)."""
    return np.einsum("ag,ag->a", a, b)


def gene_axis(symbols, genes=PROLIFERATION_GENES) -> np.ndarray:
    """Task B's unit vector: -1/sqrt(m) on the m genes of the set, 0 elsewhere."""
    pos = {str(s): i for i, s in enumerate(symbols)}
    missing = [g for g in genes if g not in pos]
    if missing:
        raise SystemExit(f"[decision] task B's genes are not all in this gene order: {missing}")
    s = np.zeros(len(pos), dtype=np.float64)
    s[[pos[g] for g in genes]] = -1.0 / np.sqrt(len(genes))
    return s


# ---------------------------------------------------------------------------
# the compound x dose layout
# ---------------------------------------------------------------------------
@dataclass
class Design:
    """The compounds that enter and the doses each policy may choose among."""
    comp: np.ndarray          # (C,) compound ids
    arm: np.ndarray           # (C, D) arm index of each eligible dose, ascending dose; -1 = padding
    rank: np.ndarray          # (C, D) the dose's rank among ALL the compound's arms (0 = lowest)
    thinned: np.ndarray       # (C,) bool

    @property
    def valid(self) -> np.ndarray:
        return self.arm >= 0

    def take(self, x: np.ndarray) -> np.ndarray:
        """(C, D) a per-arm array laid out by compound and dose; NaN on the padding."""
        return np.where(self.valid, np.asarray(x, dtype=np.float64)[np.clip(self.arm, 0, None)], np.nan)

    def choose(self, score: np.ndarray) -> np.ndarray:
        """(C,) the column of each compound's largest score."""
        s = self.take(score)
        if not np.isfinite(s[self.valid]).all():
            raise RuntimeError("a policy has no finite score on an eligible dose")
        return np.where(self.valid, s, -np.inf).argmax(axis=1)

    def at(self, x_cd: np.ndarray, choice: np.ndarray) -> np.ndarray:
        return x_cd[np.arange(self.comp.size), choice]

    def sets(self) -> dict[str, np.ndarray]:
        return {"thinned": self.thinned, "unthinned": ~self.thinned}


def build_design(comp_of_arm: np.ndarray, dose_of_arm: np.ndarray, eligible: np.ndarray,
                 thinned_comps, min_doses: int = DECISION_MIN_DOSES) -> Design:
    comp_of_arm = np.asarray(comp_of_arm, dtype=np.int64)
    dose_of_arm = np.asarray(dose_of_arm, dtype=np.float64)
    order = np.lexsort((dose_of_arm, comp_of_arm))
    cs = comp_of_arm[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], cs.size]
    rows = []
    for a, b in zip(starts, ends):
        arms = order[a:b]                                  # this compound's arms, ascending dose
        if np.unique(dose_of_arm[arms]).size != arms.size:
            raise RuntimeError(f"compound {int(cs[a])} has two arms at one dose level")
        keep = np.flatnonzero(eligible[arms])
        if keep.size >= min_doses:
            rows.append((int(cs[a]), arms[keep], keep))
    D = max((r[1].size for r in rows), default=0)
    arm = np.full((len(rows), D), -1, dtype=np.int64)
    rank = np.full((len(rows), D), -1, dtype=np.int64)
    for i, (_, arms, rk) in enumerate(rows):
        arm[i, :arms.size], rank[i, :arms.size] = arms, rk
    comp = np.array([r[0] for r in rows], dtype=np.int64)
    return Design(comp=comp, arm=arm, rank=rank,
                  thinned=np.isin(comp, np.asarray(sorted(thinned_comps), dtype=np.int64)))


def compound_axes(tau: np.ndarray, design: Design) -> np.ndarray:
    """(K, G) task A's axis of each arm's compound: the unit direction of the mean
    of `tau` over the compound's eligible doses. Zero rows off the design."""
    S = np.zeros_like(tau, dtype=np.float64)
    for arms in design.arm:
        arms = arms[arms >= 0]
        m = tau[arms].mean(axis=0)
        n = float(np.sqrt((m ** 2).sum()))
        if not np.isfinite(n) or n == 0:
            raise RuntimeError("a compound's holdout effect has no direction")
        S[arms] = m / n
    return S


# ---------------------------------------------------------------------------
# a statistic with its replicates
# ---------------------------------------------------------------------------
class Est:
    """A statistic and its resampling replicates: delete-one-compound jackknife
    ("jk") or compound bootstrap ("boot"). Statistics built on the same compounds
    have aligned replicates, so their sums, differences and ratios keep a paired
    error."""

    def __init__(self, value: float, rep: np.ndarray, kind: str):
        self.value, self.rep, self.kind = float(value), np.asarray(rep, dtype=np.float64), kind

    @classmethod
    def mean(cls, x: np.ndarray) -> "Est":
        x = np.asarray(x, dtype=np.float64)
        if x.size < 2 or not np.isfinite(x).all():
            raise ValueError("Est.mean needs >= 2 finite values")
        return cls(x.mean(), (x.sum() - x) / (x.size - 1), "jk")

    def _with(self, other, fn) -> "Est":
        if isinstance(other, Est):
            if other.kind != self.kind or other.rep.shape != self.rep.shape:
                raise ValueError("these statistics were not built on the same replicates")
            return Est(fn(self.value, other.value), fn(self.rep, other.rep), self.kind)
        return Est(fn(self.value, other), fn(self.rep, other), self.kind)

    def __add__(self, o): return self._with(o, lambda a, b: a + b)
    def __sub__(self, o): return self._with(o, lambda a, b: a - b)
    def __truediv__(self, o): return self._with(o, lambda a, b: a / b)

    @property
    def se(self) -> float | None:
        v = self.rep[np.isfinite(self.rep)]
        if v.size < 2:
            return None
        return jackknife_se(v) if self.kind == "jk" else float(v.std(ddof=1))

    @property
    def z(self) -> float | None:
        se = self.se
        return None if not se else self.value / se

    def doc(self) -> dict:
        return {"value": self.value, "se": self.se}


def average(ests: list[Est]) -> Est:
    out = ests[0]
    for e in ests[1:]:
        out = out + e
    return out / float(len(ests))


def bootstrap_counts(n: int, n_boot: int, seed: int, name: str) -> np.ndarray:
    """(n_boot, n) multiplicities of a bootstrap over n compounds; one stream per `name`."""
    h = hashlib.blake2b(f"decision|boot|{name}".encode(), digest_size=8)
    rng = np.random.default_rng([int(seed), int.from_bytes(h.digest(), "little")])
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot)


def topk_value(score: np.ndarray, w: np.ndarray, k: int, counts: np.ndarray | None = None):
    """Mean of `w` over the k items with the highest `score`. With bootstrap
    multiplicities `counts` (B, n): the same on each resample, where an item drawn
    m times stands for m items (so the last one taken may be taken in part)."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    if k > order.size:
        raise ValueError(f"top-{k} of {order.size} items")
    if counts is None:
        return float(np.asarray(w, dtype=np.float64)[order[:k]].mean())
    m = counts[:, order]
    before = np.cumsum(m, axis=1) - m
    take = np.clip(k - before, 0, m)
    return (take * np.asarray(w, dtype=np.float64)[order]).sum(axis=1) / k


# ---------------------------------------------------------------------------
# policies -> values
# ---------------------------------------------------------------------------
def dose_choices(scores: dict[str, list[np.ndarray]], design: Design) -> dict[str, list[np.ndarray]]:
    """Per policy and seed, (C,) the column each compound's argmax picks."""
    return {name: [design.choose(s) for s in seeds] for name, seeds in scores.items()}


def chosen_values(choices: dict, truth: np.ndarray, design: Design) -> dict[str, list[np.ndarray]]:
    """Per policy and seed, (C,) the truth at the chosen dose; `random` is the mean
    of the truth over the compound's eligible doses."""
    T = design.take(truth)
    if not np.isfinite(T[design.valid]).all():
        raise RuntimeError("the truth is not finite on an eligible dose")
    out = {name: [design.at(T, c) for c in ch] for name, ch in choices.items()}
    out[RANDOM] = [np.nanmean(T, axis=1)]
    return out


def dose_stats(values: dict, sel: np.ndarray) -> tuple[dict[str, Est], dict[str, list[float]]]:
    """V of each policy over the selected compounds (seeds averaged per compound),
    and the per-seed values."""
    V = {name: Est.mean(np.mean([v[sel] for v in vs], axis=0)) for name, vs in values.items()}
    return V, {name: [float(v[sel].mean()) for v in vs] for name, vs in values.items()}


def screen_stats(best: dict, values: dict, sel: np.ndarray, k: int,
                 counts: np.ndarray) -> tuple[dict[str, Est], dict[str, list[float]]]:
    """V of each policy's top-k screen over the selected compounds. `best` is each
    policy's best-dose score per compound (what it ranks on), `values` the truth
    at its chosen dose. `random` is k random compounds at a random dose."""
    V, per_seed = {}, {}
    for name, bs in best.items():
        e = [Est(topk_value(b[sel], v[sel], k), topk_value(b[sel], v[sel], k, counts), "boot")
             for b, v in zip(bs, values[name])]
        V[name], per_seed[name] = average(e), [x.value for x in e]
    r = values[RANDOM][0][sel]
    V[RANDOM] = Est(r.mean(), (counts * r).sum(axis=1) / counts.sum(axis=1), "boot")
    per_seed[RANDOM] = [float(r.mean())]
    return V, per_seed


def policy_table(V: dict[str, Est], per_seed: dict) -> dict:
    """Per policy: its gain over `random` and its captured share of the reference's gain."""
    span = V[REFERENCE] - V[RANDOM]
    stable = span.z is not None and abs(span.z) >= SHARE_MIN_Z and span.value > 0
    out = {}
    for name, v in V.items():
        if name == RANDOM:
            continue
        gain = v - V[RANDOM]
        seeds = [x - V[RANDOM].value for x in per_seed[name]]
        share = (gain / span).doc() if stable else {
            "value": gain.value / span.value if span.value else None, "se": None, "unstable_denominator": True}
        out[name] = {"gain_over_random": gain.doc(), "captured_share": share, "seed_values": seeds,
                     "seed_sd": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else None}
    return {"random_value": V[RANDOM].doc(), "reference_gain_over_random": {**span.doc(), "z": span.z},
            "captured_share_has_error": bool(stable), "policies": out}


def judge(V: dict[str, Est], Vc: dict[str, Est] | None, g1: float) -> dict:
    """Criteria D0-D5 (DECISION.md §6) and the split of the net difference, from
    the policies' values on the judged compounds (`V`) and on the control (`Vc`)."""
    def loss(d, arm):
        a, b = pkey(arm, 0.0), pkey(arm, g1)
        return (d[a] - d[b]) if a in d and b in d else None

    def crit(e: Est | None, thr: float, *, inclusive: bool = False, **extra) -> dict:
        if e is None:
            return {"pass": None, "note": "arm not present", **extra}
        z = e.z
        ok = z is not None and (z >= thr if inclusive else z > thr)
        return {"pass": bool(ok), **e.doc(), "z": z,
                ("needs_z_at_least" if inclusive else "needs_z_above"): thr, **extra}

    out = {"D0_power_gate": crit(loss(V, "pooled_real"), D0_MIN_SE, inclusive=True,
                                 quantity="V(pooled_real, gamma = 0) - V(pooled_real, gamma > 0)"),
           "D1_naive_loses": crit(loss(V, "naive"), Z, quantity="V(naive, 0) - V(naive, g)"),
           "D2_conditional_loses_testability": crit(loss(V, BASELINE), Z,
                                                    quantity="V(conditional, 0) - V(conditional, g)")}
    for arm in ADIGEN_ARMS:
        la, lb = loss(V, arm), loss(V, BASELINE)
        if la is None or lb is None:
            continue
        gain = lb - la                                    # the baseline loses more than the arm
        cost = V[pkey(arm, 0.0)] - V[pkey(BASELINE, 0.0)]
        net = V[pkey(arm, g1)] - V[pkey(BASELINE, g1)]
        zc = cost.z
        out[f"D3_gain_{arm}"] = crit(gain, Z, quantity=f"[V({arm}, g) - V({arm}, 0)] - [V(cond, g) - V(cond, 0)]")
        out[f"D4_no_cost_{arm}"] = {"pass": bool(zc is None or zc >= -Z), **cost.doc(), "z": zc,
                                    "needs_z_at_least": -Z, "quantity": f"V({arm}, 0) - V(cond, 0)"}
        out[f"net_{arm}"] = {**net.doc(), "z": net.z, "quantity": f"V({arm}, g) - V(cond, g) = D4 + D3",
                             "split_residual": abs(net.value - cost.value - gain.value)}
    if Vc is not None:
        moved = {}
        for arm in GEN_ARMS:
            e = loss(Vc, arm)
            if e is not None:
                moved[arm] = {**e.doc(), "z": e.z}
        out["D5_control_unthinned"] = {
            "pass": bool(all(m["z"] is None or abs(m["z"]) <= Z for m in moved.values())),
            "needs_abs_z_at_most": Z, "quantity": "V(arm, 0) - V(arm, g) on the unthinned compounds",
            "arms": moved}
    # D2 is the testability check: without it D3's quantity is reported, not judged.
    def unjudge(names, note: str) -> None:
        for name in names:
            if isinstance(out[name].get("pass"), bool):
                out[name] = {**out[name], "pass": None, "would_pass": out[name]["pass"], "note": note}

    d3 = [k for k in out if k.startswith("D3_gain_")]
    out["task_testable"] = out["D0_power_gate"]["pass"]
    out["D3_testable"] = out["D2_conditional_loses_testability"]["pass"]
    if out["D3_testable"] is False:
        unjudge(d3, "not testable: D2 did not pass")
    # D0 is the gate on the whole task: without it D1-D3 are descriptive. D4 (the
    # weights' cost at gamma = 0) and D5 (the control) do not depend on it.
    if out["task_testable"] is False:
        unjudge(["D1_naive_loses", "D2_conditional_loses_testability"] + d3,
                "descriptive: D0 did not pass, the task is not testable")
        out["D3_testable"] = False
    return out


# ---------------------------------------------------------------------------
# descriptive
# ---------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two continuous scores (no ties to share), without BLAS."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def dose_descriptive(choices: dict, best: dict, design: Design, sel: np.ndarray,
                     kappa: np.ndarray, tilt_sign: float, g1: float) -> dict:
    """Not judged: how often a policy picks the reference's dose; the rank
    correlation of its best-dose score with the reference's; and how far its
    chosen dose moves from gamma = 0 to gamma > 0, signed so that positive is the
    direction the thinning predicts for a line-blind estimator."""
    ref_c, ref_b = choices[REFERENCE][0], best[REFERENCE][0]
    want = tilt_sign * np.sign(kappa)
    out = {}
    for name, ch in choices.items():
        arm = name.split("|")[0]
        d = {"same_dose_as_reference": float(np.mean([(c == ref_c)[sel].mean() for c in ch])),
             "spearman_best_score_vs_reference": float(np.mean([_spearman(b[sel], ref_b[sel])
                                                                for b in best[name]]))}
        k0 = pkey(arm, 0.0)
        if name == pkey(arm, g1) and k0 in choices and len(choices[k0]) == len(ch):
            moves = [(design.at(design.rank, c1) - design.at(design.rank, c0))
                     for c1, c0 in zip(ch, choices[k0])]
            d["dose_rank_shift_as_predicted"] = float(np.mean([(want * m)[sel].mean() for m in moves]))
            d["dose_changed"] = float(np.mean([(m != 0)[sel].mean() for m in moves]))
        out[name] = d
    return out


def screen_descriptive(best: dict, sel: np.ndarray, k: int, g1: float) -> dict:
    """Not judged: the share of a policy's top-k that is also in the reference's,
    and in `conditional`'s at the same gamma and seed."""
    def top(b):
        return set(np.argsort(-b[sel], kind="stable")[:k].tolist())
    ref = top(best[REFERENCE][0])
    out = {}
    for name, bs in best.items():
        d = {"overlap_with_reference": float(np.mean([len(top(b) & ref) / k for b in bs]))}
        arm, _, g = name.partition("|")
        base = f"{BASELINE}|{g}"
        if arm in ADIGEN_ARMS and base in best and len(best[base]) == len(bs):
            d["overlap_with_conditional"] = float(np.mean([len(top(b) & top(c)) / k
                                                           for b, c in zip(bs, best[base])]))
        out[name] = d
    return out


# ---------------------------------------------------------------------------
# one task on one truth
# ---------------------------------------------------------------------------
def analyse_task(scores: dict[str, list[np.ndarray]], truth: np.ndarray, design: Design, *,
                 screens: bool, kappa: np.ndarray, tilt_sign: float, g1: float,
                 n_boot: int, seed: int) -> dict:
    """Everything for one axis and one truth. `scores` maps a policy (`pkey(arm,
    gamma)` or `full_data`) to its per-arm ATE, one array per training seed.
    The dose block is judged for task A; with `screens`, the top-k block is
    judged (task B) and the dose block is descriptive."""
    choices = dose_choices(scores, design)
    values = chosen_values(choices, truth, design)
    best = {name: [np.where(design.valid, design.take(s), -np.inf).max(axis=1) for s in seeds]
            for name, seeds in scores.items()}
    sets = design.sets()
    out, stats = {"dose": {}}, {}
    out["_per_compound"] = {"value": {name: np.mean(vs, axis=0) for name, vs in values.items()},
                            "dose_rank": {name: np.stack([design.at(design.rank, c) for c in ch])
                                          for name, ch in choices.items()}}
    for sname, sel in sets.items():
        V, per_seed = dose_stats(values, sel)
        stats[sname] = V
        out["dose"][sname] = {"n_compounds": int(sel.sum()), **policy_table(V, per_seed),
                              "descriptive": dose_descriptive(choices, best, design, sel, kappa, tilt_sign, g1)}
    if not screens:
        out["criteria"] = judge(stats["thinned"], stats["unthinned"], g1)
        return out
    out["screens"], sstats = {}, {}
    pools = {"all": np.ones(design.comp.size, dtype=bool), **sets}
    for sname, sel in pools.items():
        counts = bootstrap_counts(int(sel.sum()), n_boot, seed, sname)
        ks = sorted({k for k in DECISION_K_CURVE + (DECISION_K, DECISION_K_CONTROL) if k <= int(sel.sum())})
        blk = {"n_candidates": int(sel.sum()), "k": {}}
        for k in ks:
            V, per_seed = screen_stats(best, values, sel, k, counts)
            sstats[(sname, k)] = V
            blk["k"][str(k)] = {**policy_table(V, per_seed), "descriptive": screen_descriptive(best, sel, k, g1)}
        out["screens"][sname] = blk
    main, ctrl = ("all", DECISION_K), ("unthinned", DECISION_K_CONTROL)
    if main in sstats:
        out["criteria"] = {"judged_on": {"screen": main[0], "k": main[1],
                                         "control_screen": ctrl[0], "control_k": ctrl[1]},
                           **judge(sstats[main], sstats.get(ctrl), g1)}
    return out


# ---------------------------------------------------------------------------
# I/O: the table, the real effects, the runs
# ---------------------------------------------------------------------------
def read_verdict(cfg, path: str | None) -> tuple[dict, str]:
    """The step_a_report verdict that lists the runs (default: this population's
    --pool all verdict), refused unless it is that."""
    vpath = path or os.path.join(cfg.paths.train_output_dir, "eval_artifacts", "step_a_verdict.json")
    with open(vpath) as fh:
        verdict = json.load(fh)
    if verdict.get("pool") != "all" or verdict.get("population") != cfg.population.name:
        raise SystemExit(f"[decision] {vpath}: need this population's --pool all verdict")
    return verdict, vpath


def load_world(cfg, verdict: dict) -> dict:
    """The table, the normalised expression, the arm table of the oracle, the
    masks of each well set, and the tier instances the verdict's runs trained on."""
    from datasets import load_from_disk

    from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes
    from src.data.splits import arm_keys, dose_half, load_splits

    with open(verdict["truth"]) as fh:
        odoc = json.load(fh)
    splits = load_splits(cfg)                      # the BASE split: the oracle's frame
    if odoc.get("split_fingerprint") != splits["split_fingerprint"]:
        raise SystemExit(f"[decision] the oracle was built on split {odoc.get('split_fingerprint')}, "
                         f"not the base split {splits['split_fingerprint']}")
    oz = np.load(verdict["truth"][:-5] + "_tau.npz", allow_pickle=False)
    ak = oz["all/arm_key"].astype(str)
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "det_plate", "cell_id", "pert_iname"])
            .to_pandas())
    N = len(meta)
    comp_t = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ic = meta["is_control"].values.astype(np.int64)
    keys = arm_keys(comp_t, dl, ic).astype(str)
    lines = sorted(set(meta["cell_id"].values))
    if sorted(cfg.population.cell_ids) != lines:
        raise SystemExit(f"[decision] the table's lines {lines} are not the population's")
    line_t = np.array([lines.index(c) for c in meta["cell_id"].values], dtype=np.int64)
    m = load_expr_meta(cfg, odoc["plate_center"], splits=splits)
    y = normalize_expr(np.asarray(np.load(cfg.paths.expr_npy, mmap_mode="r")),
                       plate_codes(meta["det_plate"].values, m["plates"]), m)
    pos = {k: i for i, k in enumerate(ak)}
    arm_of_row = np.array([pos.get(k, -1) for k in keys], dtype=np.int64)
    half_of = dict(zip(keys, dose_half(comp_t, dl, ic)))

    def mask(idx):
        out = np.zeros(N, dtype=bool)
        out[np.asarray(idx, dtype=np.int64)] = True
        return out

    in_h, in_tr = mask(splits["holdout_idx"]), mask(splits["train_idx"])
    if (in_h & in_tr).any() or not (in_h | in_tr).all():
        raise SystemExit("[decision] the base split's train and holdout rows do not partition the table")

    # the tier instance of each gamma: one dir per gamma, taken from the verdict's runs
    tier_dirs: dict[float, str] = {}
    for ck, c in verdict["cells"].items():
        arm, g, _ = ck.split("|")
        if arm in GEN_ARMS:
            if tier_dirs.setdefault(float(g[1:]), c["nuisance_dir"]) != c["nuisance_dir"]:
                raise SystemExit(f"[decision] gamma = {g[1:]} runs trained on more than one tier instance")
    kept, scored = {}, None
    for g, d in sorted(tier_dirs.items()):
        with open(os.path.join(d, "splits.json")) as fh:
            ts = json.load(fh)
        if float(ts["tier"]["gamma"]) != g:
            raise SystemExit(f"[decision] {d}: gamma {ts['tier']['gamma']} under a g{g:g} label")
        k = mask(ts["train_idx"])
        if (k & ~in_tr).any() or not np.array_equal(np.sort(np.asarray(ts["holdout_idx"])),
                                                    np.sort(splits["holdout_idx"])):
            raise SystemExit(f"[decision] {d} is not a thinning of the base split")
        sc = sorted(int(v) for v in ts["tier"]["scored_compounds"].values())
        if scored is not None and sc != scored:
            raise SystemExit("[decision] the tier instances thin different compounds")
        kept[g], scored = k, sc
    with open(cfg.paths.gene_order_json) as fh:
        symbols = [str(s) for s in json.load(fh)["pr_gene_symbol"]]
    if len(symbols) != y.shape[1]:
        raise SystemExit("[decision] gene_order.json and expr.npy disagree on the number of genes")
    return {"odoc": odoc, "oz": oz, "ak": ak, "N": N, "y": y, "ctl": ic == 1, "comp_t": comp_t,
            "line_t": line_t, "lines": lines, "arm_of_row": arm_of_row, "in_h": in_h, "in_tr": in_tr,
            "kept": kept, "scored": scored, "tier_dirs": tier_dirs, "symbols": symbols,
            "comp": np.array([int(k.split("|", 1)[0]) for k in ak], dtype=np.int64),
            "dose": np.array([float(k.split("|", 1)[1]) for k in ak], dtype=np.float64),
            "half": np.array([int(half_of[k]) for k in ak], dtype=np.int64),
            "name_of": dict(zip(comp_t.tolist(), meta["pert_iname"].astype(str).tolist())),
            "g2_lines": line_group_sign(np.array(lines), cfg.population) > 0}


def load_run_rows(npz_path: str, w: dict) -> tuple[np.ndarray, np.ndarray]:
    """One run's (row_id, row_mean), checked against the oracle's arm table and
    the table's rows."""
    z = np.load(npz_path, allow_pickle=False)
    if not np.array_equal(z["all/arm_key"].astype(str), w["ak"]):
        raise SystemExit(f"[decision] {npz_path}: its arm table differs from the oracle's")
    gp = npz_path[:-len("_tau.npz")] + "_gen.npz"
    if not os.path.isfile(gp):
        raise SystemExit(f"[decision] {gp} is missing: the run was scored without --save_gen")
    gz = np.load(gp, allow_pickle=False)
    rid = gz["row_id"].astype(np.int64)
    if rid.size != w["N"] or not np.array_equal(np.sort(rid), np.arange(w["N"])):
        raise SystemExit(f"[decision] {gp} does not hold every table row exactly once (--pool all)")
    return rid, gz["row_mean"]


def real_effects(w: dict) -> dict:
    """The effects of each real well set: holdout (truth), all wells (secondary
    truth), the unthinned train wells, and each tier's kept train wells."""
    K, L = w["ak"].size, len(w["lines"])
    sets = {"holdout": w["in_h"], "all_wells": np.ones(w["N"], dtype=bool), "train": w["in_tr"]}
    sets.update({f"kept|g{g:g}": k for g, k in w["kept"].items()})
    out = {}
    for name, msk in sets.items():
        out[name] = effects_from_rows(w["y"][msk], w["arm_of_row"][msk], w["line_t"][msk], w["ctl"][msk],
                                      K, L, g2_lines=w["g2_lines"] if name == "holdout" else None)
    return out


def build_frame(w: dict) -> dict:
    """Everything the decisions rest on that no generator enters: the real effects
    of every well set, the arms each set covers in every line, the eligible arms
    (covered by all of them) and the design. `main` and `check_decision` both
    build it here, so the checks cover what the job computes."""
    real = real_effects(w)
    err = float(np.nanmax(np.abs(real["all_wells"]["pooled"] - w["oz"]["all/tau"].astype(np.float64))))
    every_line = {name: (r["cnt"] > 0).all(axis=1) for name, r in real.items()}
    eligible = np.logical_and.reduce(list(every_line.values()))
    return {"real": real, "oracle_err": err, "every_line": every_line, "eligible": eligible,
            "design": build_design(w["comp"], w["dose"], eligible, w["scored"])}


def g2_share_shift(w: dict, real: dict, design: Design, g1: float) -> dict:
    """How the thinning moved the G2 share of the kept train wells, by dose half,
    over the thinned compounds' eligible arms; and the sign of high minus low at
    gamma > 0, which fixes the direction a line-blind estimator's dose moves."""
    arms = design.arm[design.thinned]
    arms = arms[arms >= 0]
    g2 = w["g2_lines"]

    def share(cnt):
        return cnt[arms][:, g2].sum(axis=1) / cnt[arms].sum(axis=1)
    base = share(real["train"]["cnt"])
    out = {}
    for g in sorted(w["kept"]):
        d = share(real[f"kept|g{g:g}"]["cnt"]) - base
        out[f"g{g:g}"] = {"low": float(d[w["half"][arms] == 0].mean()),
                          "high": float(d[w["half"][arms] == 1].mean())}
    out["tilt_sign"] = float(np.sign(out[f"g{g1:g}"]["high"] - out[f"g{g1:g}"]["low"]))
    return out


def axis_sanity(w: dict, b_all: np.ndarray) -> dict:
    """Where the panel's compounds sit in the all-wells ranking on the task-B axis
    (best dose, as a percentile over all compounds with a complete arm)."""
    from src.eval.evaluate import PANEL
    ok = np.isfinite(b_all)
    cs = np.unique(w["comp"][ok])
    best = np.array([b_all[ok & (w["comp"] == c)].max() for c in cs])
    pct = np.argsort(np.argsort(best)) / max(best.size - 1, 1)
    idx_of = {}
    for ci, nm in w["name_of"].items():
        idx_of.setdefault(nm, int(ci))
    rows, judged = [], []
    for family, names in PANEL:
        for nm in names:
            ci = idx_of.get(nm)
            j = np.flatnonzero(cs == ci) if ci is not None else np.zeros(0, dtype=int)
            if j.size:
                rows.append({"family": family, "compound": nm, "percentile": float(pct[j[0]]),
                             "best_dose_effect": float(best[j[0]])})
                if family in AXIS_SANITY_FAMILIES:
                    judged.append(float(pct[j[0]]))
    med = float(np.median(judged)) if judged else None
    return {"n_compounds_ranked": int(cs.size), "panel": rows, "families_judged": list(AXIS_SANITY_FAMILIES),
            "median_percentile": med, "needs_at_least": AXIS_SANITY_MIN_PCT,
            "pass": bool(med is not None and med >= AXIS_SANITY_MIN_PCT)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _f(d, nd=3) -> str:
    if d is None or d.get("value") is None:
        return "n/a"
    if d.get("unstable_denominator"):
        return f"{d['value']:+.{nd}f} †"
    return f"{d['value']:+.{nd}f} ± {d['se']:.{nd}f}" if d.get("se") is not None else f"{d['value']:+.{nd}f}"


def fmt_z(z) -> str:
    return "n/a" if z is None else f"{z:+.2f}"


def print_block(title: str, blk: dict, g1: float, arms) -> None:
    print(f"\n### {title}\n")
    print(f"gain of the full-data reference over a random choice: {_f(blk['reference_gain_over_random'])}\n")
    print(f"| policy | gain over random, γ = 0 | γ = {g1:g} | captured share, γ = 0 | γ = {g1:g} | seed sd, γ = {g1:g} |")
    print("|---|---|---|---|---|---|")
    P = blk["policies"]
    for arm in arms:
        a, b = P.get(pkey(arm, 0.0)), P.get(pkey(arm, g1))
        if a is None or b is None:
            continue
        sd = "n/a" if b["seed_sd"] is None else f"{b['seed_sd']:.3f}"
        print(f"| `{arm}` | {_f(a['gain_over_random'])} | {_f(b['gain_over_random'])} | "
              f"{_f(a['captured_share'])} | {_f(b['captured_share'])} | {sd} |")
    if not blk["captured_share_has_error"]:
        print(f"\n† no error is given: the reference's gain over random is within {SHARE_MIN_Z:g} SE of zero "
              f"(or not positive), so the share's denominator is unstable.")


def print_criteria(title: str, crit: dict) -> None:
    print(f"\n### {title}\n\n| # | quantity | value ± SE | z | result |\n|---|---|---|---|---|")
    for name, c in crit.items():
        if not isinstance(c, dict) or "quantity" not in c:
            continue
        if name.startswith("D5"):
            worst = max((abs(m["z"]) for m in c["arms"].values() if m["z"] is not None), default=0.0)
            print(f"| {name} | {c['quantity']} | largest \\|z\\| over the arms | {worst:.2f} | "
                  f"{'PASS' if c['pass'] else 'FAIL'} |")
            continue
        res = ("reported" if "pass" not in c else ("PASS" if c["pass"] else "FAIL") if c["pass"] is not None
               else "not judged")
        print(f"| {name} | {c['quantity']} | {_f(c)} | {fmt_z(c.get('z'))} | {res} |")
    if crit.get("task_testable") is False:
        print("\nD0 did not pass: the planted confounding does not move this decision in the raw data. "
              "The task is NOT TESTABLE (DECISION.md §6); D1-D3 are descriptive.")
    elif crit.get("D3_testable") is False:
        print("\nD2 did not pass: `dr` against `conditional` is NOT TESTABLE on this decision "
              "(DECISION.md §6); D3's quantity is reported, not judged.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verdict", default=None,
                   help="A step_a_report verdict on --pool all (default <runs>/eval_artifacts/step_a_verdict.json).")
    p.add_argument("--out", default="decision_task.json", help="Relative to <runs>/eval_artifacts/, or a path.")
    p.add_argument("--n_boot", type=int, default=2000, help="Compound bootstrap resamples (task B).")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    t0 = time.time()
    from src.data.build_dataset import _write_json
    from src.eval import dist_metrics as dm
    print(f"[blas] numpy matrix products verified (rel err {dm.assert_blas_ok():.1e})", flush=True)
    cfg = apply_paths_args(config_from_args(args), args)
    if line_groups(cfg.population) is None:
        raise SystemExit(f"[decision] population {cfg.population.name} declares no line groups")
    ea = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    verdict, vpath = read_verdict(cfg, args.verdict)
    w = load_world(cfg, verdict)
    K, L = w["ak"].size, len(w["lines"])
    gammas = sorted(w["kept"])
    if len(gammas) != 2 or gammas[0] != 0.0:
        raise SystemExit(f"[decision] need runs at gamma = 0 and one gamma > 0, found {gammas}")
    g1 = gammas[1]

    # ---- real effects, eligibility, the design, the axes ------------------------------
    frame = build_frame(w)
    real, every_line, eligible, design = (frame[k] for k in ("real", "every_line", "eligible", "design"))
    print(f"[decision] the all-wells effect at each arm's own line mix, rebuilt from rows = the "
          f"stored oracle (max |diff| {frame['oracle_err']:.1e})", flush=True)
    if not frame["oracle_err"] < 1e-3:
        raise SystemExit("[decision] the rebuilt oracle differs from the stored one: wrong frame")
    sets = design.sets()
    n_el = design.valid.sum(axis=1)
    print(f"[decision] {K:,} arms; a well in every one of the {L} lines: "
          + ", ".join(f"{name} {int(v.sum()):,}" for name, v in every_line.items())
          + f"; eligible (all of these) {int(eligible.sum()):,}", flush=True)
    for sname, sel in sets.items():
        print(f"[decision] {sname}: {int(sel.sum()):,} compounds enter (>= {DECISION_MIN_DOSES} eligible doses; "
              + ", ".join(f"{d} doses: {int((n_el[sel] == d).sum())}" for d in sorted(set(n_el[sel].tolist()), reverse=True))
              + ")", flush=True)
    if min(int(s.sum()) for s in sets.values()) < 2 * DECISION_K_CONTROL:
        raise SystemExit("[decision] too few compounds enter for the declared k")

    s_b = gene_axis(w["symbols"])
    S_a = compound_axes(real["holdout"]["eq"], design)

    def project(tau: np.ndarray) -> dict[str, np.ndarray]:
        """A per-arm effect on the two axes; NaN where the effect is undefined."""
        return {"A": dot(tau, S_a), "B": np.einsum("ag,g->a", tau, s_b)}

    truth = {t: project(real[t]["eq"]) for t in TRUTHS}
    con = project(real["holdout"]["contrast"])
    kappa = {t: np.nanmean(design.take(con[t]), axis=1) for t in ("A", "B")}   # per compound
    shift = g2_share_shift(w, real, design, g1)
    print("[decision] G2 share of the kept train wells against the unthinned ones, thinned compounds: "
          + "; ".join(f"γ = {g:g}: low {shift[f'g{g:g}']['low']:+.3f}, high {shift[f'g{g:g}']['high']:+.3f}"
                      for g in gammas), flush=True)
    sanity = axis_sanity(w, truth["all_wells"]["B"])
    print(f"[decision] task-B axis on the oracle: panel families {AXIS_SANITY_FAMILIES} sit at a median "
          f"percentile of {sanity['median_percentile']} (expected >= {AXIS_SANITY_MIN_PCT}): "
          f"{'ok' if sanity['pass'] else 'WARNING, the axis does not rank them high'}", flush=True)
    for r in sanity["panel"]:
        print(f"[decision]   {r['family']:10s} {r['compound']:14s} percentile {r['percentile']:.3f}")

    # ---- the real-data policies, and D0 before any generator is read ----------------------
    scores: dict[str, dict[str, list[np.ndarray]]] = {"A": {}, "B": {}}

    def add(name: str, tau: np.ndarray) -> None:
        for t, v in project(tau).items():
            scores[t].setdefault(name, []).append(v)

    add(REFERENCE, real["train"]["eq"])
    for g in gammas:
        add(pkey("pooled_real", g), real[f"kept|g{g:g}"]["pooled"])
        add(pkey("stratified_real", g), real[f"kept|g{g:g}"]["eq"])

    def run_all() -> dict:
        return {t: {task: analyse_task(scores[task], truth[t][task], design, screens=(task == "B"),
                                       kappa=kappa[task], tilt_sign=shift["tilt_sign"], g1=g1,
                                       n_boot=args.n_boot, seed=args.seed)
                    for task in ("A", "B")} for t in TRUTHS}

    pre = run_all()
    print("\n## D0, the power gate (real train wells only; no generator has been read)\n")
    for task in ("A", "B"):
        c = pre[TRUTHS[0]][task]["criteria"]["D0_power_gate"]
        print(f"- task {task}: the pooled real mean loses {_f(c)} from γ = 0 to γ = {g1:g} "
              f"(z = {fmt_z(c['z'])}; needs >= {D0_MIN_SE:g}): "
              f"{'PASS' if c['pass'] else 'FAIL -- this task is NOT TESTABLE; the generators are reported descriptively'}",
              flush=True)

    # ---- the generators -------------------------------------------------------------------
    seeds: dict[str, dict[float, list[int]]] = {}
    veh_offset = {}
    rv, n_veh = _group_sum(w["y"][w["ctl"]], w["line_t"][w["ctl"]], L)
    for ck in sorted(verdict["cells"]):
        arm, g, s = ck.split("|")
        if arm not in GEN_ARMS:
            continue
        rid, row_mean = load_run_rows(verdict["cells"][ck]["npz"], w)
        e = effects_from_rows(row_mean, w["arm_of_row"][rid], w["line_t"][rid], w["ctl"][rid], K, L)
        if not (e["cnt"] == real["all_wells"]["cnt"]).all():
            raise SystemExit(f"[decision] {ck}: its generated rows are not the table's rows, cell by cell")
        add(pkey(arm, float(g[1:])), e["eq"])
        seeds.setdefault(arm, {}).setdefault(float(g[1:]), []).append(int(s[1:]))
        # The generated vehicle mean against the real one, per line (the norm of the
        # difference over genes, averaged over lines). It cancels in every decision.
        ctl = w["ctl"][rid]
        gv, _ = _group_sum(row_mean[ctl], w["line_t"][rid][ctl], L)
        veh_offset[ck] = float(np.sqrt((((gv - rv) / n_veh[:, None]) ** 2).sum(axis=1)).mean())
        print(f"[decision] loaded {ck}", flush=True)
        del row_mean, e
    for arm, by_g in seeds.items():
        if sorted(by_g) != gammas or by_g[0.0] != by_g[g1]:
            raise SystemExit(f"[decision] `{arm}` needs the same seeds at both gammas, has {by_g}")
    # Seeds are paired by position across arms (the overlap of two top-k sets), so
    # the ADIGen arms must have the baseline's seeds.
    for arm in ADIGEN_ARMS:
        if arm in seeds and seeds[arm][g1] != seeds.get(BASELINE, {}).get(g1):
            raise SystemExit(f"[decision] `{arm}` has seeds {seeds[arm][g1]}, `{BASELINE}` "
                             f"{seeds.get(BASELINE, {}).get(g1)}; they must be the same")
    res = run_all()
    for t in TRUTHS:                                      # the real-data policies do not depend on the generators
        for task in ("A", "B"):
            a, b = pre[t][task]["criteria"]["D0_power_gate"], res[t][task]["criteria"]["D0_power_gate"]
            if a != b:
                raise SystemExit("[decision] D0 changed once the generators were read")

    out = {"population": cfg.population.name, "verdict": os.path.abspath(vpath), "oracle": verdict["truth"],
           "gamma": g1, "lines": w["lines"], "line_weights": "equal",
           "definition": {"effect": "tau(c, d) = mean over lines of [mu(c, d, line) - mu(0, line)]",
                          "task_A_axis": "s_c: unit direction of the mean over the compound's eligible doses "
                                         "of its holdout effect",
                          "task_B_axis": "-1/sqrt(m) on spec.PROLIFERATION_GENES",
                          "proliferation_genes": list(PROLIFERATION_GENES),
                          "value": "mean over compounds of the truth at the chosen dose",
                          "min_doses": DECISION_MIN_DOSES, "k": DECISION_K, "k_curve": list(DECISION_K_CURVE),
                          "k_control": DECISION_K_CONTROL, "n_boot": args.n_boot, "boot_seed": args.seed,
                          "errors": {"A": "delete-one-compound jackknife", "B": "compound bootstrap"}},
           "design": {"n_arms": int(K), "arms_with_a_well_in_every_line": {k: int(v.sum()) for k, v in every_line.items()},
                      "n_arms_eligible": int(eligible.sum()),
                      "compounds": {sname: {"enter": int(sel.sum()),
                                            "by_n_doses": {str(d): int((n_el[sel] == d).sum())
                                                           for d in sorted(set(n_el[sel].tolist()))}}
                                    for sname, sel in sets.items()}},
           "seeds": {arm: by_g[g1] for arm, by_g in seeds.items()},
           "g2_share_shift": shift, "axis_sanity": sanity,
           "generated_vehicle_offset_norm": veh_offset,
           "truths": res, "primary_truth": TRUTHS[0]}

    arms = REAL_ARMS + tuple(a for a in GEN_ARMS if a in seeds)
    for t in TRUTHS:
        tag = "PRIMARY" if t == TRUTHS[0] else "secondary; shares train wells with the estimators"
        print(f"\n## Truth: {t} ({tag})")
        A, B = res[t]["A"], res[t]["B"]
        print_block(f"Task A, dose along the compound's own signature — thinned compounds "
                    f"({A['dose']['thinned']['n_compounds']})", A["dose"]["thinned"], g1, arms)
        print_block(f"Task A — unthinned compounds, the control ({A['dose']['unthinned']['n_compounds']})",
                    A["dose"]["unthinned"], g1, arms)
        print_criteria(f"Task A criteria ({t})", A["criteria"])
        for k in sorted(B["screens"]["all"]["k"], key=int):
            print_block(f"Task B, top-{k} of {B['screens']['all']['n_candidates']} compounds on the "
                        f"proliferation axis", B["screens"]["all"]["k"][k], g1, arms)
        print_block("Task B, dose on the proliferation axis — thinned compounds (descriptive)",
                    B["dose"]["thinned"], g1, arms)
        print_criteria(f"Task B criteria ({t}; top-{DECISION_K} of all candidates, control top-"
                       f"{DECISION_K_CONTROL} of the unthinned)", B["criteria"])
    out["elapsed_sec"] = round(time.time() - t0, 1)
    op = args.out if os.path.isabs(args.out) or os.sep in args.out else os.path.join(ea, args.out)
    # Per compound: the truth at each policy's chosen dose (seeds averaged) and the
    # chosen dose's rank per seed. Not in the JSON.
    npz = {"compound_idx": design.comp, "thinned": design.thinned, "n_eligible_doses": n_el}
    for t in TRUTHS:
        for task in ("A", "B"):
            pc = res[t][task].pop("_per_compound")
            pre[t][task].pop("_per_compound")
            npz.update({f"{t}/{task}/value/{name}": v.astype(np.float32) for name, v in pc["value"].items()})
            if t == TRUTHS[0]:                            # the choices do not depend on the truth
                npz.update({f"{task}/dose_rank/{name}": r.astype(np.int8) for name, r in pc["dose_rank"].items()})
    _write_json(op, out)
    np.savez_compressed(os.path.splitext(op)[0] + "_values.npz", **npz)
    print(f"\n[decision] -> {op}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
