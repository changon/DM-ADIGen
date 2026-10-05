"""P2: group-normalised, capped DR weights for the weighted risk (`STEP_A.md` §4). No PyTorch.

The weighted risk of `train_diffusion --dr_mode weighted` normalises its weights
GLOBALLY to mean 1. Step C2 showed what that costs (IMPLEMENT.md §5, "Step C2
results"): the weights move gradient mass BETWEEN groups as well as rebalancing
the confounder inside them, so compounds that were never thinned (raw weight
exactly 1) are down-weighted, and the learned share of the confounder's
interaction on them fell from 0.72 to 0.14-0.21.

`--dr_weight_norm group` normalises within each **target group** instead: the
positivity cell with the confounder dropped (`splits.target_groups`; (compound,
dose half) under `syn_c`, the arm under `cell_id`). Then

    sum_{i in g} w_i = n_train(g)      for every group g,

so a group receives exactly the mass it would receive unweighted, the weights can
only rebalance the confounder WITHIN a group, and a group whose raw weights are
constant (every unthinned compound, the vehicles) keeps weight exactly 1.

`--dr_weight_clip CAP` caps each normalised weight at CAP with the group total
PRESERVED: capped rows sit at CAP and the group's other rows are rescaled to
carry the rest, repeated until no row exceeds CAP (water-filling). So the cap
holds exactly, not just before a final renormalisation. CAP >= 1 is always
feasible, because a group's mean weight is 1.

The groups are derived on the WHOLE table and then indexed by the train rows:
`dose_half` ranks a compound's distinct dose levels, so deriving it on a subset
that lost a level would shift that compound's halves (the bug `dr_target` had,
IMPLEMENT.md §5, "P1 implementation and review").
"""
from __future__ import annotations

import numpy as np

from src.nuisances.knn_dr import ess

NORM_MODES = ("global", "group")
# THE cap of the P2 arm (`dr_p2`), fixed on 2026-10-05 before any P2 run
# (STEP_A.md §4). The launcher, the smoke and the report all read it from here.
P2_CLIP = 5.0


def group_normalize(w: np.ndarray, groups: np.ndarray, cap: float | None = None,
                    max_iter: int = 1000) -> tuple[np.ndarray, dict]:
    """(w_out float64, stats): `w` rescaled so every group sums to its row count,
    with each weight <= `cap` if a cap is given.

    Within a group the UNCAPPED rows keep their raw proportions; zeros stay zero.
    Raises if a group's weights are all zero (it would receive no gradient where
    the unweighted risk gives it n_g rows' worth), if cap < 1, or if the cap is
    INFEASIBLE for a group: with zero-weight rows, a group's positive rows may
    be too few to carry n_g at `cap` each (n_positive * cap < n_g).
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != np.asarray(groups).shape[0]:
        raise ValueError(f"w {w.shape} and groups {np.asarray(groups).shape} must be 1-d and aligned")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and >= 0")
    if cap is not None and not float(cap) >= 1.0:
        raise ValueError(f"cap={cap} must be >= 1: a group's mean weight is 1")
    uniq, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    n = np.bincount(inv, minlength=uniq.size).astype(np.float64)
    s = np.bincount(inv, weights=w, minlength=uniq.size)
    if (s <= 0).any():
        bad = uniq[s <= 0][:5]
        raise ValueError(f"{int((s <= 0).sum())} group(s) have zero total weight, e.g. {list(bad)}")

    capped = np.zeros(w.shape[0], dtype=bool)
    out = w * (n / s)[inv]
    n_iter = 0
    if cap is not None:
        cap = float(cap)
        for n_iter in range(1, max_iter + 1):
            newly = ~capped & (out > cap * (1.0 + 1e-12))
            if not newly.any():
                n_iter -= 1
                break
            capped |= newly
            n_cap = np.bincount(inv, weights=capped, minlength=uniq.size)
            s_free = np.bincount(inv, weights=np.where(capped, 0.0, w), minlength=uniq.size)
            # A group's free rows carry what the capped ones do not. If every
            # POSITIVE row of a group is capped and mass is still owed, the cap
            # cannot be met: that needs zero-weight rows (without them the free
            # rows' mean is < 1 <= cap, so a group is never capped entirely).
            owed = n - cap * n_cap
            stuck = (s_free <= 0) & (owed > 1e-9 * n)
            if stuck.any():
                raise ValueError(
                    f"cap={cap} is infeasible for {int(stuck.sum())} group(s), e.g. "
                    f"{list(uniq[stuck][:5])}: their positive-weight rows are too few to "
                    f"carry the group's row count at the cap. Raise the cap or drop it.")
            scale = np.divide(owed, s_free, out=np.ones_like(s_free), where=s_free > 0)
            out = np.where(capped, cap, w * scale[inv])
        else:
            raise RuntimeError(f"the cap did not converge in {max_iter} passes")

    tot = np.bincount(inv, weights=out, minlength=uniq.size)
    stats = {
        "n_groups": int(uniq.size),
        "max_group_sum_rel_err": float(np.abs(tot / n - 1.0).max()),
        "cap": cap, "n_capped": int(capped.sum()),
        "capped_frac": float(capped.mean()),
        "n_groups_with_cap": int(np.unique(inv[capped]).size),
        "cap_passes": int(n_iter),
        "max": float(out.max()), "min": float(out.min()),
        "mean": float(out.mean()),
        "n_exactly_one": int((out == 1.0).sum()),
        "ess_over_n": ess(out) / max(out.size, 1),
    }
    return out, stats


def train_groups(cfg, confounder: str, train_idx: np.ndarray) -> np.ndarray:
    """(n_train,) the target group of each train row, derived on the whole table."""
    from datasets import load_from_disk

    from src.data.splits import target_groups
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control"]).to_pandas())
    g = target_groups(confounder,
                      meta["compound_idx"].values.astype(np.int64),
                      meta["dose_level"].values.astype(np.float64),
                      meta["is_control"].values.astype(np.int8))
    return g[np.asarray(train_idx, dtype=np.int64)]
