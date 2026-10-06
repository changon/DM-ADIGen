"""The decisions a policy chooses among: every (compound, line) context of the
table with every dose level of its compound. Shared by the rollout driver
(torch) and the policy module (numpy), so the two can never lay targets out
differently. No PyTorch."""
from __future__ import annotations

import numpy as np

ROLLOUTS_NAME = "rollouts.npz"


def decision_targets(meta) -> dict:
    """Every (compound, line) context of the table with every dose level of its
    compound. `meta` is a pandas frame with compound_idx, dose_level, log10_conc,
    is_control, cell_id. Returns arrays aligned to the targets and the contexts."""
    trt = ~meta["is_control"].values.astype(bool)
    comp = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    lx = meta["log10_conc"].values.astype(np.float64)
    line = meta["cell_id"].values.astype(str)
    # the log10 concentration of each (compound, level): one value per level
    lv: dict[tuple, list] = {}
    for c, d, x in zip(comp[trt], dl[trt], lx[trt]):
        lv.setdefault((int(c), float(d)), []).append(float(x))
    # A dose level snaps concentrations within spec.ActionSpec.dose_level_tol_log10
    # (0.05) of a grid value, so a level's log10_conc can vary by that much; the
    # level is sampled at its median.
    levels = {}
    for (c, d), xs in lv.items():
        if max(xs) - min(xs) > 0.11:
            raise RuntimeError(f"compound {c} dose level {d}: log10_conc varies ({min(xs)}..{max(xs)})")
        levels.setdefault(c, []).append((d, float(np.median(xs))))
    for c in levels:
        levels[c].sort()
    ctx_keys = sorted({(int(c), str(l)) for c, l in zip(comp[trt], line[trt])})
    rep_row = {}
    for i in np.flatnonzero(trt):
        k = (int(comp[i]), str(line[i]))
        if k not in rep_row:
            rep_row[k] = int(i)
    t_ctx, t_dose, t_lx, t_di = [], [], [], []
    for ci, (c, l) in enumerate(ctx_keys):
        for di, (d, x) in enumerate(levels[c]):
            t_ctx.append(ci); t_dose.append(d); t_lx.append(x); t_di.append(di)
    return {"ctx_compound": np.array([c for c, _ in ctx_keys], dtype=np.int64),
            "ctx_line": np.array([l for _, l in ctx_keys]),
            "ctx_row": np.array([rep_row[k] for k in ctx_keys], dtype=np.int64),
            "t_ctx": np.array(t_ctx, dtype=np.int64), "t_dose_index": np.array(t_di, dtype=np.int64),
            "t_dose_level": np.array(t_dose, dtype=np.float64), "t_log10_conc": np.array(t_lx, dtype=np.float64)}


