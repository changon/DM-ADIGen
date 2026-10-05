"""Read-back checks of P2's weight normalisation (`STEP_A.md` §4). No PyTorch.

`src/nuisances/weight_norm.group_normalize` on synthetic weights (group sums, the
exact cap, kept proportions, zeros, the refusals) and on the real counts and
design weights of both step-C tier instances: every group sums to its row count,
every unthinned and vehicle row keeps weight exactly 1, the cap holds, weights
stay constant within a positivity cell, and the weighted syn_c mix of each scored
group against the unthinned pool's (which the cap is allowed to disturb only
where it binds).

    python -m src.tests.check_p2 [--data_dir data/mcf7_24h]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk  # noqa: E402

from src.data.splits import load_splits, positivity_cells  # noqa: E402
from src.nuisances.knn_dr import ess  # noqa: E402
from src.nuisances.weight_norm import P2_CLIP, group_normalize, train_groups  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data_dir", default="data/mcf7_24h")
ARGS = ap.parse_args()
DD = ARGS.data_dir


def kish_ess(w):
    return ess(w) / len(w)

F = []
def check(ok, msg):
    print(("ok    " if ok else "FAIL  ") + msg); F.append(msg) if not ok else None

# ---- synthetic ----
rng = np.random.default_rng(0)
g = np.repeat(np.arange(50), rng.integers(1, 9, 50))
w = rng.gamma(0.5, 2.0, g.size) + 1e-3
o, st = group_normalize(w, g)
n = np.bincount(g); tot = np.bincount(g, weights=o)
check(np.abs(tot - n).max() < 1e-9 and abs(o.mean() - 1) < 1e-12, f"group sums == row counts (max err {np.abs(tot-n).max():.1e}); global mean 1")
r = o / w
check(all(np.ptp(r[g == k]) < 1e-9 * r[g == k].max() for k in range(50)), "uncapped: raw proportions kept within every group")
o5, s5 = group_normalize(w, g, cap=2.0)
tot5 = np.bincount(g, weights=o5)
check(o5.max() <= 2.0 + 1e-12 and np.abs(tot5 - n).max() < 1e-9, f"cap 2: max {o5.max():.6f} <= 2 exactly AND sums preserved (max err {np.abs(tot5-n).max():.1e}); {s5['n_capped']} capped in {s5['cap_passes']} passes")
free = o5 < 2.0 - 1e-9
rr = o5[free] / w[free]
check(all(np.ptp(rr[g[free] == k]) < 1e-9 * max(rr[g[free] == k].max(), 1e-300) for k in np.unique(g[free])), "cap: the uncapped rows of a group keep their raw proportions")
check(s5["n_capped"] > 0 and s5["cap_passes"] >= 1, "the cap binds on the synthetic draw (the test exercises it)")
oc, _ = group_normalize(np.full(g.size, 7.3), g, cap=5.0)
check(np.abs(oc - 1).max() < 1e-15, "constant raw weights -> 1 (to one ulp), whatever the cap")
o11, _ = group_normalize(np.ones(g.size, dtype=np.float32), g, cap=5.0)
check(np.array_equal(o11, np.ones(g.size)), "raw weight exactly 1 -> EXACTLY 1 (the unthinned compounds' case)")
o1, _ = group_normalize(w, np.zeros(g.size, int))
check(np.allclose(o1, w / w.mean(), rtol=1e-12), "one group == the global mean-1 normalisation")
oz, _ = group_normalize(np.array([0.0, 2.0, 2.0, 1.0]), np.array([0, 0, 0, 1]))
check(oz[0] == 0 and abs(oz[1:3].sum() - 3) < 1e-12 and oz[3] == 1, "a zero weight stays zero; its group still sums to n")
for bad, lbl, kw in ((np.array([0.0, 0.0, 1.0]), "an all-zero group", {}), (np.array([1.0, -1.0, 1.0]), "a negative weight", {}),
                     (np.array([1.0, 2.0, 1.0]), "cap < 1", {"cap": 0.5})):
    try:
        group_normalize(bad, np.array([0, 0, 1]), **kw); check(False, f"refuses {lbl}")
    except ValueError:
        check(True, f"refuses {lbl}")
for bad, cap_, lbl in ((np.array([1.0, 0.0, 0.0, 0.0]), 2.0, "an infeasible cap (1 positive row of 4, cap 2)"),
                       (np.array([2.0, 1.0, 0.5, 0.0]), 1.0, "an infeasible cap (3 positive rows of 4, cap 1)")):
    try:
        group_normalize(bad, np.zeros(4, int), cap=cap_); check(False, f"refuses {lbl}")
    except ValueError as e:
        check("infeasible" in str(e), f"refuses {lbl}")
of, sf = group_normalize(np.array([3.0, 1.0, 0.0, 0.0]), np.zeros(4, int), cap=2.0)
check(np.allclose(of, [2.0, 2.0, 0.0, 0.0]) and abs(of.sum() - 4) < 1e-12, "a FEASIBLE cap with zero rows is met exactly ([3,1,0,0], cap 2 -> [2,2,0,0])")
# adversarial: one huge weight per group needs several passes
gg = np.repeat(np.arange(20), 6); ww = np.tile(np.array([1000., 100., 10., 1., 1., 1.]), 20)
oa, sa = group_normalize(ww, gg, cap=1.5)
check(oa.max() <= 1.5 + 1e-12 and np.abs(np.bincount(gg, weights=oa) - 6).max() < 1e-9 and sa["cap_passes"] >= 2,
      f"cascading cap converges ({sa['cap_passes']} passes, max {oa.max():.4f})")

# ---- the real C2 tier instances ----
tb = (load_from_disk(f"{DD}/lincs_tabular")
      .select_columns(["compound_idx", "dose_level", "is_control", "syn_c"]).to_pandas())
comp = tb["compound_idx"].values.astype(np.int64); dl = tb["dose_level"].values.astype(float)
ic = tb["is_control"].values.astype(np.int8); sc = tb["syn_c"].values.astype(np.int64)
for G in (0, 1):
    T = f"{DD}/nuisances_tier_Csyn_c_k0_g{G}_s42"
    p = argparse.ArgumentParser(); add_adjustment_set_cli(p); add_paths_cli(p)
    a = p.parse_args(["--data_dir", DD, "--nuisance_dir", T, "--adjustment_set", "syn_c"])
    cfg = apply_paths_args(config_from_args(a), a)
    sp = load_splits(cfg); tr = np.asarray(sp["train_idx"]); tier = sp["tier"]
    grp = train_groups(cfg, tier["confounder"], tr)
    scored = np.isin(comp[tr], [int(x) for x in tier["scored_compounds"].values()]) & (ic[tr] == 0)
    nu = np.load(f"{T}/nu_rows.npy")
    cells_all = positivity_cells("syn_c", comp, dl, ic, sc)
    for wf in ("dr_weights_counts.npz", "dr_weights_design.npz"):
        z = np.load(f"{T}/{wf}"); assert np.array_equal(z["row_id"], tr)
        raw = z["w"].astype(np.float32)
        glob = (raw / raw.mean()).astype(np.float64)
        un, su = group_normalize(raw, grp)
        cp, s = group_normalize(raw, grp, cap=P2_CLIP)
        tag = f"g{G} {wf[11:-4]:6s}"
        check(su["max_group_sum_rel_err"] < 1e-9 and s["max_group_sum_rel_err"] < 1e-9, f"{tag}: {s['n_groups']:,} groups sum to their row counts (uncapped {su['max_group_sum_rel_err']:.1e}, capped {s['max_group_sum_rel_err']:.1e})")
        check(bool((cp[~scored] == 1.0).all()) and bool((un[~scored] == 1.0).all()), f"{tag}: every unthinned and vehicle row has weight EXACTLY 1 ({int((~scored).sum()):,} rows); under the global norm they sit at {glob[~scored].mean():.4f}")
        check(cp.max() <= P2_CLIP + 1e-12, f"{tag}: cap 5 holds exactly (max {cp.max():.4f}; uncapped max {un.max():.2f}, global-norm max {glob.max():.2f})")
        # within a positivity cell the raw weight is constant, so the cap must keep it constant
        ck = cells_all[tr]; u, inv = np.unique(ck.astype(str), return_inverse=True)
        mx = np.full(u.size, -np.inf); mn = np.full(u.size, np.inf)
        np.maximum.at(mx, inv, cp); np.minimum.at(mn, inv, cp)
        if "counts" in wf:
            check(float((mx - mn).max()) < 1e-9, f"{tag}: weights stay constant within a positivity cell after the cap")
        # confounder balance: weighted syn_c mix per scored group vs the unthinned pool's
        gu, gi = np.unique(grp.astype(str), return_inverse=True)
        gall = np.array([c if c == "0|ctl" else c.rsplit("|", 1)[0] for c in cells_all.astype(str)])
        pos = {k: i for i, k in enumerate(gu)}
        nu_t = nu[ic[nu] == 0]
        gi_nu = np.array([pos.get(k, -1) for k in gall[nu_t]]); ok = gi_nu >= 0
        tgt = np.bincount(gi_nu[ok], weights=sc[nu_t][ok], minlength=gu.size) / np.maximum(np.bincount(gi_nu[ok], minlength=gu.size), 1)
        sg = np.zeros(gu.size, bool); sg[np.unique(gi[scored])] = True
        def mix(wv): return np.bincount(gi, weights=wv * sc[tr], minlength=gu.size) / np.bincount(gi, weights=wv, minlength=gu.size)
        e_un, e_cp, e_gl, e_no = (np.abs(mix(x) - tgt)[sg] for x in (un, cp, glob, np.ones(tr.size)))
        print(f"      {tag}: |weighted syn_c mix - unthinned mix| over {int(sg.sum())} scored groups, mean: unweighted {e_no.mean():.4f}, global {e_gl.mean():.4f}, "
              f"group {e_un.mean():.4f}, group+cap5 {e_cp.mean():.4f} (max {e_cp.max():.3f}); capped rows {s['n_capped']} ({100*s['capped_frac']:.2f}%) in {s['n_groups_with_cap']} groups")
        print(f"      {tag}: ESS/n global {kish_ess(glob):.3f} -> group {su['ess_over_n']:.3f} -> group+cap5 {s['ess_over_n']:.3f}; "
              f"share of total weight on scored rows: unweighted {scored.mean():.3f}, global {glob[scored].sum()/glob.sum():.3f}, group {cp[scored].sum()/cp.sum():.3f}")
print("\n" + ("ALL CHECKS PASSED" if not F else f"{len(F)} FAILURE(S)"))
sys.exit(1 if F else 0)
