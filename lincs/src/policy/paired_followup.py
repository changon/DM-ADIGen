"""Follow-up to `src.policy.learn --stage report` (IMPLEMENT.md §5, "Policy learning on
step A: results"): paired differences between arms and the model error by dose
half. Written after the runs and NOT reviewed; it only reads the rollouts and the
stage files. Run from lincs/ on a CPU node (about 4 GB, 80 s):
    python -m src.policy.paired_followup [--data_dir data/core5_24h_inj-m2] [--gamma 3]
"""
import argparse, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, ".")
from src.policy import learn as pl
from src.policy.values import load_values
from src.spec import apply_paths_args, config_from_args
_ap = argparse.ArgumentParser(); _ap.add_argument("--data_dir", default="data/core5_24h"); _ap.add_argument("--gamma", type=float, default=3.0)
_a = _ap.parse_args()
D = _a.data_dir; PY = sys.executable; E = f"runs/{os.path.basename(os.path.normpath(D))}/eval_artifacts/policy/"
G1 = _a.gamma
def tier(g):
    return subprocess.check_output([PY, "-m", "src.data.build_tiered_split", "--data_dir", D, "--confounder", "cell_id",
                                    "--gamma", str(g), "--seed", "42", "--positivity_key", "half", "--print_out_dir"]).decode().strip()
ns = argparse.Namespace(data_dir=D, nuisance_dir=None, adjustment_set="cell_id", population=None, population_compounds=None, environment_set=None)
cfg = apply_paths_args(config_from_args(ns), ns)
ut = "A"
rolls = pl.find_rollouts(cfg.paths.train_output_dir, "pl_*")
def f(e): return f"{e.value:+.3f} ± {e.se:.3f} (z {e.z:+.2f})" if e.se else f"{e.value:+.3f}"
OUT = {}
for g in (G1, 0):
    F = pl.load_frame(cfg, tier(g), 0); S = pl.Sets(F)
    valid, mu_h = F["valid"], F["mu"]["holdout"][ut]
    V = load_values(E + f"values_g{g:g}.npz")
    R = dict(json.load(open(E + f"round1_g{g:g}.json"))[ut]["generators"])
    if os.path.isfile(E + f"round2_g{g:g}.json"):
        R.update(json.load(open(E + f"round2_g{g:g}.json"))[ut]["generators"])
    thin = (F["half"] == F["thin_half"][:, None]) & valid
    fav = valid & ~thin
    ref = F["mu"]["unthinned"][ut]; ref = np.where(np.isfinite(ref), ref, np.nanmean(ref, axis=1, keepdims=True))
    ref_best_thin = thin[np.arange(F["n_ctx"]), np.argmax(np.where(valid, ref, -np.inf), axis=1)]
    test = S.sel["test"]
    subsets = {"test": test, "test, reference optimum in the THIN half": test & ref_best_thin,
               "test, reference optimum in the FAVOURED half": test & ~ref_best_thin, "control": S.sel["control"]}
    def E_(v, sel):
        return pl.est(v, F["comp"], sel, np.unique(F["comp"][sel]))
    # per-run arrays
    A = {}
    for run, rec in R.items():
        mu = pl.mu_from_rollouts(rolls[run], F)[0][ut]
        pi = pl.tilt(mu, F["pi_b"], rec["beta"], valid)
        err = np.where(valid, (mu - mu_h) ** 2, 0.0)
        A[run] = {"v": pl.value_per_context(pi, mu_h, valid), "err_pi": (pi * err).sum(1),
                  "err_thin": np.where(thin, err, 0).sum(1) / np.maximum(thin.sum(1), 1),
                  "err_fav": np.where(fav, err, 0).sum(1) / np.maximum(fav.sum(1), 1),
                  "mass_thin": np.where(thin, pi, 0).sum(1), "selfeval": pl.value_per_context(pi, mu, valid) - pl.value_per_context(pi, mu_h, valid),
                  # the signed error on the thin half relative to the favoured half: an optimism tilt
                  "bias_thin": np.where(thin, mu - mu_h, 0).sum(1) / np.maximum(thin.sum(1), 1) - np.where(fav, mu - mu_h, 0).sum(1) / np.maximum(fav.sum(1), 1)}
        assert np.allclose(A[run]["v"], V[f"{ut}/{run}"], atol=1e-3, equal_nan=True), run
    def arm(name, key):
        runs = [r for r in A if (pl.parse_run(r, "pl")["arm"], pl.parse_run(r, "pl")["lam"]) == name]
        if not runs:
            return None
        return np.mean([A[r][key] for r in sorted(runs)], axis=0)
    ARMS = {"naive": ("naive", None), "conditional": ("conditional", None), "dr": ("dr", None),
            "rt λ=0.25": ("rt", 0.25), "rt λ=0.5": ("rt", 0.5), "rt λ=0.75": ("rt", 0.75), "rt λ=1": ("rt", 1.0)}
    print(f"\n################ gamma = {g}: {int(test.sum())} test contexts, {int((test & ref_best_thin).sum())} with the reference optimum in the thin half")
    print("\n-- arms, halves averaged (test): value | mass on thin half | model error: thin-half doses | favoured-half doses | at the policy | self-evaluation")
    for a, key in ARMS.items():
        if arm(key, 'v') is None:
            continue
        print(f"  {a:12s} {f(E_(arm(key,'v'), test)):28s} {arm(key,'mass_thin')[test].mean():.3f}  thin {E_(arm(key,'err_thin'), test).value:7.2f}  fav {E_(arm(key,'err_fav'), test).value:7.2f}  @pi {E_(arm(key,'err_pi'), test).value:7.2f}  selfeval {E_(arm(key,'selfeval'), test).value:+.2f}  thin-minus-fav signed error {f(E_(arm(key,'bias_thin'), test))}")
    PAIRS = [("dr", "conditional"), ("rt λ=0.25", "conditional"), ("rt λ=0.25", "dr"), ("rt λ=1", "rt λ=0.25"), ("naive", "conditional")]
    for key, lab in (("v", "VALUE"), ("err_thin", "MODEL ERROR on thin-half doses"), ("err_fav", "MODEL ERROR on favoured-half doses"), ("err_pi", "MODEL ERROR at the policy")):
        print(f"\n-- paired differences, {lab}")
        for sn, sel in subsets.items():
            if key != "v" and sn != "test": continue
            print(f"   [{sn}; {int(sel.sum())} contexts]")
            for a, b in PAIRS:
                if arm(ARMS[a], key) is None or arm(ARMS[b], key) is None:
                    continue
                d = E_(arm(ARMS[a], key) - arm(ARMS[b], key), sel)
                OUT[(g, key, sn, a, b)] = d
                print(f"     {a:10s} - {b:12s} {f(d)}")
            if key == "v":
                for real in ("kept_pooled", "unthinned", "kept_strat"):
                    d = E_(arm(ARMS["conditional"], "v") - V[f"{ut}/{real}"], sel)
                    print(f"     conditional - {real:12s} {f(d)}")
    OUT[g] = {a: {k: arm(key, k) for k in ("v", "err_thin", "err_fav", "err_pi")} for a, key in ARMS.items() if arm(key, "v") is not None}
    OUT[(g, "F")] = (F, S, test)
# difference-in-differences across instances (same contexts and roles)
F, S, test = OUT[(G1, "F")]
print("\n################ gamma 3 minus gamma 0, paired by compound (test)")
for key, lab in (("v", "value"), ("err_thin", "model error, thin-half doses"), ("err_fav", "model error, favoured-half doses")):
    print(f"-- {lab}")
    for a in ("naive", "conditional", "dr", "rt λ=0.25"):
        if a not in OUT[G1] or a not in OUT[0]:
            continue
        d = pl.est(OUT[G1][a][key] - OUT[0][a][key], F["comp"], test, np.unique(F["comp"][test]))
        print(f"     {a:12s} {f(d)}")
    for a, b in (("dr", "conditional"), ("rt λ=0.25", "conditional"), ("rt λ=0.25", "dr")):
        if any(x not in OUT[G1] or x not in OUT[0] for x in (a, b)):
            continue
        d = pl.est((OUT[G1][a][key] - OUT[G1][b][key]) - (OUT[0][a][key] - OUT[0][b][key]), F["comp"], test, np.unique(F["comp"][test]))
        print(f"     DiD ({a} - {b}) {f(d)}")
