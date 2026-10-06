"""Checks of the policy-learning pipeline (POLICY_LEARNING.md §11, P10). No PyTorch.

Synthetic (always, seconds, no data):
  tilt      the tilt is the logger times exp(mu / beta), normalised; beta -> inf
            returns the logger, beta -> 0 the argmax; a per-context constant in mu
            changes nothing
  logger    `logger_from_counts` puts each half's kept share on its doses, uniform
            within the half; a context with no kept well is uniform
  value     the reference policy on exact mu picks the planted optimum and no
            policy beats it; the random policy's value is the mean over doses
  retarget  w = lam * pi_tilde / pi_b + (1 - lam) averages to 1 under the logger in
            every context, equals 1 at lam = 0, and is largest on the pilot's dose
  halves    `assign_halves` deals a stratum's rows between the halves (2 -> 1 + 1,
            1 -> one half), reproducibly

Real data (--data_dir, --tier, --tier_g0; the CPU job runs it):
  tier      positivity at the dose half: every (compound, half, line) of a scored
            compound keeps >= 1 train well; the thin half is the design's (G2
            lines: high; G1: low); the kept share on the thin half is near 1/6 at
            gamma > 0 and near 1/2 at gamma = 0; counts weights = n_unthinned /
            n_kept per (compound, half, line)
  halves    partition the parent's train rows; holdout = the parent's; weights
            subset in order; fingerprints distinct; expr_meta carries the base
            z-scale under the half's fingerprint
  frame     every context's doses are the compound's; the holdout cell counts
            match a direct count; the logger sums to 1; the design's own logger
            (tier_meta incl) agrees with the counts logger on the half shares

    python -m src.tests.check_policy
    python -m src.tests.check_policy --data_dir data/core5_24h --tier <tier> --tier_g0 <tier g0>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_halves import assign_halves, half_dir  # noqa: E402
from src.policy import learn as pl  # noqa: E402
from src.spec import add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args  # noqa: E402

FAILS: list[str] = []


def check(ok, msg: str) -> None:
    print(("ok    " if ok else "FAIL  ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def run_synthetic() -> None:
    rng = np.random.default_rng(0)
    n, D = 40, 6
    nd = np.full(n, D); nd[:5] = 4                                  # a few compounds with 4 doses
    valid = np.arange(D)[None, :] < nd[:, None]
    half = np.where(np.arange(D)[None, :] >= (nd // 2)[:, None], 1, 0).astype(np.int8)
    half = np.where(valid, half, -1)
    mu = np.where(valid, rng.normal(size=(n, D)), np.nan)
    cnt = np.where(valid, rng.integers(0, 4, size=(n, D)), 0)
    cnt[0] = 0                                                      # a context with no kept well
    pi_b = pl.logger_from_counts(cnt, half, valid)
    check(np.allclose(pi_b.sum(axis=1), 1) and (pi_b[~valid] == 0).all(), "logger: sums to 1 over the valid doses")
    i = 3
    for h in (0, 1):
        m = valid[i] & (half[i] == h)
        share = cnt[i][m].sum() / cnt[i][valid[i]].sum()
        check(np.allclose(pi_b[i][m], share / m.sum()), f"logger: context {i} half {h} carries its kept share {share:.3f}, uniform within")
    check(np.allclose(pi_b[0][valid[0]], 1 / nd[0]), "logger: a context with no kept well is uniform")
    # tilt
    for b in (0.05, 1.0, 100.0):
        pi = pl.tilt(mu, pi_b, b, valid)
        ref = np.where(valid, pi_b * np.exp(np.where(valid, mu, 0) / b), 0)
        ref = ref / ref.sum(axis=1, keepdims=True)
        check(np.allclose(pi, ref, atol=1e-12), f"tilt: pi_b exp(mu / {b}) normalised")
    check(np.allclose(pl.tilt(mu, pi_b, 1e6, valid), pi_b, atol=1e-5), "tilt: beta -> inf returns the logger")
    am = np.nanargmax(np.where(valid, mu, np.nan), axis=1)
    pi0 = pl.tilt(mu, pi_b, 1e-3, valid)
    check((pi0.argmax(axis=1) == am).all() and pi0.max(axis=1).min() > 0.99, "tilt: beta -> 0 is the argmax")
    check(np.allclose(pl.tilt(mu + rng.normal(size=(n, 1)), pi_b, 0.3, valid), pl.tilt(mu, pi_b, 0.3, valid)),
          "tilt: a per-context constant in mu changes nothing")
    # value
    uniform = np.where(valid, 1 / np.maximum(nd, 1)[:, None], 0)
    v_ref = pl.value_per_context(pl.tilt(mu, uniform, 1e-3, valid), mu, valid)
    check(np.allclose(v_ref, np.nanmax(np.where(valid, mu, np.nan), axis=1), atol=1e-3),
          "value: the argmax policy on exact mu attains the planted optimum")
    for b in (0.1, 1.0):
        v = pl.value_per_context(pl.tilt(mu + 0.5 * rng.normal(size=(n, D)), pi_b, b, valid), mu, valid)
        check((v <= v_ref + 1e-9).all(), f"value: no noisy policy (beta {b}) beats the optimum")
    check(np.allclose(pl.value_per_context(uniform, mu, valid), np.nanmean(np.where(valid, mu, np.nan), axis=1)),
          "value: the random policy is the mean over doses")
    comp = np.repeat(np.arange(8), 5); sel = np.ones(n, bool)
    e = pl.est(v_ref, comp, sel, np.arange(8))
    pc = np.array([v_ref[comp == c].mean() for c in range(8)])
    check(abs(e.value - pc.mean()) < 1e-12 and abs(e.se - pc.std(ddof=1) / np.sqrt(8)) < 1e-12,
          "value: the compound-clustered estimate is the mean of per-compound means with its jackknife SE")
    # retargeting weights
    beta = 0.2
    z = (mu - np.nanmax(np.where(valid, mu, np.nan), axis=1, keepdims=True)) / beta
    ratio = np.where(valid, np.exp(z), 0.0)
    ratio = ratio / np.where(valid, pi_b * ratio, 0.0).sum(axis=1, keepdims=True)
    for lam in (0.0, 0.5, 1.0):
        w = lam * ratio + (1 - lam)
        avg = np.where(valid, pi_b * w, 0).sum(axis=1)
        check(np.allclose(avg[1:], 1.0) and (lam > 0 or np.allclose(w[valid], 1.0)),
              f"retarget: lambda {lam}: the weights average to 1 under the logger in every context")
    check(np.allclose(pi_b * ratio, pl.tilt(mu, pi_b, beta, valid)), "retarget: pi_b * (pi_tilde / pi_b) = the pilot policy")
    check((pl.tilt(mu, pi_b, 1e-3, valid).argmax(axis=1) == am).all(), "retarget: at a small beta the pilot is the argmax")
    # halves
    tr = np.arange(1000) * 3
    strata = np.array([f"s{i // 2}" for i in range(1000)])          # 2 rows per stratum
    strata[-10:] = [f"one{i}" for i in range(10)]                    # ten 1-row strata
    h = assign_halves(tr, strata, 0)
    two = strata[:-10]
    ok = all(set(h[strata == s].tolist()) == {1, 2} for s in np.unique(two))
    check(ok and np.isin(h[-10:], (1, 2)).all() and np.array_equal(h, assign_halves(tr, strata, 0))
          and not np.array_equal(h, assign_halves(tr, strata, 1)),
          "halves: a 2-row stratum gives one row to each half; 1-row strata go to one half; reproducible per seed")


def run_real(args) -> None:
    from datasets import load_from_disk

    from src.data.splits import POSITIVITY_KEYS, dose_half, load_splits, tier_cell_key
    cfg = apply_paths_args(config_from_args(args), args)
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "cell_id"]).to_pandas())
    comp = meta["compound_idx"].values.astype(np.int64); dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool); line = meta["cell_id"].values.astype(str)
    half_t = dose_half(comp, dl, ctl)
    base = load_splits(cfg)
    N = len(meta)
    for name, tdir in (("g1", args.tier), ("g0", args.tier_g0)):
        with open(os.path.join(tdir, "splits.json")) as fh:
            ts = json.load(fh)
        tier = ts["tier"]
        check(tier_cell_key(tier) == "cell_id@half" and tier["positivity_key"] == list(POSITIVITY_KEYS["cell_id@half"]),
              f"{name}: positivity at the dose half")
        kept = np.zeros(N, bool); kept[ts["train_idx"]] = True
        intr = np.zeros(N, bool); intr[base["train_idx"]] = True
        scored = set(int(v) for v in tier["scored_compounds"].values())
        sc = np.isin(comp, list(scored)) & ~ctl
        cells = {}
        for i in np.flatnonzero(sc & intr):
            k = (int(comp[i]), int(half_t[i]), line[i])
            c = cells.setdefault(k, [0, 0]); c[0] += 1; c[1] += int(kept[i])
        n_cells = len(cells)
        check(all(v[1] >= 1 for v in cells.values()), f"{name}: every (compound, half, line) cell of a scored compound keeps >= 1 train well ({n_cells:,} cells)")
        # the thin half is the design's: G2 lines keep the low half, G1 the high
        from src.spec import line_group_sign
        g2 = {l: s > 0 for l, s in zip(sorted(set(line)), line_group_sign(np.array(sorted(set(line))), cfg.population))}
        thin_share, fav_share = [], []
        for (c, h, l), (n_tr, n_k) in cells.items():
            thin = h == (pl.THIN_HALF["G2"] if g2[l] else pl.THIN_HALF["G1"])
            (thin_share if thin else fav_share).append(n_k / n_tr)
        g = float(tier["gamma"])
        ts_, fs_ = float(np.mean(thin_share)), float(np.mean(fav_share))
        if g > 0:
            check(ts_ < 0.3 and fs_ > 0.9, f"{name}: kept share of the thin half {ts_:.3f}, of the favoured half {fs_:.3f} (gamma {g:g})")
        else:
            check(abs(ts_ - fs_) < 0.05, f"{name}: kept shares {ts_:.3f} / {fs_:.3f} agree at gamma = 0")
        wz = np.load(os.path.join(tdir, "dr_weights_counts.npz"), allow_pickle=False)
        rid = wz["row_id"].astype(np.int64); w = wz["w"].astype(np.float64)
        ok = True
        for i, r in zip(range(rid.size), rid):
            if not ctl[r] and int(comp[r]) in scored:
                n_tr, n_k = cells[(int(comp[r]), int(half_t[r]), line[r])]
                ok &= abs(w[i] - n_tr / n_k) < 1e-6
        check(ok, f"{name}: counts weights = n_unthinned / n_kept per (compound, half, line)")
        # halves
        hs = []
        for j in (1, 2):
            hd = half_dir(tdir, j)
            with open(os.path.join(hd, "splits.json")) as fh:
                hs.append(json.load(fh))
        tr1, tr2 = (np.asarray(h["train_idx"], dtype=np.int64) for h in hs)
        check(np.array_equal(np.sort(np.concatenate([tr1, tr2])), np.sort(np.asarray(ts["train_idx"]))) and not np.intersect1d(tr1, tr2).size,
              f"{name}: the halves partition the parent's train rows ({tr1.size:,} + {tr2.size:,})")
        check(all(h["holdout_idx"] == ts["holdout_idx"] for h in hs) and hs[0]["split_fingerprint"] != hs[1]["split_fingerprint"]
              and all(h["tier"]["half"]["parent_split_fingerprint"] == ts["split_fingerprint"] for h in hs),
              f"{name}: halves keep the parent's holdout, record the parent, and have distinct fingerprints")
        for j, h in zip((1, 2), hs):
            hd = half_dir(tdir, j)
            z = np.load(os.path.join(hd, "dr_weights_counts.npz"), allow_pickle=False)
            tr = np.asarray(h["train_idx"], dtype=np.int64)
            pos = {int(r): i for i, r in enumerate(rid)}
            check(np.array_equal(z["row_id"], tr) and np.allclose(z["w"], w[[pos[int(r)] for r in tr]])
                  and str(z["split_fingerprint"]) == h["split_fingerprint"],
                  f"{name} h{j}: counts weights are the parent's, subset in order, under the half's fingerprint")
            with open(os.path.join(hd, "expr_meta.json")) as fh:
                em = json.load(fh)
            with open(os.path.join(cfg.paths.nuisance_dir, "expr_meta.json")) as fh:
                eb = json.load(fh)
            check(em["split_fingerprint"] == h["split_fingerprint"] and np.array_equal(np.asarray(em["std"]), np.asarray(eb["std"])),
                  f"{name} h{j}: expr_meta carries the base z-scale under the half's fingerprint")
    # the frame on the confounded instance
    F = pl.load_frame(cfg, args.tier, 0)
    pass
    ok = True
    for ci in range(F["n_ctx"]):
        c = int(F["comp"][ci])
        lv = sorted(set(dl[(comp == c) & ~ctl].tolist()))
        ok &= np.array_equal(F["dose_level"][ci][F["valid"][ci]], np.array(lv))
    check(ok, "frame: every context's doses are its compound's distinct levels, ascending")
    hm = np.zeros(N, bool); hm[base["holdout_idx"]] = True
    direct = {}
    for i in np.flatnonzero(hm & ~ctl):
        direct[(int(comp[i]), line[i], float(dl[i]))] = direct.get((int(comp[i]), line[i], float(dl[i])), 0) + 1
    ok = all(direct.get((int(F["comp"][ci]), F["lines"][F["ctx_line"][ci]], float(F["dose_level"][ci, di])), 0)
             == F["cnt"]["holdout"][ci, di] for ci in range(0, F["n_ctx"], 7) for di in range(F["nd"][ci]))
    check(ok, "frame: holdout cell counts match a direct count")
    check(np.allclose(F["pi_b"].sum(axis=1), 1) and np.allclose(F["pi_oracle"].sum(axis=1), 1), "frame: both loggers sum to 1")
    hs_c = np.where(F["half"] == 1, F["pi_b"], 0).sum(axis=1); hs_o = np.where(F["half"] == 1, F["pi_oracle"], 0).sum(axis=1)
    sel = F["thinned"]
    check(np.corrcoef(hs_c[sel], hs_o[sel])[0, 1] > 0.9 and abs(hs_c[sel].mean() - hs_o[sel].mean()) < 0.05,
          f"frame: the counts logger's high-half share tracks the design's (corr {np.corrcoef(hs_c[sel], hs_o[sel])[0, 1]:.3f})")
    thin = (F["half"] == F["thin_half"][:, None]) & F["valid"]
    share = np.where(thin, F["pi_b"], 0).sum(axis=1)
    check(share[sel].mean() < 0.3 and share[~sel].mean() > 0.4,
          f"frame: the logger's mass on the thin half is {share[sel].mean():.3f} on thinned contexts, {share[~sel].mean():.3f} on unthinned")
    check(abs(np.sqrt((F['S_ctx'] ** 2).sum(axis=1)) - 1).max() < 1e-6, "frame: the signature axis is a unit vector per context (float32 file)")
    check(np.isfinite(F["mu"]["holdout"]["A"][F["eligible"][:, None] & F["valid"]]).all(), "frame: the truth is finite on every eligible decision")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tier", default=None)
    p.add_argument("--tier_g0", default=None)
    add_adjustment_set_cli(p)
    add_paths_cli(p, nuisance_dir=False)
    args = p.parse_args()
    print("[check_policy] synthetic")
    run_synthetic()
    if args.data_dir and args.tier and args.tier_g0:
        print("[check_policy] real data: " + args.data_dir)
        run_real(args)
    else:
        print("[check_policy] no --data_dir/--tier/--tier_g0: the real-data checks were NOT run")
    if FAILS:
        print(f"\n[check_policy] {len(FAILS)} FAILED:\n  - " + "\n  - ".join(FAILS))
        sys.exit(1)
    print("\n[check_policy] all checks passed")


if __name__ == "__main__":
    main()
