"""`responders.json`: the compounds step C is allowed to thin (IMPLEMENT.md §3.8.1).

Confounded selection cannot bias a compound that does not respond, so the
tiered split scores only responders:

    max over doses of ||tau_hat|| > `--mult` x the noise floor at that arm size

Both sides come from the Phase 4 `--source real` oracle, which already records
the per-compound maximum and its floor ratio (E11 defines the floor). This
module only selects and writes; it never recomputes tau.

A responder is only *thinnable* if it already has a train well in each of its
four positivity cells, {low, high} x syn_c in {0, 1} (§3.8.2). `build_tiered_split`
refuses the whole instance otherwise -- measured on mcf7_24h, some responders have
every high-half train well at one syn_c level, because syn_c is balanced within an
ARM and the holdout then removes one of ~3 wells. `--require_cells` (default)
drops those here instead, using `splits.dose_half` so the half definition cannot
drift from the builder's.

It also reports the median responder ||tau_hat||, which §3.8.2 needs to resolve
beta = `--syn_effect` x that median. Both the raw and the noise-corrected median
are written, because at n = 3 the floor (14.5) is most of the measured median
(15.2): E||tau_hat||^2 = ||tau||^2 + E||noise||^2, so the raw value is mostly
noise and would set beta several times too large.

Step A (`--confounder cell_id`): the positivity cell is (compound, dose_level,
cell_id), so a compound is thinnable when every one of its dose levels has a
train well in EVERY line of the population (and it has >= 2 levels).

numpy/json only: no torch.

    python -m src.data.responders                        # -> <nuisance_dir>/responders.json
    python -m src.data.responders --n_compounds 100 --out responders_100.json
    python -m src.data.responders --oracle OTHER.json --pool holdout
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

from datasets import load_from_disk  # noqa: E402

from src.data.build_dataset import _atomic_write  # noqa: E402
from src.data.splits import dose_half, load_splits  # noqa: E402
from src.spec import (  # noqa: E402
    add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)

def default_oracle(cfg, pool: str) -> str:
    """The oracle `evaluate --source real --pool <pool>` wrote for this build.

    Its filename carries the POPULATION name, which does not change with
    --data_dir, so discover it rather than rebuild it from the build dir.
    """
    # The UNINJECTED oracle by exact name: an injected one (`_syn*`) describes a
    # different ground truth, and responders are a property of the real data.
    d = os.path.join(cfg.paths.train_output_dir, "eval_artifacts")
    path = os.path.join(d, f"oracle_{cfg.population.name}_pool{pool}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist. Run "
            f"`python -m src.eval.evaluate --source real --pool {pool}` first, "
            f"or pass --oracle.")
    return path


def thinnable(cfg, compound_idx: set[int], confounder: str = "syn_c") -> tuple[set[int], dict]:
    """Which compounds `build_tiered_split` can actually thin.

    `thin_compound` requires a TRAIN well in every positivity cell -- the four
    cells {low, high} x syn_c in {0, 1} (step C), or every (dose_level, cell_id)
    of the compound (step A) -- and >= 2 distinct dose levels for the dose-half
    score to exist. A compound failing either cannot be thinned, and the builder
    refuses the whole instance over one of them -- so screen here.
    """
    meta = (load_from_disk(cfg.paths.tabular_dataset_dir)
            .select_columns(["compound_idx", "dose_level", "is_control", "syn_c", "cell_id"])
            .to_pandas())
    comp = meta["compound_idx"].values.astype(np.int64)
    dl = meta["dose_level"].values.astype(np.float64)
    ctl = meta["is_control"].values.astype(bool)
    syn = meta["syn_c"].values.astype(str)
    half = dose_half(comp, dl, ctl)                      # the builder's own definition
    in_train = np.zeros(len(meta), dtype=bool)
    in_train[np.asarray(load_splits(cfg)["train_idx"], dtype=np.int64)] = True

    line = meta["cell_id"].values.astype(str)
    lines = sorted(cfg.population.cell_ids)
    want = {f"{h}|syn_c={c}" for h in ("low", "high") for c in ("0", "1")}
    ok, why = set(), {"few_levels": [], "empty_cell": []}
    for ci in sorted(compound_idx):
        sel = (comp == ci) & ~ctl
        if np.unique(dl[sel]).size < 2:
            why["few_levels"].append(int(ci)); continue
        tr = sel & in_train
        if confounder == "cell_id":
            want_c = {(float(d), c) for d in np.unique(dl[sel]) for c in lines}
            cells = set(zip(dl[tr].tolist(), line[tr].tolist()))
            if want_c - cells:
                why["empty_cell"].append(int(ci)); continue
        else:
            cells = {f"{'high' if h == 1 else 'low'}|syn_c={v}"
                     for h, v in zip(half[tr], syn[tr])}
            if want - cells:
                why["empty_cell"].append(int(ci)); continue
        ok.add(int(ci))
    return ok, why


def select(per_compound: list[dict], *, mult: float, min_arms: int,
           n_compounds: int | None, frac: float | None, seed: int,
           thinnable_ci: set[int] | None = None) -> tuple[list[dict], dict]:
    """Responders, optionally thinned to a scored subset. Returns (rows, report)."""
    resp = [c for c in per_compound if c["responder"]]
    kept = [c for c in resp if int(c["n_arms"]) >= int(min_arms)]
    dropped = len(resp) - len(kept)
    n_before = len(kept)
    if thinnable_ci is not None:
        kept = [c for c in kept if int(c["compound_idx"]) in thinnable_ci]
    kept.sort(key=lambda c: str(c["pert_id"]))          # stable, name-ordered
    report = {"n_compounds_total": len(per_compound), "n_responders": len(resp),
              "n_dropped_min_arms": dropped, "min_arms": int(min_arms),
              "n_dropped_not_thinnable": n_before - len(kept),
              "mult": float(mult)}
    if n_compounds is not None and frac is not None:
        raise SystemExit("pass --n_compounds or --frac, not both")
    n = None
    if n_compounds is not None:
        n = int(n_compounds)
    elif frac is not None:
        if not 0 < frac <= 1:
            raise SystemExit(f"--frac must be in (0, 1], got {frac}")
        n = int(round(frac * len(kept)))
    if n is not None and n < len(kept):
        # A named stream, so the subset does not depend on draw order elsewhere.
        rng = np.random.default_rng([int(seed), 5])
        pick = np.sort(rng.choice(len(kept), n, replace=False))
        kept = [kept[i] for i in pick]
        report["subsampled_to"] = n
        report["subsample_seed"] = int(seed)
    report["n_scored"] = len(kept)
    return kept, report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--oracle", default=None, help="Default: the --source real oracle in <runs>/eval_artifacts/.")
    p.add_argument("--pool", default="all", choices=("all", "train", "holdout"),
                   help="Which sub-pool of the oracle to read. 'all' is the per-arm oracle (§3.7).")
    p.add_argument("--mult", type=float, default=None,
                   help="Re-threshold at this multiple of the floor. Default: reuse the "
                        "oracle's own responder flag (its --responder_mult).")
    p.add_argument("--min_arms", type=int, default=2,
                   help="Drop responders with fewer dose arms than this. 2 is the floor for "
                        "the dose-half selection score to exist at all.")
    p.add_argument("--require_cells", action="store_true", default=True,
                   help="Keep only compounds with a TRAIN well in all four "
                        "{low,high} x syn_c positivity cells, which is what "
                        "build_tiered_split requires of every scored compound.")
    p.add_argument("--no_require_cells", dest="require_cells", action="store_false")
    p.add_argument("--confounder", default="syn_c", choices=("syn_c", "cell_id"),
                   help="Whose positivity cells the thinnability screen uses: syn_c (step C, "
                        "{low,high} x syn_c) or cell_id (step A, every dose level in every line).")
    p.add_argument("--n_compounds", type=int, default=None, help="Score only this many responders.")
    p.add_argument("--frac", type=float, default=None, help="Score this fraction of them.")
    p.add_argument("--seed", type=int, default=42, help="Subsampling seed.")
    p.add_argument("--out", default="responders.json", help="Relative to the split dir, or a path.")
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    cfg = apply_paths_args(config_from_args(args), args)

    oracle = args.oracle or default_oracle(cfg, args.pool)
    with open(oracle) as fh:
        doc = json.load(fh)
    if doc.get("source") != "real":
        raise SystemExit(f"{oracle} has source={doc.get('source')!r}; responders come from "
                         f"the real-data oracle, never from a generated arm")
    if args.pool not in doc.get("pools", {}):
        raise SystemExit(f"{oracle} has no {args.pool!r} sub-pool (has {sorted(doc['pools'])})")
    block = doc["pools"][args.pool]
    per_compound = block["per_compound"]

    if args.mult is not None:
        # Re-threshold from the recorded floor ratio rather than trusting the flag.
        for c in per_compound:
            c = c
            c["responder"] = (c["max_floor_ratio"] is not None
                              and c["max_floor_ratio"] > args.mult)
        mult = args.mult
    else:
        mult = float(doc["responder_mult"])

    thin_ci = None
    if args.require_cells:
        cand = {int(c["compound_idx"]) for c in per_compound if c["responder"]}
        thin_ci, why = thinnable(cfg, cand, args.confounder)
        print(f"[responders] thinnable: {len(thin_ci):,} of {len(cand):,} responders "
              f"({len(why['few_levels'])} with < 2 dose levels, "
              f"{len(why['empty_cell'])} with an empty positivity cell)")
    kept, report = select(per_compound, mult=mult, min_arms=args.min_arms,
                          n_compounds=args.n_compounds, frac=args.frac, seed=args.seed,
                          thinnable_ci=thin_ci)
    if not kept:
        raise SystemExit("no responder survived the filters; nothing to score")

    # beta's calibration input (§3.8.2). The raw median is mostly noise at n = 3,
    # so the noise-corrected one is the defensible scale.
    norms = np.array([c["max_tau_norm_real"] for c in kept], dtype=np.float64)
    tz = np.load(doc["artifacts"]["tau_npz"], allow_pickle=False)
    pre = f"{args.pool}/"
    dn = tz[pre + "tau_norm_denoised"]
    comp_dn = []
    for c in kept:
        sel = tz[pre + "compound_idx"] == int(c["compound_idx"])
        if sel.any():
            comp_dn.append(float(dn[sel].max()))
    beta_scale = {
        "median_max_tau_norm_raw": float(np.median(norms)),
        "median_max_tau_norm_denoised": (float(np.median(comp_dn)) if comp_dn else None),
        "note": ("beta = --syn_effect x the DENOISED median (§3.8.2 as amended): "
                 "E||tau_hat||^2 = ||tau||^2 + E||noise||^2, so the raw median is "
                 "mostly the n=3 noise floor and would set beta several times too large"),
    }

    out = args.out if os.path.sep in args.out else os.path.join(cfg.paths.nuisance_dir, args.out)
    payload = {
        "compounds": [str(c["pert_id"]) for c in kept],
        "population": doc["population"],
        "oracle": os.path.abspath(oracle),
        "pool": args.pool,
        "table_fingerprint": doc["table_fingerprint"],
        "split_fingerprint": doc["split_fingerprint"],
        "selection": dict(report, confounder=args.confounder),
        "beta_scale": beta_scale,
        "detail": [{k: c[k] for k in ("pert_id", "pert_iname", "compound_idx", "n_arms",
                                      "max_tau_norm_real", "max_floor_ratio",
                                      "best_dose_level")} for c in kept],
    }
    _atomic_write(out, lambda f: json.dump(payload, f, indent=2))
    print(f"[responders] {report['n_responders']:,} responders of "
          f"{report['n_compounds_total']:,} compounds at {mult}x the floor; "
          f"{report['n_dropped_min_arms']:,} dropped for < {args.min_arms} arms, "
          f"{report['n_dropped_not_thinnable']:,} for an unthinnable cell structure; "
          f"{report['n_scored']:,} scored")
    print(f"[responders] median max ||tau||: raw {beta_scale['median_max_tau_norm_raw']:.2f}, "
          f"denoised {beta_scale['median_max_tau_norm_denoised']:.2f}  "
          f"(beta = --syn_effect x the denoised value)")
    print(f"[responders] -> {out}")


if __name__ == "__main__":
    main()
