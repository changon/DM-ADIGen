"""Step-C semi-synthetic covariate `syn_c` and its injected effect (§3.8.2, §3.8.4).

Two halves, one per phase:

  Phase 0  `assign_syn_c` -- build_dataset writes one fixed `syn_c` column so
           that every consumer sees the same draw.
  Phase 5  the injected effect `y <- y + syn_c * beta * v`, applied after
           centring and z-scoring. Two modes:
             global    one direction v for every row (step C, `syn_meta.json`)
             compound  a direction v_k per compound_idx k (step C2,
                       `syn_meta_compound_r<rho>.json`), so the effect of syn_c
                       differs by compound: v_k = normalise(sqrt(1-rho) v +
                       sqrt(rho) u_k). rho = 0 is the global mode, bit for bit.

The injection is resolved ONCE into `<nuisance_dir>/syn_meta.json` by this
module's CLI, and `dataset.py` (the trainer) and `eval/evaluate.py` (the oracle)
both read it back. They cannot share `cfg` alone: `beta` depends on a measured
scale from `responders.json`, and `v`'s seed is a separate knob from `syn_seed`
(spec.py). Resolving it twice would let the trainer and the oracle drift onto
different ground truths, which is the one error step C cannot detect. This
mirrors `expr_meta.json`: one artifact, written once, validated on every read.

`v`'s seed deliberately does NOT live in `OutcomeSpec`: that dataclass is
embedded verbatim in `spec.decisions_record`, which `check_build` compares
against every build's `population_qc.json`, so a new field would invalidate both
builds on disk and force a re-ingest for a knob that is not a Phase 0 decision.

    python -m src.data.synthetic --syn_effect 1.0        # -> <nuisance_dir>/syn_meta.json
    python -m src.data.synthetic --syn_effect 1.0 --vec_seed 7
    python -m src.data.synthetic --syn_effect 0.5 --out syn_meta_half.json
    python -m src.data.synthetic --syn_effect 1.0 --mode compound --rho 1   # step C2

A consumer picks the file with `--syn_meta` (cfg.syn_meta_name; default
`syn_meta.json`). In compound mode the recorded `v_sha1` hashes the WHOLE
direction matrix, so every guard that compares `syn_v_sha1` (arch.json, the eval
identity check, `--truth`) tells C2 from C without a new field.

`syn_c` is balanced at random within every group build_dataset passes: each
treated (compound, dose_level) arm and each plate's DMSO wells. The full data
is therefore unconfounded in `syn_c`, and `syn_c` is pre-treatment by
construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYN_META = "syn_meta.json"
SYN_MODES = ("global", "compound")
_STREAM_V = 23            # substream for the injected direction v
_STREAM_U = 29            # substream for step C2's per-compound directions u_k


def assign_syn_c(groups: Sequence[str], order_keys: Sequence[str], seed: int) -> np.ndarray:
    """(N,) int8 in {0, 1}, balanced within each group.

    A group of n rows gets floor(n/2) of one level and ceil(n/2) of the other,
    with the odd row's level drawn by coin flip (a 3-well arm is a 1/2 or 2/1
    split). Each group draws from its own generator, seeded by (`seed`, group
    name), and orders its rows by `order_keys` first, so a group's assignment
    does not depend on which other groups are in the table.
    """
    groups = np.asarray(groups, dtype=object)
    order_keys = np.asarray(order_keys, dtype=object)
    if groups.shape != order_keys.shape or groups.ndim != 1:
        raise ValueError("groups and order_keys must be 1-d and the same length")
    if pd.Series(order_keys).duplicated().any():
        raise ValueError("order_keys must be unique (they fix the row order within a group)")
    out = np.full(groups.shape[0], -1, dtype=np.int8)
    for g, idx in pd.Series(groups).groupby(groups, sort=True).indices.items():
        idx = idx[np.argsort(order_keys[idx].astype(str), kind="stable")]
        rng = np.random.default_rng([int(seed), zlib.crc32(str(g).encode())])
        lab = (np.arange(idx.size) % 2).astype(np.int8)
        if idx.size % 2 and rng.random() < 0.5:
            lab = 1 - lab
        out[idx[rng.permutation(idx.size)]] = lab
    if (out < 0).any():
        raise AssertionError("syn_c left rows unassigned")
    return out


def max_group_imbalance(groups: Sequence[str], syn_c: np.ndarray) -> int:
    """max over groups of |#(syn_c=1) - #(syn_c=0)|; 1 at most by construction."""
    s = pd.Series(np.asarray(syn_c, dtype=np.int64) * 2 - 1)
    return int(s.groupby(np.asarray(groups, dtype=object)).sum().abs().max()) if len(s) else 0


# ---------------------------------------------------------------------------
# Phase 5: the injected effect  y <- y + syn_c * beta * v   (§3.8.2)
# ---------------------------------------------------------------------------

def effect_vector(vec_seed: int, n_genes: int) -> np.ndarray:
    """The injected direction: a seeded unit vector in 978-d z-space.

    Isotropic on the sphere, so it is not aligned with any gene or with the
    leading variance directions -- the bias step C creates has to be detected,
    not read off a single gene.
    """
    rng = np.random.default_rng([int(vec_seed), _STREAM_V])
    v = rng.normal(size=int(n_genes))
    nrm = float(np.linalg.norm(v))
    if not nrm > 0:
        raise ValueError("degenerate effect vector")
    return (v / nrm).astype(np.float64)


def effect_directions(vec_seed: int, n_genes: int, n_dirs: int, rho: float) -> np.ndarray:
    """(n_dirs, G) unit rows, row k the direction of compound_idx k (§3.8.4).

        v_k = normalise(sqrt(1 - rho) * v + sqrt(rho) * u_k)

    v is step C's `effect_vector(vec_seed)`; u_k is isotropic and drawn from its
    own (vec_seed, k) stream, so a compound's direction does not depend on how
    many other compounds the table holds. rho = 0 returns v in every row bit for
    bit (no renormalisation), so step C2 at rho = 0 IS step C.
    """
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho={rho} is outside [0, 1]")
    v = effect_vector(vec_seed, n_genes)
    if rho == 0.0:
        return np.tile(v, (int(n_dirs), 1))
    a, b = np.sqrt(1.0 - rho), np.sqrt(rho)
    out = np.empty((int(n_dirs), int(n_genes)), dtype=np.float64)
    for k in range(int(n_dirs)):
        u = np.random.default_rng([int(vec_seed), _STREAM_U, k]).normal(size=int(n_genes))
        d = a * v + b * (u / np.linalg.norm(u))
        out[k] = d / np.linalg.norm(d)
    return out


def inject(y: np.ndarray, syn_c, beta: float, v: np.ndarray,
           compound_idx=None) -> np.ndarray:
    """`y + syn_c * beta * v`, on rows with syn_c == 1. Returns a new array.

    Applied AFTER centring and z-scoring (§3.8.2), to treated and vehicle rows
    alike: `syn_c` is pre-treatment, so its effect is not a treatment effect.

    `v` is one (G,) direction (step C), or a (K, G) matrix with one row per
    compound_idx (step C2), in which case row i moves along v[compound_idx[i]].
    At rho = 0 the two give bit-identical output (check_phase5 asserts it).
    """
    y = np.asarray(y)
    c = np.asarray(syn_c).astype(np.float64).reshape(-1, 1)
    if c.shape[0] != y.shape[0]:
        raise ValueError(f"syn_c has {c.shape[0]} rows for y's {y.shape[0]}")
    if not np.isin(np.unique(c), (0.0, 1.0)).all():
        raise ValueError("syn_c must be 0/1")
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        if v.shape != (y.shape[1],):
            raise ValueError(f"v has shape {v.shape}, expected {(y.shape[1],)}")
        return (y.astype(np.float64) + c * float(beta) * v[None, :]).astype(y.dtype)
    if v.ndim != 2 or v.shape[1] != y.shape[1]:
        raise ValueError(f"v has shape {v.shape}, expected (K, {y.shape[1]})")
    if compound_idx is None:
        raise ValueError("a per-compound direction matrix needs compound_idx")
    k = np.asarray(compound_idx, dtype=np.int64).reshape(-1)
    if k.shape[0] != y.shape[0]:
        raise ValueError(f"compound_idx has {k.shape[0]} rows for y's {y.shape[0]}")
    if k.size and (k.min() < 0 or k.max() >= v.shape[0]):
        raise ValueError(f"compound_idx spans [{k.min()}, {k.max()}] but there are "
                         f"{v.shape[0]} directions")
    out = y.astype(np.float64)
    one = np.flatnonzero(c[:, 0] == 1.0)
    out[one] += float(beta) * v[k[one]]
    return out.astype(y.dtype)


def directions_for(meta: dict, compound_idx) -> np.ndarray:
    """(N, G): the direction along which each given compound's syn_c = 1 rows
    move, under the resolved injection `meta` (from `load_syn_meta`)."""
    k = np.asarray(compound_idx, dtype=np.int64).reshape(-1)
    if meta.get("mode", "global") == "global":
        return np.broadcast_to(meta["v"], (k.size, meta["v"].size))
    return meta["V"][k]


def inject_meta(y: np.ndarray, syn_c, compound_idx, meta: dict) -> np.ndarray:
    """`inject` under the resolved injection `meta`: one call shape for both
    modes, so the trainer and the oracle cannot apply different ones."""
    if meta.get("mode", "global") == "global":
        return inject(y, syn_c, meta["beta"], meta["v"])
    return inject(y, syn_c, meta["beta"], meta["V"], compound_idx)


def default_meta_name(mode: str, rho: float | None) -> str:
    """syn_meta.json (step C) or syn_meta_compound_r<rho>.json (step C2)."""
    if mode == "global":
        return SYN_META
    return f"syn_meta_{mode}_r{float(rho):g}.json"


def _meta_name(cfg, name: str | None) -> str:
    """An explicit name, else the one the config selects (`--syn_meta`)."""
    return name or getattr(cfg, "syn_meta_name", None) or SYN_META


def syn_meta_path(cfg, name: str | None = None) -> str:
    """Where `synthetic.py` WRITES the resolved injection: the given split dir."""
    name = _meta_name(cfg, name)
    return name if os.path.sep in name else os.path.join(cfg.paths.nuisance_dir, name)


def _base_nuisance_dir(cfg) -> str:
    return os.path.join(cfg.paths.data_dir, "nuisances")


def resolve_syn_meta_path(cfg, name: str | None = None) -> str:
    """Where to READ it from: the split dir if it has one, else the base build's.

    The injection is a property of the POPULATION and the table -- the same beta
    and the same v -- not of a split. Every tiered instance of a build therefore
    shares the base dir's file rather than holding a copy, which is the point of
    resolving it once: two copies could drift, and a drifted beta would look
    exactly like step-C bias.
    """
    name = _meta_name(cfg, name)
    p = syn_meta_path(cfg, name)
    if os.path.isfile(p) or os.path.sep in name:
        return p
    base = os.path.join(_base_nuisance_dir(cfg), name)
    return base if os.path.isfile(base) else p


def table_syn_seed(cfg) -> int:
    """The seed `syn_c` on disk was actually drawn with (build_dataset records it)."""
    with open(cfg.paths.population_qc_json) as fh:
        return int(json.load(fh)["syn_c"]["seed"])


def _v_sha1(v: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()


def load_syn_meta(cfg, *, name: str | None = None, n_genes: int | None = None) -> dict:
    """Read and validate the resolved injection, returning it with `v` as an array
    (and, in compound mode, the (K, G) direction matrix `V`).

    Refuses a file that does not belong to this build, this `syn_c` draw, or this
    `cfg.outcome.syn_effect` -- a silently mismatched beta or v would make the
    oracle and the generator disagree about the ground truth. `name` defaults to
    the file the config selects (`cfg.syn_meta_name`, `--syn_meta`).
    """
    name = _meta_name(cfg, name)
    path = resolve_syn_meta_path(cfg, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no {name} in {cfg.paths.nuisance_dir} or {_base_nuisance_dir(cfg)}, but "
            f"cfg.outcome.syn_effect={cfg.outcome.syn_effect} is nonzero. Resolve the "
            f"injection once for the build:\n"
            f"    python -m src.data.synthetic --syn_effect {cfg.outcome.syn_effect}")
    with open(path) as fh:
        m = json.load(fh)
    bad = []
    if abs(float(m["syn_effect"]) - float(cfg.outcome.syn_effect)) > 1e-12:
        bad.append(f"syn_effect {m['syn_effect']} != cfg {cfg.outcome.syn_effect}")
    tbl = table_syn_seed(cfg)
    if int(m["syn_seed"]) != tbl:
        bad.append(f"syn_seed {m['syn_seed']} != the table's draw {tbl}")
    if int(cfg.outcome.syn_seed) != tbl:
        bad.append(f"cfg.outcome.syn_seed {cfg.outcome.syn_seed} != the table's draw {tbl}")
    if n_genes is not None and int(m["n_genes"]) != int(n_genes):
        bad.append(f"n_genes {m['n_genes']} != {n_genes}")
    # The file may be shared across a build's split dirs, so the tie to the build
    # is the table, not the split (`split_fingerprint` in it is informational).
    if str(m.get("population")) != str(cfg.population.name):
        bad.append(f"population {m.get('population')!r} != {cfg.population.name!r}")
    mode = str(m.get("mode", "global"))   # files written before step C2 are global
    v = np.asarray(m["v"], dtype=np.float64)
    # In compound mode `v` is the SHARED component and `v_sha1` hashes the whole
    # direction matrix (so the guards that compare syn_v_sha1 tell C2 from C).
    v_hash = m["v_sha1"] if mode == "global" else m.get("v_shared_sha1")
    V = None
    if mode not in SYN_MODES:
        bad.append(f"mode {mode!r} is not one of {SYN_MODES}")
    elif v.size != int(m["n_genes"]):
        bad.append(f"v has {v.size} entries for n_genes {m['n_genes']}")
    elif _v_sha1(v) != v_hash:
        bad.append("v does not match its recorded sha1")
    elif abs(float(np.linalg.norm(v)) - 1.0) > 1e-9:
        bad.append(f"v is not a unit vector (norm {float(np.linalg.norm(v)):.6f})")
    elif mode == "compound":
        # The matrix is STORED, next to this file, and the stored bytes are the
        # ground truth every guard hashes. It is not regenerated for the hash:
        # the normalisations go through BLAS, which rounds the last bits
        # differently on different CPUs (bindel vs the dev box gave two hashes
        # for one seed, §5), so a regenerated hash is not portable across the
        # nodes that train, score and check. Regeneration is kept as a check of
        # the construction, to float tolerance.
        vf = os.path.join(os.path.dirname(path), str(m.get("V_file") or ""))
        if not m.get("V_file") or not os.path.isfile(vf):
            bad.append(f"the direction matrix {m.get('V_file')!r} is missing next to {path}")
        else:
            V = np.load(vf, allow_pickle=False)
            shape = (int(m["n_directions"]), int(m["n_genes"]))
            if V.shape != shape or V.dtype != np.float64:
                bad.append(f"{os.path.basename(vf)} is {V.dtype}{V.shape}, expected float64{shape}")
            elif _v_sha1(V) != m["v_sha1"]:
                bad.append(f"{os.path.basename(vf)} does not match its recorded sha1")
            else:
                R = effect_directions(int(m["vec_seed"]), int(m["n_genes"]),
                                      int(m["n_directions"]), float(m["rho"]))
                err = float(np.abs(V - R).max())
                if err > 1e-12:
                    bad.append(f"{os.path.basename(vf)} is not the matrix vec_seed "
                               f"{m['vec_seed']}, rho {m['rho']} builds (max |diff| {err:.1e})")
            if bad:
                V = None
    if bad:
        raise RuntimeError(f"{path} does not match this run:\n  - " + "\n  - ".join(bad))
    out = dict(m, v=v, mode=mode, name=os.path.basename(path))
    if V is not None:
        out["V"] = V
    return out


def _default_scale(cfg, responders: str) -> tuple[float, str]:
    """beta's scale: the median responder ||tau_hat|| from `responders.json` (§3.8.2).

    The NOISE-CORRECTED median, because E||tau_hat||^2 = ||tau||^2 + E||noise||^2
    and the raw value carries the arm's sampling noise (Phase 4, E11).
    """
    path = responders if os.path.sep in responders else os.path.join(
        cfg.paths.nuisance_dir, responders)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist; beta's scale is the median responder "
            f"||tau_hat||. Build it with `python -m src.data.responders`, or pass "
            f"--scale explicitly.")
    with open(path) as fh:
        r = json.load(fh)
    sc = r["beta_scale"].get("median_max_tau_norm_denoised")
    if sc is None:
        raise SystemExit(f"{path} has no noise-corrected scale; pass --scale")
    return float(sc), f"{os.path.basename(path)}:beta_scale.median_max_tau_norm_denoised"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--syn_effect", type=float, required=True,
                   help="beta = this x the scale. 0 writes an inert file (v1 needs none).")
    p.add_argument("--vec_seed", type=int, default=0,
                   help="Seed of the injected direction v. Separate from --syn_seed, "
                        "which seeds the syn_c ASSIGNMENT and is fixed by the table.")
    p.add_argument("--scale", type=float, default=None,
                   help="Override beta's scale (default: the noise-corrected median "
                        "responder ||tau_hat|| from responders.json).")
    p.add_argument("--responders", default="responders.json")
    p.add_argument("--mode", default="global", choices=SYN_MODES,
                   help="global: one direction v (step C). compound: a direction per "
                        "compound_idx, v_k = normalise(sqrt(1-rho) v + sqrt(rho) u_k) "
                        "(step C2, §3.8.4).")
    p.add_argument("--rho", type=float, default=None,
                   help="compound mode only: the compound-specific share, in [0, 1].")
    p.add_argument("--out", default=None,
                   help="Default: syn_meta.json (global) or syn_meta_compound_r<rho>.json.")
    p.add_argument("--overwrite", action="store_true")
    from src.spec import (  # noqa: E402  (local: keeps this module import-light)
        add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
    if (args.mode == "compound") != (args.rho is not None):
        p.error("--rho is required with --mode compound and meaningless without it")
    cfg = apply_paths_args(config_from_args(args), args)

    from src.data.build_dataset import _atomic_write  # noqa: E402
    from src.data.splits import load_splits  # noqa: E402

    n_genes = int(cfg.outcome.n_genes)
    seed = table_syn_seed(cfg)
    if args.scale is not None:
        scale, src = float(args.scale), "--scale"
    else:
        scale, src = _default_scale(cfg, args.responders)
    beta = float(args.syn_effect) * scale
    v = effect_vector(args.vec_seed, n_genes)
    splits = load_splits(cfg)
    with open(os.path.join(cfg.paths.nuisance_dir, "nuisance_meta.json")) as fh:
        nmeta = json.load(fh)
    tfp = nmeta["table_fingerprint"]
    V = None
    if args.mode == "compound":
        # One row per compound_idx, the vehicle's (0) included: the vocab's size.
        V = effect_directions(args.vec_seed, n_genes, int(nmeta["n_compounds"]), args.rho)

    out = syn_meta_path(cfg, args.out or default_meta_name(args.mode, args.rho))
    if os.path.isfile(out) and not args.overwrite:
        with open(out) as fh:
            old = json.load(fh)
        same = (abs(float(old["syn_effect"]) - float(args.syn_effect)) < 1e-12
                and int(old["vec_seed"]) == int(args.vec_seed)
                and abs(float(old["scale"]) - scale) < 1e-12
                and str(old.get("mode", "global")) == args.mode
                and (V is None or abs(float(old["rho"]) - float(args.rho)) < 1e-12))
        if not same:
            raise SystemExit(
                f"[syn] {out} already resolves a different injection "
                f"(mode={old.get('mode', 'global')}, rho={old.get('rho')}, "
                f"syn_effect={old['syn_effect']}, vec_seed={old['vec_seed']}, "
                f"scale={old['scale']:.4f}); pass --overwrite or --out")
        # The same injection: leave the artifact alone. It is written ONCE --
        # rewriting it on another machine would re-round the compound matrix
        # (see load_syn_meta) and silently change the hash every run records.
        if V is None or os.path.isfile(os.path.join(os.path.dirname(out),
                                                    str(old.get("V_file") or "-"))):
            print(f"[syn] {out} already resolves this injection; left unchanged "
                  f"(sha1 {str(old['v_sha1'])[:12]})")
            return
        raise SystemExit(f"[syn] {out} has no stored direction matrix next to it; "
                         f"pass --overwrite to resolve it again")

    meta = {"syn_effect": float(args.syn_effect), "syn_seed": seed,
            "vec_seed": int(args.vec_seed), "scale": scale, "scale_source": src,
            "beta": beta, "n_genes": n_genes, "v_sha1": _v_sha1(v),
            "population": cfg.population.name, "table_fingerprint": tfp,
            "split_fingerprint": splits["split_fingerprint"],
            "note": ("y <- y + syn_c * beta * v, applied after centring and "
                     "z-scoring, to treated and vehicle rows alike (§3.8.2)"),
            "v": [float(x) for x in v]}
    if V is not None:
        # `v` stays the shared component; `v_sha1` now hashes the whole matrix,
        # which is what arch.json and every eval guard record and compare. The
        # matrix itself goes to a sidecar, written (atomically) before the JSON
        # that names it, so the JSON never points at a missing or older matrix.
        v_file = os.path.splitext(os.path.basename(out))[0] + "_V.npy"
        _atomic_write(os.path.join(os.path.dirname(out), v_file),
                      lambda f: np.save(f, np.ascontiguousarray(V, dtype=np.float64)),
                      mode="wb")
        cos_v = V @ v
        off = V[1:] @ V[1:].T
        off = np.abs(off[np.triu_indices(off.shape[0], 1)])
        if off.size == 0:          # a build with one compound has no pair to compare
            off = np.full(1, np.nan)
        meta.update({
            "mode": "compound", "rho": float(args.rho),
            "n_directions": int(V.shape[0]), "V_file": v_file,
            "v_shared_sha1": _v_sha1(v), "v_sha1": _v_sha1(V),
            "geometry": {"cos_with_v_median": float(np.median(cos_v)),
                         "abs_cos_between_compounds_median": float(np.median(off)),
                         "abs_cos_between_compounds_max": float(off.max())},
            "note": ("y <- y + syn_c * beta * v_k, k = the row's compound_idx (0 = "
                     "vehicle), v_k = normalise(sqrt(1-rho) v + sqrt(rho) u_k); "
                     "after centring and z-scoring, treated and vehicle rows alike "
                     "(§3.8.4)")})
    _atomic_write(out, lambda f: json.dump(meta, f, indent=2))
    print(f"[syn] syn_effect {args.syn_effect} x scale {scale:.4f} ({src}) "
          f"-> beta {beta:.4f}")
    print(f"[syn] v: unit vector, {n_genes} genes, vec_seed {args.vec_seed}, "
          f"sha1 {_v_sha1(v)[:12]}; syn_c assignment seed {seed}")
    if V is not None:
        g = meta["geometry"]
        print(f"[syn] compound mode, rho {args.rho:g}: {V.shape[0]} directions, "
              f"matrix sha1 {meta['v_sha1'][:12]}; cos(v_k, v) median "
              f"{g['cos_with_v_median']:.4f}, |cos(v_j, v_k)| median "
              f"{g['abs_cos_between_compounds_median']:.4f} / max "
              f"{g['abs_cos_between_compounds_max']:.4f}")
    print(f"[syn] -> {out}")


if __name__ == "__main__":
    main()
