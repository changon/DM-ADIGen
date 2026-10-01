"""Step-C semi-synthetic covariate `syn_c` and its injected effect (§3.8.2).

Two halves, one per phase:

  Phase 0  `assign_syn_c` -- build_dataset writes one fixed `syn_c` column so
           that every consumer sees the same draw.
  Phase 5  the injected effect `y <- y + syn_c * beta * v`, applied after
           centring and z-scoring.

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
_STREAM_V = 23            # substream for the injected direction v


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


def inject(y: np.ndarray, syn_c, beta: float, v: np.ndarray) -> np.ndarray:
    """`y + syn_c * beta * v`, on rows with syn_c == 1. Returns a new array.

    Applied AFTER centring and z-scoring (§3.8.2), to treated and vehicle rows
    alike: `syn_c` is pre-treatment, so its effect is not a treatment effect.
    """
    y = np.asarray(y)
    c = np.asarray(syn_c).astype(np.float64).reshape(-1, 1)
    if c.shape[0] != y.shape[0]:
        raise ValueError(f"syn_c has {c.shape[0]} rows for y's {y.shape[0]}")
    if not np.isin(np.unique(c), (0.0, 1.0)).all():
        raise ValueError("syn_c must be 0/1")
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (y.shape[1],):
        raise ValueError(f"v has shape {v.shape}, expected {(y.shape[1],)}")
    return (y.astype(np.float64) + c * float(beta) * v[None, :]).astype(y.dtype)


def syn_meta_path(cfg, name: str = SYN_META) -> str:
    """Where `synthetic.py` WRITES the resolved injection: the given split dir."""
    return name if os.path.sep in name else os.path.join(cfg.paths.nuisance_dir, name)


def _base_nuisance_dir(cfg) -> str:
    return os.path.join(cfg.paths.data_dir, "nuisances")


def resolve_syn_meta_path(cfg, name: str = SYN_META) -> str:
    """Where to READ it from: the split dir if it has one, else the base build's.

    The injection is a property of the POPULATION and the table -- the same beta
    and the same v -- not of a split. Every tiered instance of a build therefore
    shares the base dir's file rather than holding a copy, which is the point of
    resolving it once: two copies could drift, and a drifted beta would look
    exactly like step-C bias.
    """
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


def load_syn_meta(cfg, *, name: str = SYN_META, n_genes: int | None = None) -> dict:
    """Read and validate the resolved injection, returning it with `v` as an array.

    Refuses a file that does not belong to this build, this `syn_c` draw, or this
    `cfg.outcome.syn_effect` -- a silently mismatched beta or v would make the
    oracle and the generator disagree about the ground truth.
    """
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
    v = np.asarray(m["v"], dtype=np.float64)
    if v.size != int(m["n_genes"]):
        bad.append(f"v has {v.size} entries for n_genes {m['n_genes']}")
    elif _v_sha1(v) != m["v_sha1"]:
        bad.append("v does not match its recorded sha1")
    elif abs(float(np.linalg.norm(v)) - 1.0) > 1e-9:
        bad.append(f"v is not a unit vector (norm {float(np.linalg.norm(v)):.6f})")
    if bad:
        raise RuntimeError(f"{path} does not match this run:\n  - " + "\n  - ".join(bad))
    return dict(m, v=v)


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
    p.add_argument("--out", default=SYN_META)
    p.add_argument("--overwrite", action="store_true")
    from src.spec import (  # noqa: E402  (local: keeps this module import-light)
        add_adjustment_set_cli, add_paths_cli, apply_paths_args, config_from_args)
    add_adjustment_set_cli(p)
    add_paths_cli(p)
    args = p.parse_args()
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
        tfp = json.load(fh)["table_fingerprint"]

    out = syn_meta_path(cfg, args.out)
    if os.path.isfile(out) and not args.overwrite:
        with open(out) as fh:
            old = json.load(fh)
        same = (abs(float(old["syn_effect"]) - float(args.syn_effect)) < 1e-12
                and int(old["vec_seed"]) == int(args.vec_seed)
                and abs(float(old["scale"]) - scale) < 1e-12)
        if not same:
            raise SystemExit(
                f"[syn] {out} already resolves a different injection "
                f"(syn_effect={old['syn_effect']}, vec_seed={old['vec_seed']}, "
                f"scale={old['scale']:.4f}); pass --overwrite or --out")

    meta = {"syn_effect": float(args.syn_effect), "syn_seed": seed,
            "vec_seed": int(args.vec_seed), "scale": scale, "scale_source": src,
            "beta": beta, "n_genes": n_genes, "v_sha1": _v_sha1(v),
            "population": cfg.population.name, "table_fingerprint": tfp,
            "split_fingerprint": splits["split_fingerprint"],
            "note": ("y <- y + syn_c * beta * v, applied after centring and "
                     "z-scoring, to treated and vehicle rows alike (§3.8.2)"),
            "v": [float(x) for x in v]}
    _atomic_write(out, lambda f: json.dump(meta, f, indent=2))
    print(f"[syn] syn_effect {args.syn_effect} x scale {scale:.4f} ({src}) "
          f"-> beta {beta:.4f}")
    print(f"[syn] v: unit vector, {n_genes} genes, vec_seed {args.vec_seed}, "
          f"sha1 {meta['v_sha1'][:12]}; syn_c assignment seed {seed}")
    print(f"[syn] -> {out}")


if __name__ == "__main__":
    main()
