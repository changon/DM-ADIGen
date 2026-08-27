"""Treatment-rarity ablation (compound-level).

Designate some compounds as rare using a confounding like structure.

TWO MECHANISMS, selected by the confounding gammas:

  all gamma = 0   MCAR. Rows drop uniformly at random

  gamma > 0       Biased removal. Rare-compound rows drop with probability according to some confounding set

"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from datasets import load_from_disk

from src.spec import CaseConfig


@dataclass(frozen=True)
class RarityConfig:
    compound_frac: float = 0.0
    keep_frac: float = 0.1
    seed: int = 42
    confound_pos_gamma: float = 0.0
    confound_min_cell: int = 2
    force_compounds: tuple = ()

    @property
    def active(self) -> bool:
        return self.compound_frac > 0.0 and self.keep_frac < 1.0

    @property
    def confounded(self) -> bool:
        return self.active and self.confound_pos_gamma != 0.0

    @property
    def tag(self) -> str:
        if not self.active:
            return ""
        t = (
            f"rare{int(round(self.compound_frac * 100)):02d}"
            f"_keep{int(round(self.keep_frac * 100)):02d}"
            f"_s{self.seed}"
        )

        if self.confound_pos_gamma != 0.0:
            t += f"_pg{self.confound_pos_gamma:g}"
        # min_cell changes WHICH rows survive, not just how many. Suffixed only when it departs from the default 2, so every artifact already on disk keeps its current path.
        if self.confounded and self.confound_min_cell != 2:
            t += f"_mc{int(self.confound_min_cell)}"
        if self.force_compounds:
            t += f"_f{len(self.force_compounds)}"
        return t

    def to_dict(self) -> dict:
        return {
            "compound_frac": self.compound_frac,
            "keep_frac": self.keep_frac,
            "seed": self.seed,
            "confound_pos_gamma": self.confound_pos_gamma,
            "confound_min_cell": self.confound_min_cell,
            "force_compounds": list(self.force_compounds),
            "tag": self.tag,
            "active": self.active,
            "confounded": self.confounded,
        }


def tagged_nuisance_dir(cfg: CaseConfig, rcfg: RarityConfig) -> str:
    """Nuisance dir keyed by BOTH the ablation and the roles it was fit under.

    alpha depends on (C, E) and psi on C
    """
    from src.spec import role_tag
    base = cfg.paths.nuisance_dir
    if rcfg.active:
        base = f"{base}_{rcfg.tag}"
    return base + role_tag(cfg)


def tagged_output_subdir(output_subdir: str, rcfg: RarityConfig) -> str:
    if not rcfg.active:
        return output_subdir
    return f"{output_subdir}_{rcfg.tag}"


def pick_rare_compounds(cfg: CaseConfig, rcfg: RarityConfig) -> np.ndarray:
    """Deterministically choose which compound_idx values are rare.
    """
    if not rcfg.active:
        return np.array([], dtype=np.int64)
    ds = load_from_disk(cfg.paths.tabular_dataset_dir)
    all_compounds = np.unique(np.asarray(ds["compound_idx"], dtype=np.int64))
    eligible = all_compounds[all_compounds != 0]
    n_rare = int(round(rcfg.compound_frac * eligible.shape[0]))
    forced = np.asarray([c for c in rcfg.force_compounds if c != 0],
                        dtype=np.int64)
    if n_rare <= 0 and forced.size == 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(rcfg.seed)
    pick = rng.choice(eligible, size=max(n_rare, 0), replace=False) \
        if n_rare > 0 else np.array([], dtype=np.int64)
    if forced.size:
        unknown = forced[~np.isin(forced, eligible)]
        if unknown.size:
            raise ValueError(
                f"--rare-force-compounds contains ids not in the vocabulary "
                f"(or the control): {unknown.tolist()}")
        pick = np.union1d(pick, forced)
    pick = np.unique(pick)
    pick.sort()
    return pick.astype(np.int64)


def subsample_train_idx(
    cfg: CaseConfig, train_idx: np.ndarray, rcfg: RarityConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (subsampled_train_idx, rare_compound_ids, info_dict)."""
    if not rcfg.active:
        return train_idx, np.array([], dtype=np.int64), {"rare_compounds": []}

    rare = pick_rare_compounds(cfg, rcfg)
    if rare.size == 0:
        return train_idx, rare, {"rare_compounds": []}

    ds = load_from_disk(cfg.paths.tabular_dataset_dir)
    compound = np.asarray(ds["compound_idx"], dtype=np.int64)
    is_rare_row = np.isin(compound[train_idx], rare)

    keep_mask = np.ones(train_idx.shape[0], dtype=bool)
    rare_positions = np.where(is_rare_row)[0]
    n_rare_before = int(rare_positions.size)
    n_keep = int(round(rcfg.keep_frac * n_rare_before))
    drop_n = n_rare_before - n_keep
    conf_info: dict = {}
    if drop_n > 0:
        rng = np.random.default_rng(rcfg.seed + 1)
        if not rcfg.confounded: # MCAR
            drop_positions = rng.choice(rare_positions, size=drop_n, replace=False)
        else:
            # Budget allocated across cells, then drawn uniformly within each
            cell_id, cells, z = _confound_cells(
                ds, train_idx[rare_positions],
                cfg, rcfg.confound_pos_gamma,
                cfg.covariates.edge_margin)
            d_per, drop_n_real, conf_note = _allocate_drops(
                cell_id, len(cells), drop_n, z, rcfg.confound_min_cell)
            parts = [rng.choice(rare_positions[cell_id == i], size=int(d),  replace=False) for i, d in enumerate(d_per) if d > 0]
            drop_positions = (np.concatenate(parts) if parts else np.array([], dtype=np.int64))
            conf_info = _confound_diagnostics(cell_id, cells, rare_positions, drop_positions, z, d_per)
            conf_info["note"] = conf_note
            conf_info["drop_n_requested"] = int(drop_n)
            conf_info["drop_n_realized"] = int(drop_n_real)
            drop_n = int(drop_n_real)   # so n_rare_rows_after stays truthful

        keep_mask[drop_positions] = False

    new_train_idx = train_idx[keep_mask].copy()
    new_train_idx.sort()
    info = {
        "rare_compounds": rare.tolist(),
        "n_rare_compounds": int(rare.shape[0]),
        "n_train_before": int(train_idx.shape[0]),
        "n_train_after": int(new_train_idx.shape[0]),
        "n_rare_rows_before": n_rare_before,
        "n_rare_rows_after": n_rare_before - drop_n,
        **({"confounding": conf_info} if conf_info else {}),
    }
    return new_train_idx, rare, info


def _allocate_drops(cell_id, n_cells, drop_n, z, min_cell):
    """Rows to drop from each cell: biased by z, but never past the positivity
    floor. Returns (drops_per_cell, realized_total, note).
    """
    n_k = np.bincount(cell_id, minlength=n_cells).astype(np.float64)
    cap = np.maximum(n_k - float(min_cell), 0.0).astype(np.int64)
    capacity = int(cap.sum())
    note = "positivity floor not binding"
    if drop_n > capacity:
        note = (f"CLAMPED: requested {drop_n:,} drops but positivity "
                f"(min_cell={min_cell}) allows only {capacity:,}; realized "
                f"keep_frac is higher than asked for. Lower --rare-confound-"
                f"min-cell only if you accept unbounded alpha in thin cells.")
        drop_n = capacity

    w = 1.0 / (1.0 + np.exp(-z))
    share = n_k * w
    share = share / max(share.sum(), 1e-12)

    d = np.minimum(np.floor(share * drop_n).astype(np.int64), cap)
    # Water-fill the remainder (from flooring AND from capped cells) onto whatever headroom is left
    order = np.argsort(-share)
    while int(d.sum()) < drop_n:
        progressed = False
        for k in order:
            if int(d.sum()) >= drop_n:
                break
            if d[k] < cap[k]:
                d[k] = min(cap[k], d[k] + (drop_n - int(d.sum())))
                progressed = True
        if not progressed:
            break
    at_floor = int(((d >= cap) & (cap > 0)).sum())
    if at_floor and "CLAMPED" not in note:
        note = (f"positivity floor binding on {at_floor}/{n_cells} cells -- realized confounding is weaker than gamma asks for")
    return d, int(d.sum()), note


def _standardize(v: np.ndarray) -> np.ndarray:
    """Zero mean, unit sd."""
    sd = float(v.std())
    return (v - float(v.mean())) / (sd if sd > 1e-12 else 1.0)


def _confound_cells(ds, row_idx, cfg, pos_gamma, edge_margin=2):
    """Cell id per row, the cell key list, and the per-row selection score z.

        z = pos_gamma * standardize(z_dose * z_edge)

    z_dose is the within-compound dose rank; z_edge the edge/interior contrast.
    Cells are (compound, dose, edge). z depends on (X, A) only, never the outcome.
    """
    from src.data.build_dataset import edge_labels

    idx = np.asarray(row_idx)
    comp = np.asarray(ds["compound_idx"], dtype=np.int64)[idx]
    lx = np.round(np.asarray(ds["log10_conc"], dtype=np.float32)[idx], 3)
    edge = edge_labels(np.array([str(x) for x in ds["well"]])[idx], edge_margin)
    e_code = (edge == "edge").astype(np.int64)

    z_dose = np.zeros(len(idx), dtype=np.float64)
    for c in np.unique(comp):
        m = comp == c
        doses = np.unique(lx[m])
        if doses.size > 1:
            rank = np.searchsorted(doses, lx[m]).astype(np.float64)
            z_dose[m] = 2.0 * rank / (doses.size - 1) - 1.0
    z_dose = _standardize(z_dose)
    z_edge = (_standardize(e_code.astype(np.float64)) if e_code.ptp() > 0
              else np.zeros(len(idx)))
    z = pos_gamma * (_standardize(z_dose * z_edge) if e_code.ptp() > 0
                     else np.zeros(len(idx)))

    keys = [(int(c), float(d), int(e)) for c, d, e in zip(comp, lx, e_code)]
    cells = sorted(set(keys))
    lookup = {k: i for i, k in enumerate(cells)}
    cell_id = np.array([lookup[k] for k in keys], dtype=np.int64)
    z_cell = np.zeros(len(cells), dtype=np.float64)
    z_cell[cell_id[::-1]] = z[::-1]
    return cell_id, cells, z_cell


def _confound_diagnostics(cell_id, cells, rare_positions, drop_positions,
                          z_cell, d_per) -> dict:
    """check confounding"""
    n_k = np.array([(cell_id == k).sum() for k in range(len(cells))],
                   dtype=np.int64)
    after = n_k - np.asarray(d_per, dtype=np.int64)
    w_before = n_k / max(n_k.sum(), 1)
    w_after = after / max(after.sum(), 1)
    by_dose: dict = {}
    for k, (_, dose, _) in enumerate(cells):
        b = by_dose.setdefault(round(float(dose), 3), [0, 0])
        b[0] += int(n_k[k]); b[1] += int(d_per[k])

    # Cramer's V on (within-compound dose rank x edge): near 0 means close to no confouding
    dose_ranks: dict = {}
    for comp, dose, _ in cells:
        dose_ranks.setdefault(int(comp), set()).add(round(float(dose), 3))
    order = {c: {d: i for i, d in enumerate(sorted(v))}
             for c, v in dose_ranks.items()}
    edge_levels = sorted({int(e) for _, _, e in cells})
    n_rank = max((len(v) for v in order.values()), default=1)
    tbl = np.zeros((n_rank, max(len(edge_levels), 1)), dtype=np.float64)
    for k, (comp, dose, e) in enumerate(cells):
        if after[k] > 0:
            tbl[order[int(comp)][round(float(dose), 3)],
                edge_levels.index(int(e))] += float(after[k])
    tot = tbl.sum()
    if tot > 0 and min(tbl.shape) > 1:
        exp = np.outer(tbl.sum(1), tbl.sum(0)) / tot
        chi2 = ((tbl - exp) ** 2 / np.where(exp > 0, exp, 1.0)).sum()
        ax_v = float(np.sqrt(chi2 / (tot * (min(tbl.shape) - 1))))
    else:
        ax_v = 0.0
    return {
        "ax_cramers_v": round(ax_v, 4),
        "n_cells": len(cells),
        "min_cell_after": int(after.min()) if after.size else None,
        "n_cells_emptied": int((after <= 0).sum()),
        "frac_cells_at_floor": float((after <= after.min()).mean())
        if after.size else None,
        "mean_z_before": float((w_before * z_cell).sum()),
        "mean_z_after": float((w_after * z_cell).sum()),
        "drop_rate_by_dose": {str(k): round(v[1] / max(v[0], 1), 4)
                              for k, v in sorted(by_dose.items())},
    }


def rarity_meta_path(nuisance_dir: str) -> str:
    return os.path.join(nuisance_dir, "rarity_meta.json")


def save_rarity_meta(nuisance_dir: str, rcfg: RarityConfig, info: dict) -> None:
    os.makedirs(nuisance_dir, exist_ok=True)
    payload = {**rcfg.to_dict(), **info}
    with open(rarity_meta_path(nuisance_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_rarity_meta(nuisance_dir: str) -> dict | None:
    path = rarity_meta_path(nuisance_dir)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def add_rarity_cli_args(parser) -> None:
    g = parser.add_argument_group("rarity ablation (compound-level subsampling)")
    g.add_argument(
        "--rare-compound-frac", type=float, default=0.0,
        help="Fraction of non-control compounds to mark as rare (0 = off, "
             "0.20 = the standard ablation).",
    )
    g.add_argument(
        "--rare-keep-frac", type=float, default=0.1,
        help="Fraction of rare-compound rows to retain in the training pool. "
             "0.1 = 10%% of original.",
    )
    g.add_argument(
        "--rare-seed", type=int, default=42,
        help="Seed for picking which compounds are rare and which rows to drop.",
    )
    g.add_argument(
        "--rare-confound-pos-gamma", type=float, default=0.0,
        help="Strength of the DOSE x EDGE interaction -- the confounding lever. ")
    g.add_argument(
        "--rare-confound-min-cell", type=int, default=2,
        help="Rows every (compound, dose, edge) cell must RETAIN.",
    )
    g.add_argument(
        "--rare-force-compounds", default="",
        help="Comma-separated compound_idx values (or names resolvable in that are always rare.",
    )


def resolve_compound_ids(spec: str, vocab_path: str | None = None) -> tuple:
    """Parse a --rare-force-compounds string into a tuple of compound ids.

    Accepts ints and names; names are looked up in compound_vocab.json so a caller can write 'GS-441524,Remdesivir (GS-5734)' rather than 451,828.
    """
    if not spec:
        return ()
    items = [s.strip() for s in spec.split(",") if s.strip()]
    out, vocab = [], None
    for it in items:
        if it.lstrip("-").isdigit():
            out.append(int(it))
            continue
        if vocab is None:
            path = vocab_path or os.path.join(
                CaseConfig().paths.nuisance_dir, "compound_vocab.json")
            with open(path) as f:
                vocab = json.load(f)
        if it not in vocab:
            raise ValueError(f"compound {it!r} not in {path}")
        out.append(int(vocab[it]))
    return tuple(sorted(set(out)))


def rarity_from_args(args) -> RarityConfig:
    return RarityConfig(
        compound_frac=float(args.rare_compound_frac),
        keep_frac=float(args.rare_keep_frac),
        seed=int(args.rare_seed),
        confound_pos_gamma=float(getattr(args, "rare_confound_pos_gamma", 0.0)),
        confound_min_cell=int(getattr(args, "rare_confound_min_cell", 2)),
        force_compounds=resolve_compound_ids(
            str(getattr(args, "rare_force_compounds", "") or "")),
    )
