"""In-memory vector Dataset for LINCS L1000 (Y = the 978 landmark genes).

Rewritten from RxRx19a/src/data/dataset.py (IMPLEMENT.md §3.2, §3.6): no PIL,
no image VAE, no per-row HF access. `LincsDataset` holds its rows in memory,
with Y normalised once by the split's expr_meta.json (src.data.expr_stats):

    y = (x - centre[plate] - mean) / std        (z-space; never clamped)

The step-C injection `y <- y + syn_c * beta * v` goes here after
normalisation, from the resolved syn_meta file the config selects
(src/data/synthetic.py; step C2 moves each compound along its own v_k).

`build_cond_spec`, `cond_from_arrays`, `cond_from_batch` and `dose_probe` are
copied from RxRx19a (spec-driven, not image code). The categorical ("level")
dose encoding is dropped: LINCS doses are float variants (97 raw values, one
arm per `dose_level`), and the model sees continuous log10_conc (decision 3).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from src.data.build_dataset import CONTEXT_FIELDS, CONTEXT_SOURCE_COLUMNS, ContextEncoder
from src.data.expr_stats import load_expr_meta, normalize_expr, plate_codes
from src.data.synthetic import inject_meta, load_syn_meta
from src.models.conditioning import CondSpec, Field
from src.spec import ACTION_FIELDS, BY_NAME, CONTEXT_COL, ROLE_OF, CaseConfig

TABLE_COLUMNS = ("compound_idx", "conc", "log10_conc", "is_control", "cov_vec") + CONTEXT_SOURCE_COLUMNS


class LincsDataset(Dataset):
    """Table rows `indices` (default: all), held in memory. Returns per row:
        y:            (G,) float32 z-scored landmark expression (if load_y)
        compound_idx: int64 scalar (0 = vehicle)
        conc:         float32 scalar, uM (0 for vehicles)
        log10_conc:   float32 scalar (control sentinel for vehicles; cond_from_* turns it into NaN)
        is_control:   int8 scalar
        cov_vec:      (d_cov,) float32 covariate one-hots
        context:      (F,) int64 context codes, CONTEXT_FIELDS order
        row_id:       int64 table row id
    Set `load_y=False` to skip the expression matrix (nuisance fitters).
    """

    def __init__(
        self,
        cfg: CaseConfig,
        indices: Optional[np.ndarray] = None,
        load_y: bool = True,
        plate_center: Optional[str] = None,
    ):
        self.cfg = cfg
        ds = load_from_disk(cfg.paths.tabular_dataset_dir).select_columns(list(TABLE_COLUMNS))
        n_all = len(ds)
        self.row_ids = (np.arange(n_all, dtype=np.int64) if indices is None
                        else np.asarray(indices, dtype=np.int64))
        if self.row_ids.size and (self.row_ids.min() < 0 or self.row_ids.max() >= n_all):
            raise IndexError(f"indices outside the table (0..{n_all - 1})")
        df = ds.to_pandas().iloc[self.row_ids].reset_index(drop=True)
        self.compound_idx = torch.as_tensor(df["compound_idx"].values.astype(np.int64))
        self.conc = torch.as_tensor(df["conc"].values.astype(np.float32))
        self.log10_conc = torch.as_tensor(df["log10_conc"].values.astype(np.float32))
        self.is_control = torch.as_tensor(df["is_control"].values.astype(np.int8))
        self.cov_vec = torch.as_tensor(np.stack(df["cov_vec"].values).astype(np.float32)
                                       if len(df) else np.zeros((0, 0), np.float32))

        # Context codes precomputed once for the whole split
        enc = ContextEncoder.load_or_build(cfg)
        self.context_cardinalities = enc.cardinalities
        self.context = torch.as_tensor(enc.encode(df[list(CONTEXT_SOURCE_COLUMNS)]))   # (n, F) int64

        self.y = None
        self.expr_meta = None
        # Set whenever the step-C injection is active; None otherwise, including
        # on the load_y=False path the nuisance fitters use (they never see y).
        self.syn_meta = None
        if load_y:
            m = load_expr_meta(cfg, plate_center)
            expr = np.load(cfg.paths.expr_npy, mmap_mode="r")
            if expr.shape != (n_all, cfg.outcome.n_genes):
                raise ValueError(f"expr.npy {expr.shape} does not match the table ({n_all:,} x {cfg.outcome.n_genes})")
            x = np.asarray(expr[self.row_ids])
            pc = plate_codes(df["det_plate"].values, m["plates"])
            z = normalize_expr(x, pc, m)
            # Step C (§3.8.2): the injection lands AFTER centring and z-scoring,
            # on treated and vehicle rows alike. beta and the direction(s) come
            # from the single resolved syn_meta file (cfg.syn_meta_name), never
            # from cfg, so the trainer and the oracle share one ground truth.
            if cfg.outcome.syn_effect != 0:
                self.syn_meta = load_syn_meta(cfg, n_genes=cfg.outcome.n_genes)
                z = inject_meta(z, df["syn_c"].values.astype(np.int64),
                                df["compound_idx"].values.astype(np.int64),
                                self.syn_meta)
            self.y = torch.from_numpy(z)
            self.plate_code = torch.as_tensor(pc)
            self.expr_meta = {k: v for k, v in m.items() if k not in ("centre", "mean", "std", "cap_frac")}

    def __len__(self) -> int:
        return int(self.row_ids.size)

    def __getitem__(self, idx: int) -> dict:
        out: dict[str, torch.Tensor] = {
            "compound_idx": self.compound_idx[idx],
            "conc": self.conc[idx],
            "log10_conc": self.log10_conc[idx],
            "is_control": self.is_control[idx],
            "cov_vec": self.cov_vec[idx],
            "context": self.context[idx],       # Context the generator conditions on; CONTEXT_FIELDS order.
            "row_id": torch.tensor(self.row_ids[idx], dtype=torch.long),
        }
        if self.y is not None:
            out["y"] = self.y[idx]
        return out


def get_dataloader(
    cfg: CaseConfig,
    batch_size: int,
    *,
    indices: Optional[np.ndarray] = None,
    load_y: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """In-memory rows: num_workers=0 avoids copying the tensors into workers."""
    ds = LincsDataset(cfg, indices=indices, load_y=load_y)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


DOSE_ENCODINGS = ("scalar", "fourier")


def build_cond_spec(
    cfg,
    n_compounds: int,
    log10_conc_train: np.ndarray | torch.Tensor,
    *,
    include_env: bool = False,
    adjustment_set: Sequence[str] | None = None,
    dose_encoding: str = "scalar",
) -> CondSpec:
    """Cond spec.

    `log10_conc_train` is the raw dose column of the training, treated rows. Sets `loc`/`scale`.

    Cardinalities and level names come from `ContextEncoder`

    `include_env` defaults to false.

    `adjustment_set` promotes named fields from their default role to C.
    """
    if dose_encoding not in DOSE_ENCODINGS:
        raise ValueError(f"dose_encoding must be one of {DOSE_ENCODINGS}, got {dose_encoding!r}")
    x = np.asarray(log10_conc_train, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("log10_conc_train is empty after dropping non-finite values; dose standardisation needs real treated-row concentrations.")
    loc, scale = float(x.mean()), float(x.std())
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(
            f"log10_conc_train has zero/degenerate spread (std={scale}); dose "
            f"cannot be standardised, and a constant dose is not an action.")

    # The RESOLVED adjustment set
    adj = tuple(cfg.adjustment_set if adjustment_set is None else adjustment_set)
    _missing = [f for f in adj if f not in cfg.covariates.categorical_cols]
    if _missing:
        raise ValueError(
            f"adjustment_set names {_missing}, absent from "
            f"CovariateSpec.categorical_cols {list(cfg.covariates.categorical_cols)}, "
            f"so alpha cannot condition on it. Mark it adjustable=True in "
            f"spec.FIELDS and re-run build_dataset.")
    _uncarried = [f for f in adj if f not in CONTEXT_FIELDS]
    if _uncarried:
        raise ValueError(
            f"adjustment_set names {_uncarried}, which the context tensor does "
            f"not carry, so the generator cannot condition on what alpha adjusts "
            f"for. Add it to spec.FIELDS and re-run build_dataset.")

    enc = ContextEncoder.load_or_build(cfg)

    fields: list[Field] = []

    # --- A: the intervention, from spec.FIELDS ------------------------------
    for name in ACTION_FIELDS:
        d = BY_NAME[name]
        if d.kind == "cont":
            nf = d.n_freqs if (name != "dose" or dose_encoding == "fourier") else 0
            fields.append(Field(name=name, role="A", kind="cont",
                                loc=loc, scale=scale, nullable=d.nullable,
                                n_freqs=nf))
        else:
            card = d.cardinality
            if card is None:
                if name != "compound":
                    raise ValueError(
                        f"action field {name!r} has cardinality=None and no rule "
                        f"for resolving it from data. Give it a literal in "
                        f"spec.FIELDS or add a resolver here.")
                card = int(n_compounds)
            fields.append(Field(name=name, role="A", kind="cat", cardinality=card))

    # --- C and E: cardinalities and level names off the encoder -------------
    for name in CONTEXT_FIELDS:
        # Promotion to C happens here, so a promoted field is never also skipped
        # by include_env: the adjustment set is bias-relevant by definition.
        role = "C" if name in adj else ROLE_OF.get(name)
        if role is None or (role == "E" and not include_env):
            continue
        levels = enc.levels[name]
        fields.append(Field(
            name=name, role=role, kind="cat",
            cardinality=len(levels),
            levels=tuple(str(v) for v in levels),
        ))
    return CondSpec(tuple(fields))


def dose_probe(log10_conc_train: np.ndarray | torch.Tensor,
               n: int = 4096) -> dict[str, torch.Tensor]:
    """Probes for the generator's `calibrate_conditioning`.

    Purpose is to get loc/scale for `calibrate`, so conditionig across multiple fields is comparable. Thus it needs probes from real data to learn this.
    """
    x = torch.as_tensor(np.asarray(log10_conc_train, dtype=np.float32))
    x = x[torch.isfinite(x)]
    return {"dose": x[:n]}


def cond_from_arrays(spec: CondSpec, *, compound, log10_conc, is_control,
                     context, device=None) -> dict[str, torch.Tensor]:
    """The `{field: tensor}` dict the model expects, from parallel arrays.

    `context` is (N, F) codes in CONTEXT_FIELDS order;  fields are looked up BY NAME through CONTEXT_COL.
    A vehicle's dose is NaN (the learned null), never 0.0 or the log10 sentinel.
    """
    def _t(x, dtype=None):
        v = x if torch.is_tensor(x) else torch.as_tensor(x)
        if dtype is not None:
            v = v.to(dtype)
        return v.to(device) if device is not None else v

    ctx = _t(context)
    is_ctrl = _t(is_control)
    dose = _t(log10_conc, torch.float32)
    dose = torch.where(is_ctrl.bool(), torch.full_like(dose, float("nan")), dose)

    cond: dict[str, torch.Tensor] = {}
    for f in spec:
        if f.name == "compound":
            cond[f.name] = _t(compound, torch.long)
        elif f.name == "is_control":
            cond[f.name] = is_ctrl.long()
        elif f.name == "dose":
            if f.kind != "cont":
                raise ValueError("dose must be a continuous field on LINCS (decision 3)")
            cond[f.name] = dose
        else:
            cond[f.name] = ctx[:, CONTEXT_COL[f.name]].long()
    return cond


def cond_from_batch(batch, spec: CondSpec, device=None) -> dict[str, torch.Tensor]:
    """Training-batch wrapper around `cond_from_arrays`."""
    return cond_from_arrays( spec, compound=batch["compound_idx"], log10_conc=batch["log10_conc"],  is_control=batch["is_control"], context=batch["context"], device=device)
