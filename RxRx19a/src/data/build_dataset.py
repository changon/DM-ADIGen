"""One-time build of the tabular dataset.

Reads metadata.csv, builds a compound vocab, encodes covariates as a numeric vector, attaches per-site channel image paths, and saves an HF Arrow dataset to disk. 
Images are stored as *paths*, not pixels — loaded lazily at training.

Run:  python -m src.data.build_dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Allow `python -m src.data.build_dataset` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spec import (  # noqa: E402
    CONTEXT_FIELDS as _CONTEXT_FIELDS, DECLARED_LEVELS, CaseConfig,
    default_config)

CONTROL_COMPOUND_NAME = "__control__"
CONTROL_COMPOUND_IDX = 0  # index 0 is reserved for control

def build_compound_vocab(treatments: pd.Series, control_token: str) -> dict[str, int]:
    """Return {compound_name: int_idx}. Control gets index 0; rest sorted alphabetically."""
    unique = sorted(t for t in treatments.unique() if t != control_token)
    vocab = {CONTROL_COMPOUND_NAME: CONTROL_COMPOUND_IDX}
    for i, name in enumerate(unique, start=1):
        vocab[name] = i
    return vocab


def build_covariate_encoder(df: pd.DataFrame, cov_spec) -> dict:
    """Categorical -> one-hot column layout. 
    Metadata only; `apply_covariate_encoder` does the encoding, so the layout can be reused."""
    levels: dict[str, list[str]] = {}
    for col in cov_spec.categorical_cols:
        levels[col] = sorted_levels(df[col].unique().tolist())
    return {
        "categorical_cols": list(cov_spec.categorical_cols),
        "continuous_cols": list(cov_spec.continuous_cols),
        "levels": levels,
    }


def apply_covariate_encoder(df: pd.DataFrame, enc: dict) -> np.ndarray:
    """Return a (N, d) float32 array of one-hot covariates + continuous cols."""
    columns: list[np.ndarray] = []
    for col in enc["categorical_cols"]:
        cats = enc["levels"][col]
        col_vals = df[col].astype(str).values
        # one-hot via broadcast (avoid pandas.get_dummies to keep column order)
        oh = (col_vals[:, None] == np.array(cats)[None, :]).astype(np.float32)
        columns.append(oh)
    for col in enc["continuous_cols"]:
        columns.append(df[col].astype(np.float32).values.reshape(-1, 1))
    if not columns:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(columns, axis=1)


def covariate_encoder_path(cfg) -> str:
    return os.path.join(cfg.paths.nuisance_dir, "covariate_encoder.json")


def load_covariate_encoder(path: str, cov_spec) -> dict:
    """`covariate_encoder.json`, CHECKED against the current spec."""
    with open(path) as f:
        enc = json.load(f)
    want = tuple(cov_spec.categorical_cols)
    got = tuple(enc.get("categorical_cols", ()))
    if got != want:
        raise ValueError(
            f"covariate encoder at {path} was built for categorical_cols "
            f"{list(got)}, but spec.FIELDS now declares {list(want)}. cov_vec is "
            f"stale -- re-run `python -m src.data.build_dataset` to rebuild it.")
    return enc


def covariate_blocks(enc: dict) -> dict:
    """Field name -> its one-hot column indices in cov_vec, in layout order."""
    blocks, off = {}, 0
    for c in enc["categorical_cols"]:
        w = len(enc["levels"][c])
        blocks[c] = list(range(off, off + w))
        off += w
    return blocks


def channel_paths_for_row(row: pd.Series, images_root: str, suffixes) -> list[str]:
    """Build the 5 channel PNG paths for a single (experiment, plate, well, site)."""
    base = os.path.join(
        images_root,
        str(row["experiment"]),
        f"Plate{int(row['plate'])}",
    )
    return [
        os.path.join(base, f"{row['well']}_s{int(row['site'])}_{suf}.png")
        for suf in suffixes
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Verify a sample of image paths actually exist before saving.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only build the first N rows (smoke test).",
    )
    args = parser.parse_args()

    cfg: CaseConfig = default_config()
    paths = cfg.paths

    print(f"[build] reading {paths.metadata_csv}")
    df = pd.read_csv(paths.metadata_csv, dtype={"plate": int, "site": int})
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"[build] {len(df)} rows")

    # --- compound vocab ----------------------------------------------------
    treatment_col = cfg.action.categorical_col
    df[treatment_col] = df[treatment_col].fillna("").astype(str)
    vocab = build_compound_vocab(df[treatment_col], cfg.action.control_token)
    print(f"[build] compound vocab size: {len(vocab)} (incl. control)")

    # Control wells carry the empty treatment string (CONTROL_COMPOUND_IDX)
    df["compound_idx"] = (
        df[treatment_col]
        .where(df[treatment_col] != cfg.action.control_token, CONTROL_COMPOUND_NAME)
        .map(vocab)
        .astype(np.int32)
    )
    df["is_control"] = (df[treatment_col] == cfg.action.control_token).astype(np.int8)

    # --- continuous treatment (conc) ---------------------------------------
    conc_col = cfg.action.continuous_col
    raw_conc = df[conc_col].astype(str)
    # empty / control -> 0.0; we don't use this value for control rows.
    is_empty = raw_conc.isin(set(cfg.action.continuous_control_tokens))
    conc_float = pd.to_numeric(raw_conc.where(~is_empty, "0"), errors="coerce").fillna(0.0)
    df["conc"] = conc_float.astype(np.float32)
    if cfg.action.continuous_log10: # log10(conc), undefined for 0; controls keep a sentinel value of -10.
        log10_conc = np.where(
            df["conc"].values > 0,
            np.log10(np.clip(df["conc"].values, 1e-10, None)),
            -10.0,
        )
        df["log10_conc"] = log10_conc.astype(np.float32)

    # --- covariates --------------------------------------------------------
    # `edge` is derived, not a metadata column; see context.edge_labels.
    if "edge" in cfg.covariates.categorical_cols:
        df["edge"] = edge_labels(df["well"], cfg.covariates.edge_margin)
        print(f"[build] edge covariate: "
              f"{(df['edge'] == 'edge').mean():.1%} of wells on the plate edge")
    enc = build_covariate_encoder(df, cfg.covariates)
    cov_mat = apply_covariate_encoder(df, enc)
    print(f"[build] covariate vector dim: {cov_mat.shape[1]}")

    # --- image paths -------------------------------------------------------
    print("[build] assembling image paths ...")
    suffixes = cfg.image.channel_suffixes
    channel_paths = [
        channel_paths_for_row(row, paths.images_root, suffixes)
        for _, row in df.iterrows()
    ]

    if args.check_paths:
        print("[build] spot-checking 200 random image paths exist on disk ...")
        sample_idx = np.random.RandomState(0).choice(len(df), size=min(200, len(df)), replace=False)
        missing = []
        for i in sample_idx:
            for p in channel_paths[i]:
                if not os.path.isfile(p):
                    missing.append(p)
        if missing:
            print(f"[build] WARNING: {len(missing)} missing files among the sample. First 10:")
            for m in missing[:10]:
                print("   ", m)
        else:
            print("[build] all sampled paths exist")

    # --- assemble final table ---------------------------------------------
    keep_cols = [
        "site_id", "well_id", "cell_type", "experiment", "plate", "well", "site",
        "disease_condition", "treatment", "treatment_conc",
        "compound_idx", "is_control", "conc",
    ]
    if cfg.action.continuous_log10:
        keep_cols.append("log10_conc")
    out_df = df[keep_cols].copy()
    out_df["cov_vec"] = list(cov_mat)
    out_df["channel_paths"] = channel_paths

    # --- save --------------------------------------------------------------
    from datasets import Dataset

    os.makedirs(paths.tabular_dataset_dir, exist_ok=True)
    os.makedirs(paths.nuisance_dir, exist_ok=True)

    ds = Dataset.from_pandas(out_df, preserve_index=False)
    ds.save_to_disk(paths.tabular_dataset_dir)
    print(f"[build] saved HF dataset -> {paths.tabular_dataset_dir}")

    with open(os.path.join(paths.nuisance_dir, "compound_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    with open(os.path.join(paths.nuisance_dir, "covariate_encoder.json"), "w") as f:
        json.dump(enc, f, indent=2)
    print(f"[build] saved vocab + covariate encoder -> {paths.nuisance_dir}")


CONTEXT_FIELDS = _CONTEXT_FIELDS

_WELL_RE = re.compile(r"^([A-Z]+)(\d+)$")


def sorted_levels(values) -> list:
    """Ordered level list for a categorical field. sorting `plate` as strings in one and ints in other"""
    uniq = list(dict.fromkeys(str(v) for v in values))
    try:
        return sorted(uniq, key=lambda v: (0, float(v)))
    except (TypeError, ValueError):
        return sorted(uniq)


@dataclass
class ContextEncoder:
    """Maps metadata rows -> (F,) integer context codes."""

    levels: dict[str, list]          # field -> ordered level list (categoricals)

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.levels[f]) for f in CONTEXT_FIELDS]

    # -- build / persist ----------------------------------------------------
    @classmethod
    def build(cls, df: pd.DataFrame) -> "ContextEncoder":
        rows, cols = _well_row_col(df["well"])
        levels = {
            "cell_type": sorted(pd.unique(df["cell_type"]).tolist()),
            "experiment": sorted(pd.unique(df["experiment"]).tolist()),
            "plate": sorted(int(p) for p in pd.unique(df["plate"])),
            "well_row": sorted(set(rows.tolist())),
            "well_col": sorted(set(int(c) for c in cols.tolist())),
            "site": sorted(int(s) for s in pd.unique(df["site"])),
        }
        # levels: the set is fixed by definition and its ORDER is the index a checkpoint was trained on.
        levels.update({k: list(v) for k, v in DECLARED_LEVELS.items()})
        missing = [f for f in CONTEXT_FIELDS if f not in levels]
        if missing:
            raise KeyError(
                f"spec.FIELDS declares {missing}, which ContextEncoder.build does "
                f"not know how to enumerate levels for. Add it to `levels` above "
                f"(and to `raw` in encode).")
        return cls(levels=levels)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"fields": list(CONTEXT_FIELDS), "levels": self.levels}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ContextEncoder":
        with open(path) as f:
            d = json.load(f)
        if tuple(d["fields"]) != CONTEXT_FIELDS:
            raise ValueError(
                f"context encoder at {path} was built for fields {d['fields']}, "
                f"but the code now expects {list(CONTEXT_FIELDS)}"
            )
        return cls(levels=d["levels"])

    @classmethod
    def load_or_build(cls, cfg) -> "ContextEncoder":
        path = context_encoder_path(cfg)
        if os.path.isfile(path):
            return cls.load(path)
        enc = cls.build(pd.read_csv(cfg.paths.metadata_csv))
        enc.save(path)
        return enc

    # -- encode -------------------------------------------------------------
    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """(N, F) int64 context codes. Unseen levels map to 0"""
        rows, cols = _well_row_col(df["well"])
        raw = {
            "cell_type": df["cell_type"].tolist(),
            "experiment": df["experiment"].tolist(),
            "plate": [int(p) for p in df["plate"]],
            "well_row": rows.tolist(),
            "well_col": [int(c) for c in cols.tolist()],
            "site": [int(s) for s in df["site"]],
            "disease": [disease_level(v) for v in df["disease_condition"]],
            "edge": edge_labels(df["well"]).tolist(),
        }
        missing = [f for f in CONTEXT_FIELDS if f not in raw]
        if missing:
            raise KeyError(
                f"spec.FIELDS declares {missing}, which ContextEncoder.encode does "
                f"not know how to compute. Add it to `raw` above.")
        out = np.zeros((len(df), len(CONTEXT_FIELDS)), dtype=np.int64)
        for j, f in enumerate(CONTEXT_FIELDS):
            lut = {lv: i for i, lv in enumerate(self.levels[f])}
            out[:, j] = [lut.get(v, 0) for v in raw[f]]
        return out

    def encode_row(self, row) -> np.ndarray:
        """Single HF-dataset row (a dict) -> (F,) int64."""
        return self.encode(pd.DataFrame([{
            "cell_type": row["cell_type"], "experiment": row["experiment"],
            "plate": row["plate"], "well": row["well"], "site": row["site"],
            "disease_condition": row["disease_condition"],
        }]))[0]


def disease_level(v) -> str:
    """Raw disease_condition cell -> a declared `disease` level. Missing -> 'Unlabeled'."""
    s = str(v)
    if s in DECLARED_LEVELS["disease"]:
        return s
    if s in ("nan", "None", "", "NaN"):
        return "Unlabeled"
    return "Unlabeled"


def _well_row_col(well: "pd.Series") -> tuple[np.ndarray, np.ndarray]:
    r, c = [], []
    for w in well.astype(str):
        m = _WELL_RE.match(w.strip())
        if m:
            r.append(m.group(1)); c.append(int(m.group(2)))
        else:
            r.append("?"); c.append(0)
    return np.array(r), np.array(c)


def well_row_col(well) -> tuple[np.ndarray, np.ndarray]:
    """(row letters, column numbers) for a well Series."""
    return _well_row_col(pd.Series(well).astype(str))

def edge_labels(well, margin: int = 2) -> np.ndarray:
    """(N,) of "edge" / "interior" from well position. The outer `margin` rows/columns of the plate are the edge. See CovariateSpec.edge_margin. """
    rows, cols = well_row_col(well)
    r_levels = sorted(set(rows.tolist()))
    ri = np.array([r_levels.index(v) for v in rows])
    c = cols.astype(np.int64)
    is_edge = ((ri < margin) | (ri >= len(r_levels) - margin)
               | (c < c.min() + margin) | (c > c.max() - margin))
    return np.where(is_edge, "edge", "interior")


def context_encoder_path(cfg) -> str:
    return os.path.join(os.path.dirname(cfg.paths.tabular_dataset_dir), "context_encoder.json")


def context_for_rows(cfg, meta, pick: np.ndarray) -> np.ndarray:
    """(n, F) context codes for `pick` rows of the HF tabular dataset `meta`.

    Eval uses this to generate images at contexts from the empirical p(C, E) --- the marginalization
    """
    enc = ContextEncoder.load_or_build(cfg)
    df = pd.DataFrame({
        "cell_type": np.asarray(meta["cell_type"])[pick],
        "experiment": np.asarray(meta["experiment"])[pick],
        "plate": np.asarray(meta["plate"])[pick],
        "well": np.asarray(meta["well"])[pick],
        "site": np.asarray(meta["site"])[pick],
        "disease_condition": np.asarray(
            [str(x) for x in meta["disease_condition"]])[pick],
    })
    return enc.encode(df)

if __name__ == "__main__":
    main()
