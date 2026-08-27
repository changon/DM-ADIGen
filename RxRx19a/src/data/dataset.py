"""Multi-channel image Dataset for RxRx19a.

Reads the HF tabular dataset produced by build_dataset.py.
Returns batches for models
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.build_dataset import CONTEXT_FIELDS, ContextEncoder
from src.models.conditioning import CondSpec, Field
from src.spec import (
    ACTION_FIELDS, BY_NAME, CONTEXT_COL, ROLE_OF, CaseConfig, ImageSpec)


class RxRx19aDataset(Dataset):
    """Returns:
        image:        (C, H, W) float in [-1, 1], C = n_channels
        compound_idx: int64 scalar
        conc:         float32 scalar (raw)
        log10_conc:   float32 scalar (or sentinel for control)
        is_control:   int8 scalar
        infected:     int8 scalar (1 iff the row is in cfg.population.disease_condition)
        cov_vec:      (d_cov,) float32 covariate features
    Set `load_images=False` to skip image I/O (used by the nuisance fitter).
    """

    def __init__(
        self,
        cfg: CaseConfig,
        indices: Optional[np.ndarray] = None,
        load_images: bool = True,
        latents: Optional["LatentSpec"] = None,
    ):
        """`latents`: if given, `image` is the precomputed VAE latent (80,16,16) instead of the pixel image (5,128,128). 
        """
        self.cfg = cfg
        self.load_images = load_images
        ds = load_from_disk(cfg.paths.tabular_dataset_dir)
        self._row_ids = (np.arange(len(ds)) if indices is None
                         else np.asarray(indices, dtype=np.int64))
        if indices is not None:
            ds = ds.select(indices.tolist())
        self.ds = ds
        self.latent_spec = latents
        self._latents = None
        if latents is not None:
            self._latents = open_latents(latents)     # memmap (N_all, 80, 16, 16)

        # Context codes precomputed once for the whole split
        enc = ContextEncoder.load_or_build(cfg)
        self.context_cardinalities = enc.cardinalities
        self._context = enc.encode(pd.DataFrame({
            "cell_type": ds["cell_type"],
            "experiment": ds["experiment"],
            "plate": ds["plate"],
            "well": ds["well"],
            "site": ds["site"],
            "disease_condition": [str(x) for x in ds["disease_condition"]],
        }))                                   # (N, F) int64

        self.resize = transforms.Resize(
            (cfg.image.resolution, cfg.image.resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

        mean = cfg.image.normalize_mean or [0.5] * cfg.image.n_channels
        std = cfg.image.normalize_std or [0.5] * cfg.image.n_channels
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.ds)

    def _load_image(self, channel_paths: list[str]) -> torch.Tensor:
        chans = []
        for p in channel_paths:
            img = Image.open(p)
            if img.mode != "L": # 8-bit grayscale; force "L" for a single-channel tensor.
                img = img.convert("L")
            t = transforms.functional.pil_to_tensor(img).float() / 255.0  # (1,H,W)
            chans.append(t)
        x = torch.cat(chans, dim=0)  # (C, H, W)
        x = self.resize(x.unsqueeze(0)).squeeze(0)
        x = (x - self.mean) / self.std  # roughly [-1, 1]
        return x

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        out: dict[str, torch.Tensor] = {
            "compound_idx": torch.tensor(row["compound_idx"], dtype=torch.long),
            "conc": torch.tensor(row["conc"], dtype=torch.float32),
            "is_control": torch.tensor(row["is_control"], dtype=torch.int8),
            "infected": torch.tensor(
                int(str(row["disease_condition"]) == self.cfg.population.disease_condition),
                dtype=torch.int8,
            ),
            "cov_vec": torch.tensor(row["cov_vec"], dtype=torch.float32),
            "context": torch.as_tensor(self._context[idx], dtype=torch.long),       # Context the generator conditions on; see src/data/context.py for the field order.
        }
        if "log10_conc" in row:
            out["log10_conc"] = torch.tensor(row["log10_conc"], dtype=torch.float32)
        if self.load_images:
            if self._latents is not None: # by ORIGINAL row id, stored fp16; upcast for fp32/amp.
                z = self._latents[self._row_ids[idx]]
                z = torch.from_numpy(np.asarray(z, dtype=np.float32))
                out["image"] = normalize_latents(z, self.latent_spec) # No-op unless the spec carries per-channel stats.
            else:
                out["image"] = self._load_image(row["channel_paths"])
        return out


def get_dataloader(
    cfg: CaseConfig,
    batch_size: int,
    *,
    indices: Optional[np.ndarray] = None,
    load_images: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    ds = RxRx19aDataset(cfg, indices=indices, load_images=load_images)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ===========================================================================
# LATENT REPRESENTATION  (was src/data/latents.py, merged 2026-08-26)
#
# The DiT denoises VAE latents, not pixels. ImageSpec.latent_vae is the only
# CHOICE here; z_per_channel, resolution, scaling_factor and the per-channel
# stats are all MEASURED at encode time and persisted next to the latents.
# ===========================================================================



# Declared in spec.ImageSpec.latent_vae -- re-exported so the CLI default and the
# config cannot disagree.

DEFAULT_VAE = ImageSpec().latent_vae

@dataclass
class LatentSpec:
    path: str
    n_images_channels: int   # fluorescence channels; mirrors ImageSpec.n_channels (5)
    z_per_channel: int       # VAE latent channels per fluorescence channel (16 for f8-d16)
    resolution: int          # latent H = W (16)
    vae: str
    scaling_factor: float
    # Per-channel standardization stats, len == n_channels; None = single-scalar scaling
    channel_mean: "list[float] | None" = None
    channel_std: "list[float] | None" = None

    @property
    def n_channels(self) -> int:
        """Channels the DiT actually denoises."""
        return self.n_images_channels * self.z_per_channel   # 5 * 16 = 80

    @property
    def normalized(self) -> bool:
        return self.channel_mean is not None and self.channel_std is not None

    @classmethod
    def load(cls, path: str) -> "LatentSpec":
        with open(_meta_path(path)) as f:
            d = json.load(f)
        d["path"] = path # `path` in the meta is where the array was written
        return cls(**d)

    def save(self) -> None:
        with open(_meta_path(self.path), "w") as f:
            json.dump(self.__dict__, f, indent=2)


def _meta_path(path: str) -> str:
    return os.path.splitext(path)[0] + "_meta.json"


def default_latent_path(cfg) -> str:
    return os.path.join(os.path.dirname(cfg.paths.tabular_dataset_dir),
                        "latents.npy")


def load_vae(name: str = DEFAULT_VAE, device="cuda"):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(name).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_images(vae, imgs: torch.Tensor, device) -> torch.Tensor:
    """(B, 5, H, W) in [-1,1]  ->  (B, 20, H/8, W/8) scaled latents."""
    s = float(vae.config.scaling_factor)
    zs = []
    for c in range(imgs.shape[1]):
        x = imgs[:, c : c + 1].to(device).repeat(1, 3, 1, 1)
        zs.append(vae.encode(x).latent_dist.mean * s)
    return torch.cat(zs, dim=1)


@torch.no_grad()
def decode_latents(vae, z: torch.Tensor, n_image_channels: int = 5) -> torch.Tensor:
    """(B, 20, h, w) scaled latents -> (B, 5, H, W) images in [-1,1].
    """
    s = float(vae.config.scaling_factor)
    zpc = z.shape[1] // n_image_channels
    outs = []
    for c in range(n_image_channels):
        zc = z[:, c * zpc : (c + 1) * zpc] / s
        d = vae.decode(zc).sample                       # (B,3,H,W)
        outs.append(d.mean(dim=1, keepdim=True))
    return torch.cat(outs, dim=1).clamp(-1, 1)


def open_latents(spec: LatentSpec, mode: str = "r") -> np.memmap:
    """Memmap the precomputed latents: (N, 20, 16, 16) float16."""
    return np.load(spec.path, mmap_mode=mode)

# Standardizing per channel puts the whole variance budget on image content. 
def compute_channel_stats(spec: LatentSpec, stride: int = 50,  indices: "np.ndarray | None" = None):
    """Per-channel (mean, std) over the stored latents. `stride` subsamples rows"""
    z = open_latents(spec)
    sub = z[indices] if indices is not None else z[::stride]
    sub = np.asarray(sub, dtype=np.float32)
    mean = sub.mean(axis=(0, 2, 3))
    std = sub.std(axis=(0, 2, 3))
    return mean, np.maximum(std, 1e-6)


def _stat_tensors(spec: LatentSpec, like: torch.Tensor):
    """(mean, std) shaped to broadcast over (B, C, H, W) or (C, H, W)."""
    shape = (1, -1, 1, 1) if like.dim() == 4 else (-1, 1, 1)
    kw = {"dtype": like.dtype, "device": like.device}
    return (torch.tensor(spec.channel_mean, **kw).view(*shape),
            torch.tensor(spec.channel_std, **kw).view(*shape))


def normalize_latents(z: torch.Tensor, spec: LatentSpec) -> torch.Tensor:
    """Stored latent -> what the model denoises. No-op on a legacy spec."""
    if not spec.normalized:
        return z
    mean, std = _stat_tensors(spec, z)
    return (z - mean) / std


def denormalize_latents(z: torch.Tensor, spec: LatentSpec) -> torch.Tensor:
    """Inverse of `normalize_latents`; apply before `decode_latents`."""
    if not spec.normalized:
        return z
    mean, std = _stat_tensors(spec, z)
    return z * std + mean

def build_cond_spec(
    cfg,
    n_compounds: int,
    log10_conc_train: np.ndarray | torch.Tensor,
    *,
    include_env: bool = False,
    adjustment_set: Sequence[str] | None = None,
) -> CondSpec:
    """Cond spec.

    `log10_conc_train` is the raw dose column of the training, treated rows. Sets `loc`/`scale`.

    Cardinalities and level names come from `ContextEncoder`

    `include_env` defaults to false.

    `adjustment_set` promotes named fields from their default role to C.
    """
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
            fields.append(Field(name=name, role="A", kind="cont",
                                loc=loc, scale=scale, nullable=d.nullable))
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
    """Probes for `DiT2DModel.calibrate_conditioning`.

    Purpose is to get loc/scale for `calibrate`, so conditionig across multiple fields is comparable. Thus it needs probes from real data to learn this.
    """
    x = torch.as_tensor(np.asarray(log10_conc_train, dtype=np.float32))
    x = x[torch.isfinite(x)]
    return {"dose": x[:n]}


def cond_from_arrays(spec: CondSpec, *, compound, log10_conc, is_control,
                     context, device=None) -> dict[str, torch.Tensor]:
    """The `{field: tensor}` dict the model expects, from parallel arrays.

    `context` is (N, F) codes in CONTEXT_FIELDS order; 
    fields are looked up BY NAME through CONTEXT_COL. 
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
            cond[f.name] = dose
        else:
            cond[f.name] = ctx[:, CONTEXT_COL[f.name]].long()
    return cond


def cond_from_batch(batch, spec: CondSpec, device=None) -> dict[str, torch.Tensor]:
    """Training-batch wrapper around `cond_from_arrays`."""
    return cond_from_arrays(
        spec, compound=batch["compound_idx"], log10_conc=batch["log10_conc"],
        is_control=batch["is_control"], context=batch["context"], device=device)
