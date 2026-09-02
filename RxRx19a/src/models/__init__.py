"""Generator construction for the RxRx19a diffusion model.

    model(sample, timestep, cond, drop=None) -> object with `.sample`

`DiT2DModel` (see `dit.py`), an adaLN-Zero Diffusion Transformer.
What it conditions on is described by a `CondSpec` (see `conditioning.py`)

---------------------------
Training writes an `arch.json` next to the checkpoints holding the complete spec -- fields, roles, kinds, vocabularies, standardisation constants
`build_generator_from_ckpt` reads it and rebuilds.

"""
from __future__ import annotations

import json
import os
from typing import Any

import torch

from .conditioning import CondEmbedder, CondSpec, Field
from .dit import COND_MODES, DIT_SIZES, DiT2DModel

__all__ = [
    "ARCH_FILENAME",
    "COND_MODES",
    "DIT_SIZES",
    "CondEmbedder",
    "CondSpec",
    "DiT2DModel",
    "Field",
    "build_generator",
    "build_generator_from_ckpt",
    "read_arch_spec",
    "write_arch_spec",
]

ARCH_FILENAME = "arch.json"


def build_generator(
    cfg,
    cond_spec: CondSpec,
    *,
    dit_size: str = "B",
    patch_size: int = 8,
    class_dropout_prob: float = 0.0,
    cond_mode: str = "adaln",
    n_channels: int | None = None,
    resolution: int | None = None,
) -> torch.nn.Module:
    """Build the denoiser.

    `cond_spec` is the conditioning contract and is required. Specifies:
    - what embedding tables and MLPs exist, how wide each one is, 
    - which fields CFG may null
    - how continuous fields are standardised. 

    `n_channels` / `resolution` override the pixel setup,  permitting latent modeling.
        - denoises (80, 16, 16) VAE latents rather than (5, 128, 128) pixels 
    """
    n_channels = cfg.image.n_channels if n_channels is None else int(n_channels)
    resolution = cfg.image.resolution if resolution is None else int(resolution)
    if dit_size not in DIT_SIZES:
        raise ValueError(f"dit_size must be one of {sorted(DIT_SIZES)}, got {dit_size!r}")

    n_tokens = (resolution // patch_size) ** 2
    if n_tokens < 16:
        raise ValueError(
            f"resolution={resolution} with patch_size={patch_size} gives only "
            f"{n_tokens} tokens. That is almost certainly a latent run left on the "
            f"pixel default; lower patch_size.")

    return DiT2DModel(
        sample_size=resolution,
        in_channels=n_channels,        # image channels only -- conditioning is
        out_channels=n_channels,       # passed separately, not in the channel dim
        cond_spec=cond_spec,
        patch_size=patch_size,
        class_dropout_prob=class_dropout_prob,
        cond_mode=cond_mode,
        **DIT_SIZES[dit_size],
    )


def arch_spec(
    cond_spec: CondSpec,
    *,
    dit_size: str,
    patch_size: int,
    class_dropout_prob: float,
    n_channels: int,
    resolution: int,
    cond_mode: str = "adaln",
    **extra: Any,
) -> dict[str, Any]:
    """The dict training records in arch.json. Everything `build_generator` needs."""
    return dict(
        arch="dit",
        cond_spec=cond_spec.to_list(),
        dit_size=dit_size,
        patch_size=int(patch_size),
        class_dropout_prob=float(class_dropout_prob),
        cond_mode=str(cond_mode),
        n_channels=int(n_channels),
        resolution=int(resolution),
        **extra,
    )


def write_arch_spec(ckpt_root: str, spec: dict[str, Any]) -> str:
    """Record the architecture of a training run next to its checkpoints.

    Written via a temp file + `os.replace` to avoid concurrency or crashes during requeues
    """
    os.makedirs(ckpt_root, exist_ok=True)
    path = os.path.join(ckpt_root, ARCH_FILENAME)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(spec, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_arch_spec(ckpt_dir: str) -> dict[str, Any] | None:
    """Load the arch spec for a checkpoint, or None if there is none.

    Accepts either the run root or a `checkpoint-NNNN` dir inside it (for eval scripts)
    """
    for d in (ckpt_dir, os.path.dirname(os.path.normpath(ckpt_dir))):
        path = os.path.join(d, ARCH_FILENAME)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return None


def build_generator_from_ckpt(cfg, ckpt_dir: str) -> torch.nn.Module:
    """Rebuild the denoiser a checkpoint was trained with.
    """
    spec = read_arch_spec(ckpt_dir)
    if spec is None:
        raise FileNotFoundError(
            f"no {ARCH_FILENAME} for checkpoint {ckpt_dir!r} (looked there and in "
            f"its parent). Training writes it next to the checkpoints; without it "
            f"the architecture cannot be recovered.")

    if spec.get("arch") != "dit":
        raise ValueError(
            f"{ckpt_dir}: arch.json says arch={spec.get('arch')!r}. Only 'dit' is "
            f"supported; the UNet backbone was removed on 2026-08-26.")

    missing = [k for k in ("cond_spec", "dit_size", "patch_size",
                           "class_dropout_prob", "n_channels", "resolution")
               if k not in spec]
    if missing:
        raise KeyError(
            f"{ckpt_dir}: arch.json is missing {missing}. It predates the spec-based "
            f"layout and cannot be rebuilt -- rebuilding it under current defaults "
            f"would load its weights into a different architecture.")

    return build_generator(
        cfg,
        CondSpec.from_list(spec["cond_spec"]),
        dit_size=spec["dit_size"],
        patch_size=int(spec["patch_size"]),
        class_dropout_prob=float(spec["class_dropout_prob"]),
        # absent in pre-xattn arch.json; those runs are all adaln
        cond_mode=str(spec.get("cond_mode", "adaln")),
        n_channels=int(spec["n_channels"]),
        resolution=int(spec["resolution"]),
    )
