"""Generator construction for LINCS (IMPLEMENT.md §3.4). Adapted from
RxRx19a/src/models/__init__.py (always copy, §3.1.1).

    model(sample, timestep, cond, drop=None) -> object with `.sample`

Two backbones share that contract and the `CondSpec` conditioning:

    arch="mlp"    MLPDenoiser  (mlp.py)     the headline backbone (decision 4)
    arch="dit1d"  DiT1DModel   (dit1d.py)   1-d DiT over gene patches (ablation)

---------------------------
Training writes an `arch.json` next to the checkpoints with everything the
denoiser needs to be rebuilt: `arch`, the RESOLVED constructor kwargs
(`arch_kwargs`: hidden / depth / heads / patch, not just a size letter, so
re-tuning a size table never orphans a checkpoint), `n_genes`,
`class_dropout_prob` and the full `cond_spec` (fields, roles, vocabularies,
standardisation constants). `build_generator_from_ckpt` reads that file only.
A 2-d RxRx checkpoint (arch="dit") is rejected.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch

from .conditioning import CondEmbedder, CondSpec, Field
from .dit1d import DIT_SIZES, DiT1DModel, patch_geometry
from .mlp import MLP_SIZES, MLPDenoiser

__all__ = [
    "ARCHS",
    "ARCH_FILENAME",
    "DIT_SIZES",
    "MLP_SIZES",
    "CondEmbedder",
    "CondSpec",
    "DiT1DModel",
    "Field",
    "MLPDenoiser",
    "arch_spec",
    "build_generator",
    "build_generator_from_ckpt",
    "read_arch_spec",
    "resolve_arch_kwargs",
    "write_arch_spec",
]

ARCH_FILENAME = "arch.json"
ARCHS = ("mlp", "dit1d")

# Constructor kwargs each arch's arch_kwargs must hold exactly.
_ARCH_KWARGS = {
    "mlp": ("hidden_size", "depth", "mlp_ratio"),
    "dit1d": ("hidden_size", "depth", "num_heads", "mlp_ratio", "patch_size"),
}


def resolve_arch_kwargs(arch: str, size: str, *, patch_size: int | None = None) -> dict[str, Any]:
    """The constructor kwargs of `arch` at `size` (MLP_SIZES / DIT_SIZES)."""
    if arch == "mlp":
        if size not in MLP_SIZES:
            raise ValueError(f"mlp size must be one of {sorted(MLP_SIZES)}, got {size!r}")
        return dict(MLP_SIZES[size], mlp_ratio=1.0)
    if arch == "dit1d":
        if size not in DIT_SIZES:
            raise ValueError(f"dit1d size must be one of {sorted(DIT_SIZES)}, got {size!r}")
        if patch_size is None:
            raise ValueError("dit1d needs patch_size")
        return dict(DIT_SIZES[size], mlp_ratio=4.0, patch_size=int(patch_size))
    raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")


def build_generator(
    cond_spec: CondSpec,
    *,
    arch: str,
    n_genes: int,
    arch_kwargs: dict[str, Any],
    class_dropout_prob: float = 0.0,
) -> torch.nn.Module:
    """Build the denoiser.

    `cond_spec` is the conditioning contract and is required. Specifies:
    - what embedding tables and MLPs exist, how wide each one is,
    - which fields CFG may null
    - how continuous fields are standardised.

    `arch_kwargs` are the resolved constructor kwargs (`resolve_arch_kwargs`).
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    want = set(_ARCH_KWARGS[arch])
    if set(arch_kwargs) != want:
        raise ValueError(f"arch_kwargs for {arch} must be exactly {sorted(want)}, got {sorted(arch_kwargs)}")
    cls = MLPDenoiser if arch == "mlp" else DiT1DModel
    return cls(n_genes=int(n_genes), cond_spec=cond_spec,
               class_dropout_prob=float(class_dropout_prob), **arch_kwargs)


def arch_spec(
    cond_spec: CondSpec,
    *,
    arch: str,
    size: str,
    n_genes: int,
    arch_kwargs: dict[str, Any],
    class_dropout_prob: float,
    **extra: Any,
) -> dict[str, Any]:
    """The dict training records in arch.json. Everything `build_generator`
    needs, plus the derived patch geometry (dit1d) and the trainer's extras."""
    spec = dict(
        arch=str(arch),
        size=str(size),
        n_genes=int(n_genes),
        arch_kwargs=dict(arch_kwargs),
        class_dropout_prob=float(class_dropout_prob),
        cond_mode="adaln",             # xattn is not ported to LINCS
        cond_spec=cond_spec.to_list(),
    )
    if arch == "dit1d":
        spec["patch_geometry"] = patch_geometry(n_genes, arch_kwargs["patch_size"])
    clash = sorted(set(spec) & set(extra))
    if clash:
        raise ValueError(f"extra arch.json keys {clash} would overwrite the rebuild keys")
    spec.update(extra)
    return spec


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


def build_generator_from_ckpt(ckpt_dir: str) -> torch.nn.Module:
    """Rebuild the denoiser a checkpoint was trained with, from arch.json alone.

    Checking the checkpoint's gene order (`gene_pr_ids`) against the data it is
    evaluated on is the caller's job (eval).
    """
    spec = read_arch_spec(ckpt_dir)
    if spec is None:
        raise FileNotFoundError(
            f"no {ARCH_FILENAME} for checkpoint {ckpt_dir!r} (looked there and in "
            f"its parent). Training writes it next to the checkpoints; without it "
            f"the architecture cannot be recovered.")

    arch = spec.get("arch")
    if arch not in ARCHS:
        raise ValueError(
            f"{ckpt_dir}: arch.json says arch={arch!r}; LINCS builds {ARCHS}. "
            f"A 2-d RxRx19a DiT checkpoint (arch='dit') is not a LINCS generator.")

    missing = [k for k in ("cond_spec", "n_genes", "arch_kwargs", "class_dropout_prob")
               if k not in spec]
    if missing:
        raise KeyError(
            f"{ckpt_dir}: arch.json is missing {missing}; rebuilding under current "
            f"defaults would load its weights into a different architecture.")

    model = build_generator(
        CondSpec.from_list(spec["cond_spec"]),
        arch=arch,
        n_genes=int(spec["n_genes"]),
        arch_kwargs=dict(spec["arch_kwargs"]),
        class_dropout_prob=float(spec["class_dropout_prob"]),
    )
    # the recorded derived facts must agree with the rebuilt model
    if "gene_pr_ids" in spec and len(spec["gene_pr_ids"]) != model.n_genes:
        raise ValueError(f"{ckpt_dir}: arch.json lists {len(spec['gene_pr_ids'])} genes but n_genes={model.n_genes}")
    if arch == "dit1d" and "patch_geometry" in spec:
        got = {"pad": model.pad, "n_tokens": model.n_tokens}
        if {k: int(v) for k, v in spec["patch_geometry"].items()} != got:
            raise ValueError(f"{ckpt_dir}: arch.json patch_geometry {spec['patch_geometry']} != rebuilt {got}")
    return model
