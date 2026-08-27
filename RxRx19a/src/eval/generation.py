"""Generation primitives shared by the evaluation suite.

    _build_model / _load_weights   rebuild a checkpoint from its arch.json
    LatentCtx                              latent-space denoising + decode to pixels
    SampleTarget / _sample_targets_from_real   actions at real decision contexts
    _generate_batch                        sample images, optionally CFG-guided
    _load_real_images                      real rows as (N, C, H, W) in [-1, 1]

The architecture and the conditioning contract both come from the checkpoint's arch.json
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from safetensors.torch import load_file as safetensors_load_file
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_dataset import CONTEXT_FIELDS, context_for_rows  # noqa: E402
from src.data.dataset import RxRx19aDataset  # noqa: E402
from src.data.dataset import cond_from_arrays  # noqa: E402
from src.models import build_generator_from_ckpt  # noqa: E402
from src.spec import CaseConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Model rebuild
# ---------------------------------------------------------------------------

def _build_model(cfg: CaseConfig, ckpt_dir: str) -> torch.nn.Module:
    """Rebuild the denoiser a checkpoint was trained with.

    Everything — DiT size, patch size, image channels, and the full `CondSpec` (fields, roles, vocabularies, dose loc/scale) — comes from the `arch.json` that training writes next to the checkpoints.
    """
    return build_generator_from_ckpt(cfg, ckpt_dir)


def _cond_spec(model: torch.nn.Module):
    """The conditioning contract of a (possibly DDP-wrapped) model."""
    return getattr(model, "module", model).cond_spec


class LatentCtx:
    """Everything needed to denoise in latent space and come back to pixels. Built from arch.json
    """

    def __init__(self, spec, vae, n_image_channels: int, latent_norm=None):
        self.spec = spec
        self.vae = vae
        self.n_image_channels = n_image_channels
        # Does the MODEL denoise per-channel-standardized latents? `None` means
        # "follow the spec" (right for anything reading latents off disk);
        # `from_ckpt` pins it from arch.json so a checkpoint trained before the
        # standardization change still decodes the way it was trained.
        self.latent_norm = (spec.normalized if latent_norm is None
                            else bool(latent_norm))

    @classmethod
    def from_ckpt(cls, cfg, ckpt_dir: str, device) -> "LatentCtx | None":
        from src.models import read_arch_spec
        a = read_arch_spec(ckpt_dir) or {}
        if not a.get("latent"):
            return None
        from src.data.dataset import LatentSpec, default_latent_path, load_vae
        spec = LatentSpec.load(default_latent_path(cfg))
        if a.get("vae") and a["vae"] != spec.vae:
            raise RuntimeError(
                f"checkpoint was trained on latents from {a['vae']!r} but the "
                f"latents on disk were encoded with {spec.vae!r}"
            )
        return cls(spec, load_vae(spec.vae, device), cfg.image.n_channels,
                   latent_norm=bool(a.get("latent_norm", False)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        from src.data.dataset import decode_latents, denormalize_latents
        if self.latent_norm:
            z = denormalize_latents(z, self.spec)
        return decode_latents(self.vae, z, self.n_image_channels)


def _supports_cfg_null(model: torch.nn.Module) -> bool:
    """True if the model was trained with CFG dropout, i.e. it has a real unconditional (no-compound, no-dose) branch to guide away from."""
    m = getattr(model, "module", model)  # unwrap DDP/accelerate
    return float(getattr(m, "class_dropout_prob", 0.0)) > 0.0


def _load_weights(model: torch.nn.Module, ckpt_dir: str, which: str) -> None:
    fname = "model.safetensors" if which == "train" else "model_1.safetensors"
    path = os.path.join(ckpt_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    state = safetensors_load_file(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch loading {path}. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


# ---------------------------------------------------------------------------
# Sampling driver
# ---------------------------------------------------------------------------

@dataclass
class SampleTarget:
    """One action (A) at one decision context (C, E).

    `context` is a full row of `ContextEncoder` codes in `CONTEXT_FIELDS` order
    """

    compound_idx: int
    log10_conc: float
    is_control: int              # 0/1
    context: tuple[int, ...]     # (F,) ContextEncoder codes, CONTEXT_FIELDS order
    infected: int = 1            # 0/1; alpha-side bookkeeping, never model input

    def __post_init__(self):
        if len(self.context) != len(CONTEXT_FIELDS):
            raise ValueError(
                f"SampleTarget.context has {len(self.context)} entries, expected "
                f"{len(CONTEXT_FIELDS)} (one per CONTEXT_FIELDS). Pass a row of "
                f"context_for_rows(cfg, meta, pick), not individual field indices.")


def _targets_from_rows(cfg: CaseConfig, ds, pick: np.ndarray) -> list[SampleTarget]:
    """SampleTargets for `pick` rows of an HF tabular dataset, in order."""
    compound_idx = np.asarray(ds["compound_idx"], dtype=np.int64)[pick]
    log10_conc = np.asarray(ds["log10_conc"], dtype=np.float32)[pick]
    is_control = np.asarray(ds["is_control"], dtype=np.int64)[pick]
    dis_str = np.array([str(x) for x in ds["disease_condition"]])[pick]
    infected = (dis_str == cfg.population.disease_condition).astype(np.int64)
    ctx = context_for_rows(cfg, ds, pick)
    return [
        SampleTarget(int(c), float(l), int(ic), tuple(int(v) for v in row), int(inf))
        for c, l, ic, row, inf in zip(
            compound_idx, log10_conc, is_control, ctx, infected)
    ]


def _sample_targets_from_real(
    cfg: CaseConfig, indices: np.ndarray, n_targets: int, seed: int,
) -> list[SampleTarget]:
    """Draw n_targets actions-at-contexts uniformly with replacement from the rows referenced by `indices` in the HF dataset.
    """
    ds = load_from_disk(cfg.paths.tabular_dataset_dir).select(indices.tolist())
    rng = np.random.default_rng(seed)
    return _targets_from_rows(cfg, ds, rng.integers(0, len(ds), size=n_targets))


@torch.no_grad()
def _generate_batch(
    model: torch.nn.Module,
    scheduler,
    cfg: CaseConfig,
    targets: list[SampleTarget],
    n_inference_steps: int,
    device: torch.device,
    seed: int,
    guidance_scale: float = 1.0,
    latent_ctx: "LatentCtx | None" = None,
) -> torch.Tensor:
    """Generate one batch of images for the provided targets. 
    Returns (B, 5, 128, 128) in [-1, 1] -- ALWAYS pixels.

    If `latent_ctx` is given the diffusion runs in VAE latent space (80,16,16) and the result is decoded back to pixels before returning
    """
    B = len(targets)
    if latent_ctx is not None:
        H = W = latent_ctx.spec.resolution          # 16
        n_ch = latent_ctx.spec.n_channels           # 80
    else:
        H = W = cfg.image.resolution                # 128
        n_ch = cfg.image.n_channels                 # 5
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, n_ch, H, W, generator=g, device=device)

    spec = _cond_spec(model)
    compound = torch.tensor([t.compound_idx for t in targets], dtype=torch.long, device=device)
    lx = torch.tensor([t.log10_conc for t in targets], dtype=torch.float32, device=device)
    ic = torch.tensor([t.is_control for t in targets], dtype=torch.long, device=device)
    ctx = torch.tensor(np.array([t.context for t in targets]), dtype=torch.long,
                       device=device)

    use_guidance = abs(guidance_scale - 1.0) > 1e-6
    null_cfg = use_guidance and _supports_cfg_null(model)
    scheduler.set_timesteps(n_inference_steps, device=device)
    # One conditioning dict for the whole trajectory
    cond = cond_from_arrays(spec, compound=compound, log10_conc=lx,
                            is_control=ic, context=ctx, device=device)
    if use_guidance and not null_cfg:
        ref_cond = cond_from_arrays(
            spec, compound=torch.zeros_like(compound), log10_conc=lx,
            is_control=torch.ones_like(ic), context=ctx, device=device)
    drop_all = torch.ones(B, dtype=torch.bool, device=device) if null_cfg else None
    for t in scheduler.timesteps:
        model_in = x.to(memory_format=torch.channels_last)
        pred = model(model_in, t, cond, return_dict=False)[0]
        if use_guidance:
            if null_cfg:
                # Same cond dict: `drop` every role-A field (compound, is_control, dose) and leave C/E
                # this a treatment marginal rather than no tx sample.
                pred_ref = model(model_in, t, cond, drop=drop_all,
                                 return_dict=False)[0]
            else:
                pred_ref = model(model_in, t, ref_cond, return_dict=False)[0]
            pred = pred_ref + guidance_scale * (pred - pred_ref)
        x = scheduler.step(pred, t, x).prev_sample
    if latent_ctx is not None:
        # Back to pixels
        return latent_ctx.decode(x)
    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Real-image loader (returns tensors in [-1, 1])
# ---------------------------------------------------------------------------

def _load_real_images(cfg: CaseConfig, indices: np.ndarray, num_workers: int = 4,
                      batch_size: int = 64) -> torch.Tensor:
    ds = RxRx19aDataset(cfg, indices=indices, load_images=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=False)
    out = []
    for batch in loader:
        out.append(batch["image"])
    return torch.cat(out, dim=0)
