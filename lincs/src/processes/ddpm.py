"""Diff noise schedules"""
from __future__ import annotations

_BASE = dict(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")


def schedule_kwargs(zero_snr: bool) -> dict:
    """Constructor kwargs shared by the train and eval schedulers.
    """
    if not zero_snr:
        return dict(_BASE)
    return dict(_BASE, rescale_betas_zero_snr=True, prediction_type="v_prediction")


def make_train_scheduler(zero_snr: bool):
    from diffusers import DDPMScheduler
    return DDPMScheduler(**schedule_kwargs(zero_snr))


def make_eval_scheduler(zero_snr: bool, kind: str = "ddim", **extra):
    """kind: 'ddim' | 'ddpm' | 'dpm'.

    `clip_sample=False` is not optional on LINCS. DDIMScheduler and
    DDPMScheduler both default to `clip_sample=True`, which clips the predicted
    x0 to [-1, 1] at EVERY step -- an image-range assumption. The outcome here is
    a z-scored gene vector with no such bound, so the default truncates the
    sampled distribution and biases ||tau_hat|| downward (IMPLEMENT.md §2.2: the
    same reason `generation` must not clamp). Measured on the Phase 4 smoke:
    with the default, 0.0% of sampled values fell outside [-1, 1] on both DDPM
    arms, while the flow-matching arm -- whose Euler step does no clipping -- was
    unaffected.

    Training is not affected and is left alone: it only calls `add_noise` and
    `get_velocity`, neither of which reads `clip_sample`.
    """
    kw = dict(schedule_kwargs(zero_snr), **extra)
    if kind == "ddpm":
        from diffusers import DDPMScheduler
        return DDPMScheduler(**dict(kw, clip_sample=False))
    if zero_snr:
        kw.setdefault("timestep_spacing", "trailing")
    if kind == "dpm":
        from diffusers import DPMSolverMultistepScheduler
        # DPMSolver has no `clip_sample`; its analogue is dynamic thresholding,
        # which is off by default. Pin it so a diffusers default cannot turn it on.
        return DPMSolverMultistepScheduler(**dict(kw, thresholding=False))
    from diffusers import DDIMScheduler
    return DDIMScheduler(**dict(kw, clip_sample=False))


def training_target(scheduler, clean, noise, timesteps):
    """The regression target for the denoising loss.
    """
    if scheduler.config.prediction_type == "v_prediction":
        return scheduler.get_velocity(clean, noise, timesteps)
    return noise


def zero_snr_from_arch(arch: dict | None) -> bool:
    """Read the flag off arch.json. Absent => legacy run => False."""
    return bool((arch or {}).get("zero_snr", False))


def make_eval_scheduler_for_ckpt(ckpt_dir: str, kind: str = "ddim", **extra):
    """The scheduler a given checkpoint must be sampled under.

    Reads `arch.json` rather than assuming a schedule

    A flow-matching arm (`arch.json:diffusion_method == "fm"`) returns a `FlowMatching` object instead of a diffusers scheduler, though it exposes the same `set_timesteps` / `timesteps` / `step(...).prev_sample`
    """
    from src.models import read_arch_spec
    arch = read_arch_spec(ckpt_dir)
    if str((arch or {}).get("diffusion_method", "ddpm")) == "fm":
        from src.processes.flow_matching import FlowMatching
        return FlowMatching(int((arch or {}).get("num_train_timesteps", 1000)))
    return make_eval_scheduler(zero_snr_from_arch(arch), kind, **extra)
