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
    """
    kw = dict(schedule_kwargs(zero_snr), **extra)
    if kind == "ddpm":
        from diffusers import DDPMScheduler
        return DDPMScheduler(**kw)
    if zero_snr:
        kw.setdefault("timestep_spacing", "trailing")
    if kind == "dpm":
        from diffusers import DPMSolverMultistepScheduler
        return DPMSolverMultistepScheduler(**kw)
    from diffusers import DDIMScheduler
    return DDIMScheduler(**kw)


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
