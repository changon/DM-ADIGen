"""The two generative paradigms, behind one interface.

Neither is a model -- they have no parameters. Each defines a noising path, a regression target and a sampler, and takes a denoiser as an argument, so the same DiT trains under either. 
`arch.json:diffusion_method` records which model config is trianed

    ddpm            variance-preserving cosine schedule, eps or v-prediction
    flow_matching   straight-line rectified flow, velocity target, Euler ODE
"""
from .ddpm import (  # noqa: F401
    make_train_scheduler,
    make_eval_scheduler,
    make_eval_scheduler_for_ckpt,
    training_target,
    schedule_kwargs,
    zero_snr_from_arch,
)
from .flow_matching import FlowMatching, make_train_flow_matching  # noqa: F401