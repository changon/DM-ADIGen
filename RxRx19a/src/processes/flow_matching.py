"""Rectified-flow / v-prediction flow matching on 2D latents.

`src/processes/ddpm.py` gives relevants paths for noise pred or flow matching.

  * PATH.   DDPM interpolates x_t along a curved VP schedule; 
            FM uses the STRAIGHT line  x_tau = (1 - tau) * noise + tau * x0 ,  tau in [0, 1].
  * TARGET. Both regress a velocity. DDPM's `v = get_velocity(...)` is the VP velocity along the cosine arc; 
            FM's is the constant straight-line velocity  u = x0 - noise  (rectified flow).
  * SAMPLER. DDPM integrates an SDE/DDIM ODE backwards from t=T (pure noise) to t=0. 
            FM integrates the probability-flow ODE FORWARD with deterministic Euler from tau=0 (noise) to tau=1 (data).

"""
from __future__ import annotations

from types import SimpleNamespace

import torch

class FlowMatching:
    """Flow matching for latent DiT arms.
    """

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = int(num_train_timesteps)
        # Set by set_timesteps() for the sampler; unused during training.
        self._dt: float | None = None
        self.timesteps: torch.Tensor | None = None

    # ------------------------------------------------------------------ train
    def sample_tau(self, batch: int, device, generator=None) -> torch.Tensor:
        """tau ~ Uniform[0, 1), one per batch element."""
        return torch.rand(batch, device=device, generator=generator)

    def model_timesteps(self, tau: torch.Tensor) -> torch.Tensor:
        """Index-scale timestep fed to the DiT embedder: t = tau * N (float)."""
        return tau * self.num_train_timesteps

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  tau: torch.Tensor) -> torch.Tensor:
        """straight path: x_tau = (1 - tau) * noise + tau * x0."""
        t = tau.view(-1, *([1] * (x0.ndim - 1)))
        return (1.0 - t) * noise + t * x0

    def velocity_target(self, x0: torch.Tensor,
                        noise: torch.Tensor) -> torch.Tensor:
        """Constant rectified-flow velocity along the path: u = x0 - noise."""
        return x0 - noise

    # Deterministic Euler integration of the flow ODE 
    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        n = int(num_inference_steps)
        self._dt = 1.0 / n
        i = torch.arange(n, device=device, dtype=torch.float32)
        self.timesteps = ((i + 0.5) / n) * self.num_train_timesteps

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor):
        """One forward Euler step: x <- x + dt * v. 
        """
        if self._dt is None:
            raise RuntimeError("call set_timesteps(...) before step(...)")
        return SimpleNamespace(prev_sample=sample + self._dt * model_output)


def make_train_flow_matching(num_train_timesteps: int = 1000) -> FlowMatching:
    return FlowMatching(num_train_timesteps)
