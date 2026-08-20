from __future__ import annotations

import torch
from torch import Tensor

from ghmvae.priors.base import Prior


class AffineTransformedPrior(Prior):
    """Evaluate a prior fitted after coordinate-wise latent standardization."""

    def __init__(self, base_prior: Prior, center: Tensor, scale: Tensor) -> None:
        super().__init__()
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("center and scale must be one-dimensional tensors with equal shape.")
        self.base_prior = base_prior
        self.register_buffer("center", center.detach().float())
        self.register_buffer("scale", scale.detach().float().clamp_min(1e-6))
        self.name = f"affine_{base_prior.name}"

    @property
    def log_weights(self) -> Tensor:
        return getattr(self.base_prior, "log_weights")

    @property
    def n_components(self) -> int:
        return int(getattr(self.base_prior, "n_components"))

    def _standardize(self, z: Tensor) -> Tensor:
        return (z - self.center.to(z)) / self.scale.to(z)

    def _log_abs_det(self, z: Tensor) -> Tensor:
        return self.scale.to(z).log().sum()

    def component_log_prob(self, z: Tensor) -> Tensor:
        component = getattr(self.base_prior, "component_log_prob")
        return component(self._standardize(z)) - self._log_abs_det(z)

    def log_prob(self, z: Tensor) -> Tensor:
        return self.base_prior.log_prob(self._standardize(z)) - self._log_abs_det(z)

    def responsibilities(self, z: Tensor) -> Tensor:
        return self.base_prior.responsibilities(self._standardize(z))

    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        samples = self.base_prior.sample(n, device=device)
        return samples * self.scale.to(samples) + self.center.to(samples)
