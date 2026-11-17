from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ghmvae.priors.base import Prior
from ghmvae.priors.diag_gmm import DiagonalGaussianMixturePrior


class LearnableDiagonalGaussianMixturePrior(Prior):
    """Jointly optimized diagonal Gaussian mixture used by the VaDE baseline."""

    name = "learnable_diag_gmm"

    def __init__(
        self,
        logits: Tensor,
        means: Tensor,
        log_vars: Tensor,
        min_log_var: float = -8.0,
        max_log_var: float = 6.0,
    ) -> None:
        super().__init__()
        if means.ndim != 2 or log_vars.shape != means.shape:
            raise ValueError("means and log_vars must have shape (n_components, latent_dim).")
        if logits.shape != (means.shape[0],):
            raise ValueError("logits must have shape (n_components,).")
        self.logits = nn.Parameter(logits.detach().float().clone())
        self.means = nn.Parameter(means.detach().float().clone())
        self.log_vars_unconstrained = nn.Parameter(log_vars.detach().float().clone())
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

    @classmethod
    def from_fitted(
        cls,
        prior: DiagonalGaussianMixturePrior,
        min_log_var: float = -8.0,
        max_log_var: float = 6.0,
    ) -> "LearnableDiagonalGaussianMixturePrior":
        return cls(
            logits=prior.log_weights,
            means=prior.means,
            log_vars=prior.log_vars,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
        )

    @property
    def n_components(self) -> int:
        return int(self.means.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def log_weights(self) -> Tensor:
        return torch.log_softmax(self.logits, dim=0)

    @property
    def log_vars(self) -> Tensor:
        return self.log_vars_unconstrained.clamp(self.min_log_var, self.max_log_var)

    def component_log_prob(self, z: Tensor) -> Tensor:
        z_expanded = z.unsqueeze(1)
        means = self.means.unsqueeze(0).to(z)
        log_vars = self.log_vars.unsqueeze(0).to(z)
        return -0.5 * (
            math.log(2.0 * math.pi)
            + log_vars
            + (z_expanded - means).pow(2) / log_vars.exp()
        ).sum(dim=2)

    def expected_component_log_prob(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute E_q[log p(z|c)] exactly for diagonal Gaussian q and p."""

        means = self.means.unsqueeze(0).to(mu)
        component_log_vars = self.log_vars.unsqueeze(0).to(mu)
        posterior_var = logvar.exp().unsqueeze(1)
        squared_offset = (mu.unsqueeze(1) - means).pow(2)
        return -0.5 * (
            math.log(2.0 * math.pi)
            + component_log_vars
            + (posterior_var + squared_offset) / component_log_vars.exp()
        ).sum(dim=2)

    def log_prob(self, z: Tensor) -> Tensor:
        return torch.logsumexp(self.component_log_prob(z) + self.log_weights.to(z), dim=1)

    def responsibilities(self, z: Tensor) -> Tensor:
        return torch.softmax(self.component_log_prob(z) + self.log_weights.to(z), dim=1)

    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        target = torch.device(device) if device is not None else self.means.device
        labels = torch.multinomial(self.log_weights.exp().to(target), int(n), replacement=True)
        means = self.means.to(target)[labels]
        std = torch.exp(0.5 * self.log_vars.to(target)[labels])
        return means + std * torch.randn_like(means)
