from __future__ import annotations

import math

import torch
from torch import Tensor

from ghmvae.priors.base import Prior


class StandardNormalPrior(Prior):
    name = "standard_normal"

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)

    def log_prob(self, z: Tensor) -> Tensor:
        return -0.5 * (math.log(2.0 * math.pi) + z.pow(2)).sum(dim=1)

    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        return torch.randn(int(n), self.latent_dim, device=device)
