from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


class Prior(nn.Module, ABC):
    """Base class for differentiable latent priors used in the VAE ELBO."""

    name: str = "prior"

    @abstractmethod
    def log_prob(self, z: Tensor) -> Tensor:
        """Return log p(z) for a batch of latent points."""

    def responsibilities(self, z: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.name} does not expose responsibilities.")

    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        raise NotImplementedError(f"{self.name} does not implement sampling.")


@dataclass
class PriorFitResult:
    prior: Prior
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PriorFitter(ABC):
    """Estimator that turns encoded latent points into a Prior object."""

    @abstractmethod
    def fit(self, z: Tensor) -> PriorFitResult:
        """Fit a prior to latent points and return the fitted prior."""
