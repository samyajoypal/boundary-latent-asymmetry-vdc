from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class AffineGaugeVAE(nn.Module):
    """Freeze a pretrained encoder and expose only a diagonal latent gauge.

    The nonlinear feature map and posterior heads remain fixed. A positive
    coordinate-wise scale and a shift can adapt jointly with the decoder. This
    prevents a mixture prior from inducing an arbitrary nonlinear
    reparameterization while retaining an exact diagonal Gaussian posterior.
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        if not hasattr(base_model, "latent_dim"):
            raise TypeError("base_model must expose latent_dim.")
        self.base_model = base_model
        self.input_dim = int(getattr(base_model, "input_dim"))
        self.latent_dim = int(getattr(base_model, "latent_dim"))
        self.min_logvar = float(getattr(base_model, "min_logvar", -8.0))
        self.max_logvar = float(getattr(base_model, "max_logvar", 6.0))
        self.latent_shift = nn.Parameter(torch.zeros(self.latent_dim))
        self.latent_log_scale = nn.Parameter(torch.zeros(self.latent_dim))

        frozen_modules = []
        for name in ("encoder", "encoder_conv", "fc_mu", "fc_logvar"):
            module = getattr(base_model, name, None)
            if isinstance(module, nn.Module):
                frozen_modules.append(module)
        if not frozen_modules:
            raise TypeError("base_model does not expose a supported encoder.")
        for module in frozen_modules:
            module.requires_grad_(False)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        base_mu, base_logvar = self.base_model.encode(x)
        log_scale = self.latent_log_scale.to(base_mu)
        scale = torch.exp(log_scale)
        mu = base_mu * scale + self.latent_shift.to(base_mu)
        logvar = (base_logvar + 2.0 * log_scale).clamp(
            self.min_logvar, self.max_logvar
        )
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: Tensor) -> Tensor:
        return self.base_model.decode(z)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {"recon": self.decode(z), "z": z, "mu": mu, "logvar": logvar}

    @staticmethod
    def q_log_prob(z: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (z - mu).pow(2) / logvar.exp()
        ).sum(dim=1)
