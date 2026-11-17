from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _mlp(sizes: Sequence[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_features, out_features in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation())
    return nn.Sequential(*layers)


class MLPVAE(nn.Module):
    """A compact MLP VAE for tabular/synthetic smoke tests.

    The model keeps the encoder and decoder deliberately ordinary. For image
    experiments we can add a convolutional VAE without changing the prior or
    training interfaces.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        min_logvar: float = -8.0,
        max_logvar: float = 6.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        encoder_sizes = [self.input_dim, *hidden_dims]
        decoder_sizes = [self.latent_dim, *reversed(hidden_dims)]
        self.encoder = _mlp(encoder_sizes)
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.decoder = _mlp(decoder_sizes)
        self.fc_out = nn.Linear(hidden_dims[0], self.input_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.reshape(x.shape[0], self.input_dim)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(self.min_logvar, self.max_logvar)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder(z)
        return self.fc_out(h)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "z": z, "mu": mu, "logvar": logvar}

    @staticmethod
    def q_log_prob(z: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (math.log(2.0 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()).sum(dim=1)
