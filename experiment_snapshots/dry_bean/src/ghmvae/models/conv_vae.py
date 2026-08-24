from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ConvVAE(nn.Module):
    """Convolutional VAE for square grayscale images such as digits and MNIST."""

    def __init__(
        self,
        image_shape: Sequence[int],
        latent_dim: int,
        channels: Sequence[int] = (32, 64),
        min_logvar: float = -8.0,
        max_logvar: float = 6.0,
    ) -> None:
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError("image_shape must be (channels, height, width).")
        if len(channels) != 2:
            raise ValueError("ConvVAE currently expects two convolutional channel widths.")
        self.image_shape = tuple(int(value) for value in image_shape)
        self.input_dim = math.prod(self.image_shape)
        self.latent_dim = int(latent_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        in_channels, _, _ = self.image_shape
        c1, c2 = (int(value) for value in channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *self.image_shape)
            conv_shape = tuple(int(value) for value in self.encoder_conv(dummy).shape[1:])
        self.conv_shape = conv_shape
        conv_features = math.prod(conv_shape)
        self.fc_mu = nn.Linear(conv_features, self.latent_dim)
        self.fc_logvar = nn.Linear(conv_features, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim, conv_features)
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c1, in_channels, kernel_size=4, stride=2, padding=1),
        )

        with torch.no_grad():
            decoded = self.decoder_conv(torch.zeros(1, *conv_shape))
        if tuple(decoded.shape[1:]) != self.image_shape:
            raise ValueError(
                f"Image shape {self.image_shape} is not reconstructed exactly by the two-stage ConvVAE."
            )

    def _images(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], *self.image_shape)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder_conv(self._images(x)).flatten(start_dim=1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden).clamp(self.min_logvar, self.max_logvar)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: Tensor) -> Tensor:
        hidden = self.decoder_input(z).reshape(z.shape[0], *self.conv_shape)
        return self.decoder_conv(hidden).reshape(z.shape[0], self.input_dim)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {"recon": self.decode(z), "z": z, "mu": mu, "logvar": logvar}

    @staticmethod
    def q_log_prob(z: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (
            math.log(2.0 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()
        ).sum(dim=1)
