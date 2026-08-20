from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ConvContrastiveClusterer(nn.Module):
    """Convolutional encoder with instance and cluster projection heads."""

    def __init__(
        self,
        image_shape: Sequence[int],
        latent_dim: int,
        n_clusters: int,
        channels: Sequence[int] = (32, 64),
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError("image_shape must be (channels, height, width).")
        if len(channels) != 2:
            raise ValueError("channels must contain exactly two widths.")
        self.image_shape = tuple(int(value) for value in image_shape)
        self.input_dim = math.prod(self.image_shape)
        in_channels = self.image_shape[0]
        c1, c2 = (int(value) for value in channels)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            conv_features = math.prod(
                self.encoder_conv(torch.zeros(1, *self.image_shape)).shape[1:]
            )
        self.encoder_head = nn.Linear(conv_features, int(latent_dim))
        self.instance_head = nn.Sequential(
            nn.Linear(int(latent_dim), int(latent_dim)),
            nn.ReLU(),
            nn.Linear(int(latent_dim), int(projection_dim)),
        )
        self.cluster_head = nn.Linear(int(latent_dim), int(n_clusters))

    def encode(self, x: Tensor) -> Tensor:
        images = x.reshape(x.shape[0], *self.image_shape)
        return self.encoder_head(self.encoder_conv(images).flatten(start_dim=1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.encode(x)
        projection = self.instance_head(features)
        cluster_probabilities = self.cluster_head(features).softmax(dim=1)
        return features, projection, cluster_probabilities

