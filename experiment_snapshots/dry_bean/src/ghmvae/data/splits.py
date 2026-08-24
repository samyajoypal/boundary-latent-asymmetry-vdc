from __future__ import annotations

import torch
from torch import Tensor


def unlabeled_optimization_selection_split(
    x: Tensor,
    selection_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split observations without consulting labels."""

    if x.ndim < 2:
        raise ValueError("x must have an observation dimension and at least one feature dimension.")
    n_observations = int(x.shape[0])
    if n_observations < 4:
        raise ValueError("At least four observations are required for an unlabeled selection split.")
    fraction = float(selection_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("selection_fraction must lie strictly between 0 and 0.5.")
    n_selection = min(max(int(round(fraction * n_observations)), 2), n_observations - 2)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(n_observations, generator=generator)
    selection_indices = permutation[:n_selection]
    optimization_indices = permutation[n_selection:]
    return (
        x[optimization_indices],
        x[selection_indices],
        optimization_indices,
        selection_indices,
    )
