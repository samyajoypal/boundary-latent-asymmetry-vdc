from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ghmvae.models.vae import MLPVAE
from ghmvae.priors.base import Prior, PriorFitter
from ghmvae.priors.standard_normal import StandardNormalPrior
from ghmvae.training.losses import elbo_loss


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    beta: float = 1.0
    recon_variance: float = 1.0
    device: str = "cpu"
    seed: int = 0


@dataclass
class AlternatingTrainConfig:
    pretrain: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=5))
    finetune: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=5))
    cycles: int = 2


def _loader(x: Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(x.float())
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle)


def train_vae(model: MLPVAE, x: Tensor, prior: Prior, config: TrainConfig) -> list[dict[str, float]]:
    torch.manual_seed(int(config.seed))
    device = torch.device(config.device)
    model.to(device)
    prior.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    history: list[dict[str, float]] = []

    for epoch in range(int(config.epochs)):
        totals: dict[str, float] = {}
        n_batches = 0
        model.train()
        for (batch,) in _loader(x, config.batch_size, shuffle=True):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = elbo_loss(
                model,
                batch,
                prior,
                beta=config.beta,
                recon_variance=config.recon_variance,
            )
            loss.backward()
            optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1
        epoch_metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
        epoch_metrics["epoch"] = float(epoch)
        history.append(epoch_metrics)
    return history


@torch.no_grad()
def encode_dataset(model: MLPVAE, x: Tensor, batch_size: int = 256, device: str = "cpu") -> dict[str, Tensor]:
    model.eval()
    model.to(torch.device(device))
    mus: list[Tensor] = []
    logvars: list[Tensor] = []
    for (batch,) in _loader(x, batch_size, shuffle=False):
        batch = batch.to(torch.device(device))
        mu, logvar = model.encode(batch)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    return {"mu": torch.cat(mus, dim=0), "logvar": torch.cat(logvars, dim=0)}


def alternating_train(
    model: MLPVAE,
    x: Tensor,
    prior_fitter: PriorFitter,
    config: AlternatingTrainConfig,
) -> tuple[Prior, dict[str, object]]:
    standard_prior = StandardNormalPrior(model.latent_dim)
    history: dict[str, object] = {"pretrain": [], "cycles": []}
    history["pretrain"] = train_vae(model, x, standard_prior, config.pretrain)

    fitted_prior: Prior = standard_prior
    for cycle in range(int(config.cycles)):
        encoded = encode_dataset(
            model,
            x,
            batch_size=config.finetune.batch_size,
            device=config.finetune.device,
        )
        fit_result = prior_fitter.fit(encoded["mu"])
        fitted_prior = fit_result.prior
        phase_history = train_vae(model, x, fitted_prior, config.finetune)
        history["cycles"].append(
            {
                "cycle": cycle,
                "fit_diagnostics": fit_result.diagnostics,
                "train": phase_history,
            }
        )
    return fitted_prior, history
