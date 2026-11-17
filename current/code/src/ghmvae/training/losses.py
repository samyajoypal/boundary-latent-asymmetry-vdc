from __future__ import annotations

import math

import torch
from torch import Tensor

from ghmvae.models.vae import MLPVAE
from ghmvae.priors.base import Prior


def gaussian_reconstruction_log_prob(x: Tensor, recon: Tensor, variance: float = 1.0) -> Tensor:
    x = x.reshape(recon.shape)
    var = torch.as_tensor(float(variance), device=x.device, dtype=x.dtype)
    return -0.5 * (math.log(2.0 * math.pi) + torch.log(var) + (x - recon).pow(2) / var).sum(dim=1)


def elbo_loss(
    model: MLPVAE,
    x: Tensor,
    prior: Prior,
    beta: float = 1.0,
    recon_variance: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    out = model(x)
    recon_log_prob = gaussian_reconstruction_log_prob(x, out["recon"], variance=recon_variance)
    q_log_prob = model.q_log_prob(out["z"], out["mu"], out["logvar"])
    prior_log_prob = prior.log_prob(out["z"])
    kl_estimate = q_log_prob - prior_log_prob
    elbo = recon_log_prob - float(beta) * kl_estimate
    loss = -elbo.mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "elbo": float(elbo.mean().detach().cpu()),
        "recon_log_prob": float(recon_log_prob.mean().detach().cpu()),
        "kl_estimate": float(kl_estimate.mean().detach().cpu()),
        "prior_log_prob": float(prior_log_prob.mean().detach().cpu()),
        "q_log_prob": float(q_log_prob.mean().detach().cpu()),
    }
    return loss, metrics
