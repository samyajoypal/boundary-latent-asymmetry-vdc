from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ghmvae.priors.base import Prior, PriorFitResult, PriorFitter


class DiagonalGaussianMixturePrior(Prior):
    """Differentiable diagonal-covariance Gaussian mixture prior."""

    name = "diag_gmm"

    def __init__(self, log_weights: Tensor, means: Tensor, log_vars: Tensor) -> None:
        super().__init__()
        if means.ndim != 2:
            raise ValueError("means must have shape (n_components, latent_dim).")
        if log_vars.shape != means.shape:
            raise ValueError("log_vars must have the same shape as means.")
        if log_weights.shape != (means.shape[0],):
            raise ValueError("log_weights must have shape (n_components,).")
        self.register_buffer("log_weights", torch.log_softmax(log_weights.detach().float(), dim=0))
        self.register_buffer("means", means.detach().float())
        self.register_buffer("log_vars", log_vars.detach().float())

    @property
    def n_components(self) -> int:
        return int(self.means.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.means.shape[1])

    def component_log_prob(self, z: Tensor) -> Tensor:
        z = z.unsqueeze(1)
        means = self.means.unsqueeze(0).to(device=z.device, dtype=z.dtype)
        log_vars = self.log_vars.unsqueeze(0).to(device=z.device, dtype=z.dtype)
        return -0.5 * (
            math.log(2.0 * math.pi)
            + log_vars
            + (z - means).pow(2) / log_vars.exp()
        ).sum(dim=2)

    def log_prob(self, z: Tensor) -> Tensor:
        log_weights = self.log_weights.to(device=z.device, dtype=z.dtype)
        return torch.logsumexp(self.component_log_prob(z) + log_weights, dim=1)

    def responsibilities(self, z: Tensor) -> Tensor:
        log_weights = self.log_weights.to(device=z.device, dtype=z.dtype)
        return torch.softmax(self.component_log_prob(z) + log_weights, dim=1)

    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        device = torch.device(device) if device is not None else self.means.device
        probs = self.log_weights.exp().to(device=device)
        labels = torch.multinomial(probs, int(n), replacement=True)
        means = self.means.to(device=device)[labels]
        std = torch.exp(0.5 * self.log_vars.to(device=device)[labels])
        return means + torch.randn_like(means) * std


@dataclass
class DiagonalGaussianMixtureFitter(PriorFitter):
    n_components: int
    n_iter: int = 50
    covariance_floor: float = 1e-4
    seed: int = 0
    min_component_weight: float = 1e-4
    init_steps: int = 20

    def fit(self, z: Tensor) -> PriorFitResult:
        with torch.no_grad():
            z_fit = z.detach().float().cpu()
            if z_fit.ndim != 2:
                raise ValueError("z must have shape (n_samples, latent_dim).")
            n_samples, latent_dim = z_fit.shape
            if n_samples < self.n_components:
                raise ValueError("n_samples must be at least n_components.")

            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.seed))
            means = _kmeans_plus_plus(z_fit, self.n_components, generator)
            assignments = torch.zeros(n_samples, dtype=torch.long)
            for _ in range(int(self.init_steps)):
                distances = torch.cdist(z_fit, means).pow(2)
                assignments = distances.argmin(dim=1)
                updated = means.clone()
                for component in range(self.n_components):
                    members = z_fit[assignments == component]
                    if members.numel():
                        updated[component] = members.mean(dim=0)
                if torch.allclose(updated, means, atol=1e-5, rtol=1e-5):
                    means = updated
                    break
                means = updated
            assignments = torch.cdist(z_fit, means).pow(2).argmin(dim=1)
            global_var = z_fit.var(dim=0, unbiased=False).clamp_min(self.covariance_floor)
            vars_ = global_var.repeat(self.n_components, 1).clone()
            weights = torch.full((self.n_components,), 1.0 / self.n_components)
            for component in range(self.n_components):
                members = z_fit[assignments == component]
                if members.shape[0] >= 2:
                    vars_[component] = members.var(dim=0, unbiased=False).clamp_min(self.covariance_floor)
                    weights[component] = float(members.shape[0]) / n_samples
            weights = weights / weights.sum()

            history: list[float] = []
            for _ in range(int(self.n_iter)):
                log_resp_unnorm = _diag_gmm_component_log_prob(z_fit, means, vars_.log()) + weights.log()
                log_norm = torch.logsumexp(log_resp_unnorm, dim=1)
                history.append(float(log_norm.mean().item()))
                resp = torch.softmax(log_resp_unnorm, dim=1)

                nk = resp.sum(dim=0).clamp_min(self.min_component_weight * n_samples)
                weights = nk / nk.sum()
                means = (resp.T @ z_fit) / nk.unsqueeze(1)
                centered = z_fit.unsqueeze(1) - means.unsqueeze(0)
                vars_ = (resp.unsqueeze(2) * centered.pow(2)).sum(dim=0) / nk.unsqueeze(1)
                vars_ = vars_.clamp_min(self.covariance_floor)

                empty = (resp.sum(dim=0) < 1.0).nonzero(as_tuple=False).flatten()
                if empty.numel():
                    replacement_ids = torch.randperm(n_samples, generator=generator)[: empty.numel()]
                    means[empty] = z_fit[replacement_ids]
                    vars_[empty] = global_var

            log_weights = weights.log()
            prior = DiagonalGaussianMixturePrior(log_weights, means, vars_.log())
            diagnostics = {
                "mean_log_likelihood": history,
                "n_components": self.n_components,
                "latent_dim": latent_dim,
            }
            return PriorFitResult(prior=prior, diagnostics=diagnostics)


def _diag_gmm_component_log_prob(z: Tensor, means: Tensor, log_vars: Tensor) -> Tensor:
    return -0.5 * (
        math.log(2.0 * math.pi)
        + log_vars.unsqueeze(0)
        + (z.unsqueeze(1) - means.unsqueeze(0)).pow(2) / log_vars.exp().unsqueeze(0)
    ).sum(dim=2)


def _kmeans_plus_plus(z: Tensor, n_components: int, generator: torch.Generator) -> Tensor:
    n_samples = int(z.shape[0])
    first = int(torch.randint(n_samples, (1,), generator=generator).item())
    centers = [z[first]]
    closest_sq = (z - centers[0]).pow(2).sum(dim=1)
    for _ in range(1, int(n_components)):
        total = closest_sq.sum()
        if float(total) <= 1e-12:
            selected = int(torch.randint(n_samples, (1,), generator=generator).item())
        else:
            probabilities = closest_sq / total
            selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
        centers.append(z[selected])
        candidate_sq = (z - z[selected]).pow(2).sum(dim=1)
        closest_sq = torch.minimum(closest_sq, candidate_sq)
    return torch.stack(centers, dim=0).clone()
