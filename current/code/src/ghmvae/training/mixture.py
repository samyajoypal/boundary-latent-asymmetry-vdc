from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ghmvae.priors.base import Prior, PriorFitter
from ghmvae.training.engine import _loader, encode_dataset
from ghmvae.training.losses import gaussian_reconstruction_log_prob


@dataclass
class MixtureTrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    prior_lr: float | None = None
    beta: float = 1.0
    recon_variance: float = 1.0
    n_mc_samples: int = 1
    responsibility_temperature: float = 1.0
    cluster_mi_weight: float = 0.0
    optimize_prior: bool = False
    grad_clip_norm: float = 10.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    device: str = "cpu"
    seed: int = 0
    objective: Literal["collapsed", "factorized"] = "collapsed"
    assignment_anchor_weight: float = 0.0
    assignment_anchor_confidence_threshold: float = 0.0
    latent_mean_anchor_weight: float = 0.0
    posterior_anchor_weight: float = 0.0
    posterior_anchor_max_validation_kl: float | None = None


@dataclass
class DeterministicPretrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    initial_logvar: float = -4.0


def _log_weights(prior: Prior, z: Tensor) -> Tensor:
    values = getattr(prior, "log_weights", None)
    if values is None:
        raise TypeError(f"Prior {prior.name!r} does not expose mixture log_weights.")
    return values.to(device=z.device, dtype=z.dtype)


def _expected_component_log_prob(
    prior: Prior,
    mu: Tensor,
    logvar: Tensor,
    z_samples: Tensor,
) -> Tensor:
    exact = getattr(prior, "expected_component_log_prob", None)
    if callable(exact):
        return exact(mu, logvar)
    component_log_prob = getattr(prior, "component_log_prob", None)
    if not callable(component_log_prob):
        raise TypeError(f"Prior {prior.name!r} does not expose component_log_prob.")
    n_samples, batch_size, latent_dim = z_samples.shape
    flat = z_samples.reshape(n_samples * batch_size, latent_dim)
    values = component_log_prob(flat)
    return values.reshape(n_samples, batch_size, -1).mean(dim=0)


def diagonal_gaussian_forward_kl(
    reference_mu: Tensor,
    reference_logvar: Tensor,
    candidate_mu: Tensor,
    candidate_logvar: Tensor,
) -> Tensor:
    """Return KL(reference || candidate) for each diagonal Gaussian pair."""

    if not (
        reference_mu.shape
        == reference_logvar.shape
        == candidate_mu.shape
        == candidate_logvar.shape
    ):
        raise ValueError("All posterior tensors must have the same shape.")
    if reference_mu.ndim != 2:
        raise ValueError("Posterior tensors must have shape (n_observations, latent_dim).")
    reference_logvar = reference_logvar.clamp(min=-30.0, max=20.0)
    candidate_logvar = candidate_logvar.clamp(min=-30.0, max=20.0)
    variance_ratio = torch.exp(reference_logvar - candidate_logvar)
    mean_term = (reference_mu - candidate_mu).square() * torch.exp(-candidate_logvar)
    return 0.5 * (
        candidate_logvar
        - reference_logvar
        + variance_ratio
        + mean_term
        - 1.0
    ).sum(dim=1)


def mixture_aware_elbo_loss(
    model: nn.Module,
    x: Tensor,
    prior: Prior,
    beta: float = 1.0,
    recon_variance: float = 1.0,
    n_mc_samples: int = 1,
    responsibility_temperature: float = 1.0,
    cluster_mi_weight: float = 0.0,
    encoded_posterior: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """Factorized q(z|x)q(c|x) objective retained as an ablation."""

    mu, logvar = (
        model.encode(x) if encoded_posterior is None else encoded_posterior
    )
    sample_count = max(int(n_mc_samples), 1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn((sample_count, *mu.shape), device=mu.device, dtype=mu.dtype)
    z_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps

    flat_z = z_samples.reshape(sample_count * x.shape[0], mu.shape[1])
    recon = model.decode(flat_z)
    repeated_x = x.unsqueeze(0).expand(sample_count, *x.shape).reshape(
        sample_count * x.shape[0], *x.shape[1:]
    )
    recon_log_prob = gaussian_reconstruction_log_prob(
        repeated_x,
        recon,
        variance=recon_variance,
    ).reshape(sample_count, x.shape[0]).mean(dim=0)

    expected_component = _expected_component_log_prob(prior, mu, logvar, z_samples)
    log_weights = _log_weights(prior, mu)
    temperature = max(float(responsibility_temperature), 1e-4)
    log_joint = expected_component + log_weights.unsqueeze(0)
    responsibilities = torch.softmax(log_joint / temperature, dim=1)
    log_responsibilities = torch.log(responsibilities.clamp_min(1e-12))

    expected_log_prior = (responsibilities * log_joint).sum(dim=1)
    categorical_entropy = -(responsibilities * log_responsibilities).sum(dim=1)
    gaussian_entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + logvar).sum(dim=1)
    latent_term = expected_log_prior + categorical_entropy + gaussian_entropy
    elbo = recon_log_prob + float(beta) * latent_term

    mean_responsibility = responsibilities.mean(dim=0)
    marginal_entropy = -(mean_responsibility * mean_responsibility.clamp_min(1e-12).log()).sum()
    mean_conditional_entropy = categorical_entropy.mean()
    cluster_mutual_information = marginal_entropy - mean_conditional_entropy
    objective = elbo.mean() + float(cluster_mi_weight) * cluster_mutual_information
    loss = -objective

    metrics = {
        "loss": float(loss.detach().cpu()),
        "elbo": float(elbo.mean().detach().cpu()),
        "recon_log_prob": float(recon_log_prob.mean().detach().cpu()),
        "latent_term": float(latent_term.mean().detach().cpu()),
        "categorical_entropy": float(mean_conditional_entropy.detach().cpu()),
        "marginal_entropy": float(marginal_entropy.detach().cpu()),
        "cluster_mutual_information": float(cluster_mutual_information.detach().cpu()),
        "active_components": float((mean_responsibility > 0.01).sum().detach().cpu()),
    }
    return loss, metrics, responsibilities


def collapsed_mixture_elbo_loss(
    model: nn.Module,
    x: Tensor,
    prior: Prior,
    beta: float = 1.0,
    recon_variance: float = 1.0,
    n_mc_samples: int = 1,
    encoded_posterior: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """Collapsed mixture ELBO using the exact differentiable mixture density."""

    mu, logvar = (
        model.encode(x) if encoded_posterior is None else encoded_posterior
    )
    sample_count = max(int(n_mc_samples), 1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn((sample_count, *mu.shape), device=mu.device, dtype=mu.dtype)
    z_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps

    flat_z = z_samples.reshape(sample_count * x.shape[0], mu.shape[1])
    recon = model.decode(flat_z)
    repeated_x = x.unsqueeze(0).expand(sample_count, *x.shape).reshape(
        sample_count * x.shape[0], *x.shape[1:]
    )
    recon_log_prob = gaussian_reconstruction_log_prob(
        repeated_x,
        recon,
        variance=recon_variance,
    ).reshape(sample_count, x.shape[0]).mean(dim=0)

    prior_log_prob = prior.log_prob(flat_z).reshape(sample_count, x.shape[0]).mean(dim=0)
    gaussian_entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + logvar).sum(dim=1)
    latent_term = prior_log_prob + gaussian_entropy
    elbo = recon_log_prob + float(beta) * latent_term
    loss = -elbo.mean()

    sample_responsibilities = prior.responsibilities(flat_z).reshape(sample_count, x.shape[0], -1)
    responsibilities = sample_responsibilities.mean(dim=0)
    mean_responsibility = responsibilities.mean(dim=0)
    conditional_entropy = -(
        responsibilities * responsibilities.clamp_min(1e-12).log()
    ).sum(dim=1).mean()
    marginal_entropy = -(
        mean_responsibility * mean_responsibility.clamp_min(1e-12).log()
    ).sum()

    metrics = {
        "loss": float(loss.detach().cpu()),
        "elbo": float(elbo.mean().detach().cpu()),
        "recon_log_prob": float(recon_log_prob.mean().detach().cpu()),
        "latent_term": float(latent_term.mean().detach().cpu()),
        "prior_log_prob": float(prior_log_prob.mean().detach().cpu()),
        "categorical_entropy": float(conditional_entropy.detach().cpu()),
        "marginal_entropy": float(marginal_entropy.detach().cpu()),
        "cluster_mutual_information": float((marginal_entropy - conditional_entropy).detach().cpu()),
        "active_components": float((mean_responsibility > 0.01).sum().detach().cpu()),
    }
    return loss, metrics, responsibilities


@torch.no_grad()
def posterior_sample_bank(
    model: nn.Module,
    x: Tensor,
    n_mc_samples: int,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 0,
) -> Tensor:
    """Return fixed common-random-number draws with shape (S, N, D)."""

    encoded = encode_dataset(model, x, batch_size=batch_size, device=device)
    mu = encoded["mu"]
    logvar = encoded["logvar"]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn(
        (max(int(n_mc_samples), 1), *mu.shape),
        generator=generator,
        dtype=mu.dtype,
    )
    return mu.unsqueeze(0) + torch.exp(0.5 * logvar).unsqueeze(0) * eps


@torch.no_grad()
def latent_prior_score(prior: Prior, z_bank: Tensor, device: str = "cpu") -> tuple[float, Tensor]:
    """Score a prior on an unmodified posterior bank, averaging draws by observation."""

    if z_bank.ndim != 3:
        raise ValueError("z_bank must have shape (n_mc_samples, n_observations, latent_dim).")
    sample_count, n_observations, latent_dim = z_bank.shape
    prior.to(device)
    flat = z_bank.reshape(sample_count * n_observations, latent_dim).to(device)
    per_observation = prior.log_prob(flat).reshape(sample_count, n_observations).mean(dim=0).cpu()
    return float(per_observation.mean()), per_observation


def accept_prior_candidate(
    current_prior: Prior | None,
    candidate_prior: Prior,
    z_bank: Tensor,
    candidate_diagnostics: dict[str, Any] | None = None,
    tolerance: float = 1e-6,
    device: str = "cpu",
) -> tuple[Prior, dict[str, Any]]:
    """Accept a fitted candidate only when it preserves the fixed-bank score."""

    candidate_score, _ = latent_prior_score(candidate_prior, z_bank, device=device)
    current_score = -math.inf
    if current_prior is not None:
        current_score, _ = latent_prior_score(current_prior, z_bank, device=device)
    accepted = current_prior is None or candidate_score + float(tolerance) >= current_score
    selected = candidate_prior if accepted else current_prior
    diagnostics = {
        **(candidate_diagnostics or {}),
        "current_prior_score": float(current_score),
        "candidate_prior_score": float(candidate_score),
        "prior_score_change": float(candidate_score - current_score) if math.isfinite(current_score) else math.inf,
        "prior_update_accepted": bool(accepted),
        "prior_acceptance_tolerance": float(tolerance),
    }
    return selected, diagnostics


def accepted_prior_refit(
    current_prior: Prior | None,
    fitter: PriorFitter,
    z_bank: Tensor,
    tolerance: float = 1e-6,
    device: str = "cpu",
) -> tuple[Prior, dict[str, Any]]:
    """Fit on a fixed bank and reject candidates that reduce its latent log score."""

    flat = z_bank.reshape(-1, z_bank.shape[-1])
    fit_result = fitter.fit(flat)
    return accept_prior_candidate(
        current_prior,
        fit_result.prior,
        z_bank,
        candidate_diagnostics=fit_result.diagnostics,
        tolerance=tolerance,
        device=device,
    )


@torch.no_grad()
def evaluate_collapsed_elbo(
    model: nn.Module,
    x: Tensor,
    prior: Prior,
    beta: float = 1.0,
    recon_variance: float = 1.0,
    n_mc_samples: int = 3,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 0,
    latent_mean_anchor: Tensor | None = None,
    latent_mean_anchor_weight: float = 0.0,
    posterior_anchor_logvar: Tensor | None = None,
    posterior_anchor_weight: float = 0.0,
) -> dict[str, float]:
    """Evaluate a fixed-draw collapsed objective for unlabeled validation."""

    if float(latent_mean_anchor_weight) < 0.0:
        raise ValueError("latent_mean_anchor_weight must be nonnegative.")
    if float(posterior_anchor_weight) < 0.0:
        raise ValueError("posterior_anchor_weight must be nonnegative.")
    if latent_mean_anchor is not None:
        if latent_mean_anchor.ndim != 2 or latent_mean_anchor.shape[0] != x.shape[0]:
            raise ValueError(
                "latent_mean_anchor must have shape (n_observations, latent_dim)."
            )
    if posterior_anchor_logvar is not None:
        if latent_mean_anchor is None:
            raise ValueError(
                "posterior_anchor_logvar requires latent_mean_anchor as its reference mean."
            )
        if posterior_anchor_logvar.shape != latent_mean_anchor.shape:
            raise ValueError(
                "posterior_anchor_logvar must have the same shape as latent_mean_anchor."
            )
    if float(posterior_anchor_weight) > 0.0 and posterior_anchor_logvar is None:
        raise ValueError(
            "A positive posterior_anchor_weight requires posterior_anchor_logvar."
        )

    sample_count = max(int(n_mc_samples), 1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    model.eval()
    model.to(device)
    prior.to(device)
    totals = {
        "elbo": 0.0,
        "recon_log_prob": 0.0,
        "prior_log_prob": 0.0,
        "latent_term": 0.0,
        "latent_mean_anchor_squared_error": 0.0,
        "posterior_anchor_kl": 0.0,
    }
    n_seen = 0
    validation_tensors = [x.float()]
    if latent_mean_anchor is not None:
        validation_tensors.append(latent_mean_anchor.detach().float())
    if posterior_anchor_logvar is not None:
        validation_tensors.append(posterior_anchor_logvar.detach().float())
    validation_loader = DataLoader(
        TensorDataset(*validation_tensors),
        batch_size=int(batch_size),
        shuffle=False,
    )
    for payload in validation_loader:
        batch_cpu = payload[0]
        anchor_cpu = payload[1] if len(payload) > 1 else None
        anchor_logvar_cpu = payload[2] if len(payload) > 2 else None
        batch = batch_cpu.to(device)
        mu, logvar = model.encode(batch)
        eps = torch.randn(
            (sample_count, *mu.shape),
            generator=generator,
            dtype=mu.dtype,
        ).to(device)
        z_samples = mu.unsqueeze(0) + torch.exp(0.5 * logvar).unsqueeze(0) * eps
        flat_z = z_samples.reshape(sample_count * batch.shape[0], mu.shape[1])
        repeated_x = batch.unsqueeze(0).expand(sample_count, *batch.shape).reshape(
            sample_count * batch.shape[0], *batch.shape[1:]
        )
        recon = model.decode(flat_z)
        recon_log_prob = gaussian_reconstruction_log_prob(
            repeated_x,
            recon,
            variance=recon_variance,
        ).reshape(sample_count, batch.shape[0]).mean(dim=0)
        prior_log_prob = prior.log_prob(flat_z).reshape(sample_count, batch.shape[0]).mean(dim=0)
        entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + logvar).sum(dim=1)
        latent_term = prior_log_prob + entropy
        elbo = recon_log_prob + float(beta) * latent_term
        values = {
            "elbo": elbo,
            "recon_log_prob": recon_log_prob,
            "prior_log_prob": prior_log_prob,
            "latent_term": latent_term,
        }
        for key, value in values.items():
            totals[key] += float(value.sum().cpu())
        if anchor_cpu is not None:
            anchor = anchor_cpu.to(device=device, dtype=mu.dtype)
            if anchor.shape != mu.shape:
                raise ValueError(
                    "latent_mean_anchor latent dimension does not match model.encode(x)."
                )
            per_observation_squared_error = (mu - anchor).square().mean(dim=1)
            totals["latent_mean_anchor_squared_error"] += float(
                per_observation_squared_error.sum().cpu()
            )
            if anchor_logvar_cpu is not None:
                anchor_logvar = anchor_logvar_cpu.to(
                    device=device, dtype=logvar.dtype
                )
                posterior_kl = diagonal_gaussian_forward_kl(
                    anchor,
                    anchor_logvar,
                    mu,
                    logvar,
                )
                totals["posterior_anchor_kl"] += float(posterior_kl.sum().cpu())
        n_seen += int(batch.shape[0])
    metrics = {key: value / max(n_seen, 1) for key, value in totals.items()}
    metrics["regularized_objective"] = float(metrics["elbo"]) - float(
        latent_mean_anchor_weight
    ) * float(metrics["latent_mean_anchor_squared_error"]) - float(
        posterior_anchor_weight
    ) * float(metrics["posterior_anchor_kl"])
    return metrics


def pretrain_deterministic_latent(
    model: nn.Module,
    x: Tensor,
    config: DeterministicPretrainConfig,
) -> list[dict[str, float]]:
    """Pretrain reconstruction through the posterior mean before mixture fitting."""

    torch.manual_seed(int(config.seed))
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    history: list[dict[str, float]] = []

    for epoch in range(int(config.epochs)):
        total = 0.0
        n_seen = 0
        model.train()
        for (batch,) in _loader(x, config.batch_size, shuffle=True):
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            recon = model.decode(mu)
            loss = F.mse_loss(recon.reshape(batch.shape), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * int(batch.shape[0])
            n_seen += int(batch.shape[0])
        history.append({"epoch": float(epoch), "recon_loss": total / max(n_seen, 1)})

    logvar_layer = getattr(model, "fc_logvar", None)
    if isinstance(logvar_layer, nn.Linear):
        with torch.no_grad():
            logvar_layer.weight.zero_()
            logvar_layer.bias.fill_(float(config.initial_logvar))
    return history


def train_mixture_vae(
    model: nn.Module,
    x: Tensor,
    prior: Prior,
    config: MixtureTrainConfig,
    validation_x: Tensor | None = None,
    validation_mc_samples: int = 3,
    validation_seed: int = 0,
    state_sink: dict[str, Any] | None = None,
    assignment_anchor: Tensor | None = None,
    latent_mean_anchor: Tensor | None = None,
    posterior_anchor_logvar: Tensor | None = None,
    validation_latent_mean_anchor: Tensor | None = None,
    validation_posterior_anchor_logvar: Tensor | None = None,
) -> list[dict[str, float]]:
    torch.manual_seed(int(config.seed))
    if float(config.latent_mean_anchor_weight) < 0.0:
        raise ValueError("latent_mean_anchor_weight must be nonnegative.")
    if float(config.posterior_anchor_weight) < 0.0:
        raise ValueError("posterior_anchor_weight must be nonnegative.")
    if (
        config.posterior_anchor_max_validation_kl is not None
        and float(config.posterior_anchor_max_validation_kl) <= 0.0
    ):
        raise ValueError("posterior_anchor_max_validation_kl must be positive.")
    device = torch.device(config.device)
    model.to(device)
    prior.to(device)

    parameter_groups: list[dict[str, Any]] = [{"params": list(model.parameters()), "lr": float(config.lr)}]
    prior_parameters = [parameter for parameter in prior.parameters() if parameter.requires_grad]
    if config.optimize_prior and prior_parameters:
        parameter_groups.append(
            {
                "params": prior_parameters,
                "lr": float(config.prior_lr if config.prior_lr is not None else config.lr),
            }
        )
    optimizer = torch.optim.Adam(parameter_groups)
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_model_state: dict[str, Tensor] | None = None
    best_prior_state: dict[str, Tensor] | None = None
    stale_epochs = 0

    if validation_x is not None:
        initial_validation = evaluate_collapsed_elbo(
            model,
            validation_x,
            prior,
            beta=config.beta,
            recon_variance=config.recon_variance,
            n_mc_samples=validation_mc_samples,
            batch_size=config.batch_size,
            device=config.device,
            seed=validation_seed,
            latent_mean_anchor=validation_latent_mean_anchor,
            latent_mean_anchor_weight=config.latent_mean_anchor_weight,
            posterior_anchor_logvar=validation_posterior_anchor_logvar,
            posterior_anchor_weight=config.posterior_anchor_weight,
        )
        if (
            config.posterior_anchor_max_validation_kl is not None
            and float(initial_validation["posterior_anchor_kl"])
            > float(config.posterior_anchor_max_validation_kl)
        ):
            raise ValueError(
                "The starting checkpoint is outside the posterior trust region."
            )
        best_loss = -float(initial_validation["regularized_objective"])
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        if config.optimize_prior:
            best_prior_state = {
                key: value.detach().cpu().clone()
                for key, value in prior.state_dict().items()
            }

    training_tensors = [x.float()]
    assignment_anchor_position: int | None = None
    latent_mean_anchor_position: int | None = None
    posterior_anchor_logvar_position: int | None = None
    if assignment_anchor is not None:
        if assignment_anchor.ndim != 2 or assignment_anchor.shape[0] != x.shape[0]:
            raise ValueError(
                "assignment_anchor must have shape (n_observations, n_components)."
            )
        assignment_anchor_position = len(training_tensors)
        training_tensors.append(assignment_anchor.detach().float())
    if latent_mean_anchor is not None:
        if latent_mean_anchor.ndim != 2 or latent_mean_anchor.shape[0] != x.shape[0]:
            raise ValueError(
                "latent_mean_anchor must have shape (n_observations, latent_dim)."
            )
        latent_mean_anchor_position = len(training_tensors)
        training_tensors.append(latent_mean_anchor.detach().float())
    if posterior_anchor_logvar is not None:
        if latent_mean_anchor is None:
            raise ValueError(
                "posterior_anchor_logvar requires latent_mean_anchor as its reference mean."
            )
        if posterior_anchor_logvar.shape != latent_mean_anchor.shape:
            raise ValueError(
                "posterior_anchor_logvar must have the same shape as latent_mean_anchor."
            )
        posterior_anchor_logvar_position = len(training_tensors)
        training_tensors.append(posterior_anchor_logvar.detach().float())
    if float(config.posterior_anchor_weight) > 0.0 and posterior_anchor_logvar is None:
        raise ValueError(
            "A positive posterior_anchor_weight requires posterior_anchor_logvar."
        )
    if validation_latent_mean_anchor is not None:
        if validation_x is None:
            raise ValueError(
                "validation_latent_mean_anchor requires validation_x."
            )
        if (
            validation_latent_mean_anchor.ndim != 2
            or validation_latent_mean_anchor.shape[0] != validation_x.shape[0]
        ):
            raise ValueError(
                "validation_latent_mean_anchor must have shape "
                "(n_validation_observations, latent_dim)."
            )
    if validation_posterior_anchor_logvar is not None:
        if validation_latent_mean_anchor is None:
            raise ValueError(
                "validation_posterior_anchor_logvar requires "
                "validation_latent_mean_anchor."
            )
        if (
            validation_posterior_anchor_logvar.shape
            != validation_latent_mean_anchor.shape
        ):
            raise ValueError(
                "validation_posterior_anchor_logvar must have the same shape as "
                "validation_latent_mean_anchor."
            )
    if (
        float(config.posterior_anchor_weight) > 0.0
        and validation_x is not None
        and validation_posterior_anchor_logvar is None
    ):
        raise ValueError(
            "A positive posterior_anchor_weight requires a validation posterior "
            "anchor whenever validation_x is supplied."
        )
    training_loader = DataLoader(
        TensorDataset(*training_tensors),
        batch_size=int(config.batch_size),
        shuffle=True,
    )

    for epoch in range(int(config.epochs)):
        totals: dict[str, float] = {}
        n_batches = 0
        model.train()
        for payload in training_loader:
            batch = payload[0]
            assignment_anchor_batch = (
                payload[assignment_anchor_position]
                if assignment_anchor_position is not None
                else None
            )
            latent_mean_anchor_batch = (
                payload[latent_mean_anchor_position]
                if latent_mean_anchor_position is not None
                else None
            )
            posterior_anchor_logvar_batch = (
                payload[posterior_anchor_logvar_position]
                if posterior_anchor_logvar_position is not None
                else None
            )
            batch = batch.to(device)
            current_mu, current_logvar = model.encode(batch)
            if config.objective == "collapsed":
                if config.cluster_mi_weight != 0.0 or config.responsibility_temperature != 1.0:
                    raise ValueError(
                        "The primary collapsed objective requires cluster_mi_weight=0 and "
                        "responsibility_temperature=1; use objective='factorized' for those ablations."
                    )
                loss, metrics, responsibilities = collapsed_mixture_elbo_loss(
                    model,
                    batch,
                    prior,
                    beta=config.beta,
                    recon_variance=config.recon_variance,
                    n_mc_samples=config.n_mc_samples,
                    encoded_posterior=(current_mu, current_logvar),
                )
            elif config.objective == "factorized":
                loss, metrics, responsibilities = mixture_aware_elbo_loss(
                    model,
                    batch,
                    prior,
                    beta=config.beta,
                    recon_variance=config.recon_variance,
                    n_mc_samples=config.n_mc_samples,
                    responsibility_temperature=config.responsibility_temperature,
                    cluster_mi_weight=config.cluster_mi_weight,
                    encoded_posterior=(current_mu, current_logvar),
                )
            else:
                raise ValueError(f"Unknown mixture objective {config.objective!r}.")
            base_loss = loss
            if (
                assignment_anchor_batch is not None
                and float(config.assignment_anchor_weight) > 0.0
            ):
                assignment_anchor_batch = assignment_anchor_batch.to(device).clamp_min(1e-12)
                assignment_anchor_batch = assignment_anchor_batch / assignment_anchor_batch.sum(
                    dim=1, keepdim=True
                )
                assignment_kl = (
                    assignment_anchor_batch
                    * (
                        assignment_anchor_batch.log()
                        - responsibilities.clamp_min(1e-12).log()
                    )
                ).sum(dim=1)
                confidence = assignment_anchor_batch.max(dim=1).values
                mask = (
                    confidence
                    >= float(config.assignment_anchor_confidence_threshold)
                ).float()
                anchor_penalty = (assignment_kl * mask).sum() / mask.sum().clamp_min(1.0)
                loss = loss + float(config.assignment_anchor_weight) * anchor_penalty
                metrics["assignment_anchor_kl"] = float(
                    anchor_penalty.detach().cpu()
                )
            if latent_mean_anchor_batch is not None and (
                float(config.latent_mean_anchor_weight) > 0.0
                or float(config.posterior_anchor_weight) > 0.0
            ):
                latent_mean_anchor_batch = latent_mean_anchor_batch.to(
                    device=device, dtype=current_mu.dtype
                )
                if latent_mean_anchor_batch.shape != current_mu.shape:
                    raise ValueError(
                        "latent_mean_anchor latent dimension does not match model.encode(x)."
                    )
                latent_mean_anchor_mse = F.mse_loss(current_mu, latent_mean_anchor_batch)
                metrics["latent_mean_anchor_squared_error"] = float(
                    latent_mean_anchor_mse.detach().cpu()
                )
                if float(config.latent_mean_anchor_weight) > 0.0:
                    latent_mean_anchor_penalty = (
                        float(config.latent_mean_anchor_weight)
                        * latent_mean_anchor_mse
                    )
                    loss = loss + latent_mean_anchor_penalty
                    metrics["latent_mean_anchor_penalty"] = float(
                        latent_mean_anchor_penalty.detach().cpu()
                    )
                if float(config.posterior_anchor_weight) > 0.0:
                    if posterior_anchor_logvar_batch is None:
                        raise RuntimeError("Posterior anchor log-variance batch is missing.")
                    reference_logvar = posterior_anchor_logvar_batch.to(
                        device=device, dtype=current_logvar.dtype
                    )
                    posterior_anchor_kl = diagonal_gaussian_forward_kl(
                        latent_mean_anchor_batch,
                        reference_logvar,
                        current_mu,
                        current_logvar,
                    ).mean()
                    posterior_anchor_penalty = (
                        float(config.posterior_anchor_weight) * posterior_anchor_kl
                    )
                    loss = loss + posterior_anchor_penalty
                    metrics["posterior_anchor_kl"] = float(
                        posterior_anchor_kl.detach().cpu()
                    )
                    metrics["posterior_anchor_penalty"] = float(
                        posterior_anchor_penalty.detach().cpu()
                    )
            metrics["base_loss"] = float(base_loss.detach().cpu())
            metrics["loss"] = float(loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip_norm > 0:
                parameters = list(model.parameters()) + (prior_parameters if config.optimize_prior else [])
                torch.nn.utils.clip_grad_norm_(parameters, float(config.grad_clip_norm))
            optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1
        epoch_metrics = {key: value / max(n_batches, 1) for key, value in totals.items()}
        epoch_metrics["epoch"] = float(epoch)
        history.append(epoch_metrics)
        epoch_loss = float(epoch_metrics.get("loss", math.inf))
        if validation_x is not None:
            validation = evaluate_collapsed_elbo(
                model,
                validation_x,
                prior,
                beta=config.beta,
                recon_variance=config.recon_variance,
                n_mc_samples=validation_mc_samples,
                batch_size=config.batch_size,
                device=config.device,
                seed=validation_seed,
                latent_mean_anchor=validation_latent_mean_anchor,
                latent_mean_anchor_weight=config.latent_mean_anchor_weight,
                posterior_anchor_logvar=validation_posterior_anchor_logvar,
                posterior_anchor_weight=config.posterior_anchor_weight,
            )
            for key, value in validation.items():
                epoch_metrics[f"validation_{key}"] = value
            epoch_loss = -float(validation["regularized_objective"])
            validation_feasible = (
                config.posterior_anchor_max_validation_kl is None
                or float(validation["posterior_anchor_kl"])
                <= float(config.posterior_anchor_max_validation_kl)
            )
            epoch_metrics["validation_posterior_anchor_feasible"] = float(
                validation_feasible
            )
        else:
            validation_feasible = True
        if validation_feasible and epoch_loss < best_loss - float(
            config.early_stopping_min_delta
        ):
            best_loss = epoch_loss
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if config.optimize_prior:
                best_prior_state = {key: value.detach().cpu().clone() for key, value in prior.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if config.early_stopping_patience > 0 and stale_epochs >= int(config.early_stopping_patience):
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_prior_state is not None:
        prior.load_state_dict(best_prior_state)
    if state_sink is not None:
        state_sink.clear()
        state_sink.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "best_monitored_loss": float(best_loss),
                "initial_validation_elbo": (
                    float(initial_validation["elbo"])
                    if validation_x is not None
                    else None
                ),
                "initial_validation_regularized_objective": (
                    float(initial_validation["regularized_objective"])
                    if validation_x is not None
                    else None
                ),
                "epochs_completed": len(history),
                "posterior_anchor_max_validation_kl": (
                    float(config.posterior_anchor_max_validation_kl)
                    if config.posterior_anchor_max_validation_kl is not None
                    else None
                ),
            }
        )
    return history


@torch.no_grad()
def encode_mixture_responsibilities(
    model: nn.Module,
    x: Tensor,
    prior: Prior,
    batch_size: int = 256,
    device: str = "cpu",
    n_mc_samples: int = 32,
    seed: int = 0,
) -> Tensor:
    z_bank = posterior_sample_bank(
        model,
        x,
        n_mc_samples=n_mc_samples,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    prior.to(device)
    sample_count, n_observations, latent_dim = z_bank.shape
    flat = z_bank.reshape(sample_count * n_observations, latent_dim).to(device)
    return prior.responsibilities(flat).reshape(sample_count, n_observations, -1).mean(dim=0).cpu()
