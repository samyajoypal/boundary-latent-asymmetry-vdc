from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from ghmvae.priors.gh_parameterization import unit_mean_gig_parameters


def _standardize(x: Tensor) -> Tensor:
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-6)


def _direct_features_from_latent(z: Tensor, input_dim: int, noise: float, generator: torch.Generator) -> Tensor:
    z0 = z[:, 0]
    z1 = z[:, 1]
    features = [
        z0,
        z1,
        torch.sin(z0),
        torch.cos(z1),
        z0 * z1,
        z0.pow(2),
        z1.pow(2),
        torch.tanh(z0 - z1),
        torch.sin(z0 + z1),
        torch.cos(z0 * z1 / 3.0),
    ]
    x = torch.stack(features, dim=1)
    if input_dim > x.shape[1]:
        pad = torch.randn(z.shape[0], input_dim - x.shape[1], generator=generator) * noise
        x = torch.cat([x, pad], dim=1)
    else:
        x = x[:, :input_dim]
    x = x + noise * torch.randn(x.shape, generator=generator)
    return _standardize(x)


def _obfuscated_features_from_latent(
    z: Tensor,
    input_dim: int,
    noise: float,
    nuisance_dims: int,
    generator: torch.Generator,
) -> Tensor:
    signal_dim = max(1, int(input_dim) - max(int(nuisance_dims), 0))
    pool_width = max(8, math.ceil(signal_dim / 4))
    weights = torch.randn(z.shape[1], pool_width, generator=generator)
    weights = weights / weights.norm(dim=0, keepdim=True).clamp_min(1e-6)
    weights = 1.25 * weights
    bias = 2.0 * math.pi * torch.rand(pool_width, generator=generator)
    projected = z @ weights + bias
    projected = projected + 0.08 * torch.randn(projected.shape, generator=generator)
    pool = torch.cat(
        [
            torch.sin(projected),
            torch.cos(0.7 * projected),
            torch.tanh(projected),
            projected.pow(2),
        ],
        dim=1,
    )
    if pool.shape[1] < signal_dim:
        repeats = math.ceil(signal_dim / pool.shape[1])
        pool = pool.repeat(1, repeats)
    x = pool[:, :signal_dim]
    x = x + noise * torch.randn(x.shape, generator=generator)

    if nuisance_dims > 0:
        gaussian = torch.randn(z.shape[0], nuisance_dims, generator=generator)
        gamma = torch.distributions.Gamma(1.5, 0.5).sample((z.shape[0], nuisance_dims))
        heavy_noise = gaussian * torch.sqrt(torch.as_tensor(3.0) / gamma.clamp_min(1e-6))
        nuisance = 0.7 * _standardize(heavy_noise)
        x = torch.cat([x, nuisance], dim=1)

    mixing = torch.randn(x.shape[1], input_dim, generator=generator)
    q, _ = torch.linalg.qr(mixing, mode="reduced")
    if q.shape[1] < input_dim:
        pad = torch.randn(q.shape[0], input_dim - q.shape[1], generator=generator)
        q = torch.cat([q, pad], dim=1)
    x = x @ q[:, :input_dim]
    return _standardize(x)


def _features_from_latent(
    z: Tensor,
    input_dim: int,
    noise: float,
    generator: torch.Generator,
    feature_mode: str = "direct",
    nuisance_dims: int = 0,
) -> Tensor:
    if feature_mode == "direct":
        return _direct_features_from_latent(z, input_dim=input_dim, noise=noise, generator=generator)
    if feature_mode == "obfuscated":
        return _obfuscated_features_from_latent(
            z,
            input_dim=input_dim,
            noise=noise,
            nuisance_dims=nuisance_dims,
            generator=generator,
        )
    raise ValueError("feature_mode must be either 'direct' or 'obfuscated'.")


def make_nonlinear_mixture(
    n_samples: int = 360,
    input_dim: int = 8,
    latent_dim: int = 2,
    n_components: int = 3,
    noise: float = 0.08,
    center_scale: float = 2.5,
    cluster_scale: float = 0.35,
    feature_mode: str = "direct",
    nuisance_dims: int = 0,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate nonlinear features from a Gaussian latent mixture."""

    if latent_dim != 2:
        raise ValueError("The current synthetic generator uses latent_dim=2.")
    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    labels = torch.arange(n_samples) % int(n_components)
    labels = labels[torch.randperm(n_samples, generator=generator)]

    angles = torch.linspace(0, 2 * math.pi, int(n_components) + 1)[:-1]
    centers = torch.stack([center_scale * torch.cos(angles), center_scale * torch.sin(angles)], dim=1)
    z = centers[labels] + cluster_scale * torch.randn(n_samples, latent_dim, generator=generator)
    x = _features_from_latent(
        z,
        input_dim=input_dim,
        noise=noise,
        generator=generator,
        feature_mode=feature_mode,
        nuisance_dims=nuisance_dims,
    )
    return x.float(), labels.long(), z.float()


def make_skewed_heavy_mixture(
    n_samples: int = 360,
    input_dim: int = 8,
    latent_dim: int = 2,
    n_components: int = 3,
    noise: float = 0.08,
    skew_strength: float = 0.75,
    tail_df: float = 3.0,
    contamination: float = 0.03,
    center_scale: float = 2.8,
    cluster_scale: float = 0.28,
    feature_mode: str = "direct",
    nuisance_dims: int = 0,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Synthetic nonlinear data with skewed and heavy-tailed latent clusters."""

    if latent_dim != 2:
        raise ValueError("The current synthetic generator uses latent_dim=2.")
    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    labels = torch.arange(n_samples) % int(n_components)
    labels = labels[torch.randperm(n_samples, generator=generator)]

    angles = torch.linspace(0, 2 * math.pi, int(n_components) + 1)[:-1]
    centers = torch.stack([center_scale * torch.cos(angles), center_scale * torch.sin(angles)], dim=1)
    directions = torch.stack([torch.cos(angles + 0.7), torch.sin(angles + 0.7)], dim=1)
    perp = torch.stack([-directions[:, 1], directions[:, 0]], dim=1)

    gamma = torch.distributions.Gamma(tail_df / 2.0, 0.5).sample((n_samples,))
    tail_scale = torch.sqrt(torch.as_tensor(tail_df) / gamma).unsqueeze(1)
    gaussian = cluster_scale * torch.randn(n_samples, latent_dim, generator=generator) * tail_scale
    expo = torch.distributions.Exponential(1.0).sample((n_samples, 1)) - 1.0
    skew = skew_strength * expo * directions[labels]
    curved = 0.5 * cluster_scale * torch.randn(n_samples, 1, generator=generator) * perp[labels]
    z = centers[labels] + gaussian + skew + curved

    n_out = int(round(float(contamination) * n_samples))
    if n_out > 0:
        out_ids = torch.randperm(n_samples, generator=generator)[:n_out]
        z[out_ids] = z[out_ids] + 2.5 * torch.randn(n_out, latent_dim, generator=generator)

    x = _features_from_latent(
        z,
        input_dim=input_dim,
        noise=noise,
        generator=generator,
        feature_mode=feature_mode,
        nuisance_dims=nuisance_dims,
    )
    return x.float(), labels.long(), z.float()


def make_matched_decoder_mixture(
    n_samples: int = 600,
    input_dim: int = 12,
    latent_dim: int = 2,
    n_components: int = 3,
    family: str = "gaussian",
    noise: float = 0.08,
    center_scale: float = 1.8,
    cluster_scale: float = 0.45,
    skew_strength: float = 1.0,
    tail_df: float = 3.0,
    hidden_width: int = 64,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate data from a fixed neural decoder applied to a mixture latent.

    Unlike the obfuscated robustness scenarios, this construction lies close
    to the assumed decoder class and tests whether mixture-aware training can
    recover the advantage available in a well-specified latent model.
    """

    if latent_dim != 2:
        raise ValueError("The matched simulation currently uses latent_dim=2.")
    valid_families = {"gaussian", "skew", "heavy", "skew_heavy"}
    if family not in valid_families:
        raise ValueError(f"family must be one of {sorted(valid_families)}.")

    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    labels = torch.arange(n_samples) % int(n_components)
    labels = labels[torch.randperm(n_samples, generator=generator)]

    angles = torch.linspace(0, 2 * math.pi, int(n_components) + 1)[:-1]
    centers = torch.stack([center_scale * torch.cos(angles), center_scale * torch.sin(angles)], dim=1)
    directions = torch.stack([torch.cos(angles + 0.55), torch.sin(angles + 0.55)], dim=1)

    gaussian = cluster_scale * torch.randn(n_samples, latent_dim, generator=generator)
    if family in {"heavy", "skew_heavy"}:
        gamma = torch.distributions.Gamma(tail_df / 2.0, tail_df / 2.0).sample((n_samples, 1))
        gaussian = gaussian / torch.sqrt(gamma.clamp_min(1e-6))

    skew = torch.zeros_like(gaussian)
    if family in {"skew", "skew_heavy"}:
        half_normal = torch.randn(n_samples, 1, generator=generator).abs() - math.sqrt(2.0 / math.pi)
        skew = float(skew_strength) * half_normal * directions[labels]
        if family == "skew_heavy":
            skew_scale = torch.distributions.Gamma(tail_df / 2.0, tail_df / 2.0).sample((n_samples, 1))
            skew = skew / torch.sqrt(skew_scale.clamp_min(1e-6))
    z = centers[labels] + gaussian + skew

    w1 = torch.randn(latent_dim, hidden_width, generator=generator) / math.sqrt(latent_dim)
    b1 = 0.2 * torch.randn(hidden_width, generator=generator)
    w2 = torch.randn(hidden_width, hidden_width, generator=generator) / math.sqrt(hidden_width)
    b2 = 0.2 * torch.randn(hidden_width, generator=generator)
    w3 = torch.randn(hidden_width, input_dim, generator=generator) / math.sqrt(hidden_width)
    skip = 0.35 * torch.randn(latent_dim, input_dim, generator=generator)
    hidden1 = torch.tanh(z @ w1 + b1)
    hidden2 = torch.tanh(hidden1 @ w2 + b2)
    x = hidden2 @ w3 + z @ skip
    x = x + float(noise) * torch.randn(x.shape, generator=generator)
    return _standardize(x).float(), labels.long(), z.float()


def make_parametric_mixture(
    n_samples: int = 600,
    input_dim: int = 12,
    latent_dim: int = 2,
    n_components: int = 3,
    family: str = "gaussian",
    noise: float = 0.08,
    center_scale: float = 1.8,
    cluster_scale: float = 0.45,
    skew_strength: float = 1.0,
    tail_df: float = 5.0,
    alpha_bar: float = 1.0,
    gh_lambda: float = 0.7,
    covariance_ratio: float = 1.0,
    class_probs: tuple[float, ...] | None = None,
    hidden_width: int = 64,
    seed: int = 0,
    component_families: tuple[str, ...] | None = None,
    skew_orientation: str = "radial",
) -> tuple[Tensor, Tensor, Tensor, dict[str, object]]:
    """Generate a neural-decoder benchmark from exact candidate-family latents."""

    from scipy.stats import geninvgauss

    valid_families = {
        "gaussian",
        "student_t",
        "skew_normal",
        "skew_t",
        "variance_gamma",
        "hyperbolic",
        "nig",
        "full_gh",
    }
    if family not in valid_families:
        raise ValueError(f"family must be one of {sorted(valid_families)}.")
    if latent_dim != 2:
        raise ValueError("The frozen parametric simulation uses latent_dim=2.")
    if n_components < 2:
        raise ValueError("n_components must be at least two.")
    if covariance_ratio < 1.0:
        raise ValueError("covariance_ratio must be at least one.")
    valid_orientations = {"radial", "clockwise_neighbor", "counterclockwise_neighbor"}
    if skew_orientation not in valid_orientations:
        raise ValueError(f"skew_orientation must be one of {sorted(valid_orientations)}.")
    if component_families is None:
        family_by_component = (family,) * n_components
    else:
        family_by_component = tuple(component_families)
        if len(family_by_component) != n_components:
            raise ValueError("component_families must contain one family per component.")
        unknown_families = sorted(set(family_by_component) - valid_families)
        if unknown_families:
            raise ValueError(f"Unsupported component families: {unknown_families}")

    if class_probs is None:
        probabilities = np.full(n_components, 1.0 / n_components)
    else:
        probabilities = np.asarray(class_probs, dtype=float)
        if probabilities.shape != (n_components,) or np.any(probabilities <= 0.0):
            raise ValueError("class_probs must contain one positive value per component.")
        probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(int(seed))
    labels_np = rng.choice(n_components, size=n_samples, p=probabilities)
    angles = np.linspace(0.0, 2.0 * math.pi, n_components, endpoint=False)
    centers = np.column_stack([center_scale * np.cos(angles), center_scale * np.sin(angles)])
    if skew_orientation == "radial":
        directions = np.column_stack([np.cos(angles + 0.55), np.sin(angles + 0.55)])
    else:
        offset = 1 if skew_orientation == "clockwise_neighbor" else -1
        directions = np.roll(centers, -offset, axis=0) - centers
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    z = np.empty((n_samples, latent_dim), dtype=np.float64)

    if "student_t" in family_by_component and tail_df <= 2.0:
        raise ValueError("The unit-mean Student-t representation requires tail_df > 2.")
    if "skew_t" in family_by_component and tail_df <= 2.0:
        raise ValueError("The unit-mean skew-t representation requires tail_df > 2.")

    component_scales = np.geomspace(
        covariance_ratio ** -0.25,
        covariance_ratio ** 0.25,
        n_components,
    )
    true_components: list[dict[str, object]] = []
    for component in range(n_components):
        component_family = family_by_component[component]
        ids = np.flatnonzero(labels_np == component)
        count = int(ids.size)
        rotation_angle = angles[component] / 2.0 + 0.2
        rotation = np.array(
            [
                [math.cos(rotation_angle), -math.sin(rotation_angle)],
                [math.sin(rotation_angle), math.cos(rotation_angle)],
            ]
        )
        anisotropy = math.sqrt(covariance_ratio)
        eigenvalues = cluster_scale**2 * component_scales[component] ** 2 * np.array(
            [anisotropy, 1.0 / anisotropy]
        )
        sigma = rotation @ np.diag(eigenvalues) @ rotation.T
        gaussian = rng.multivariate_normal(np.zeros(latent_dim), sigma, size=count)
        beta = skew_strength * cluster_scale * directions[component]
        if component_family in {"gaussian", "student_t"}:
            beta = np.zeros(latent_dim)

        lmbda: float | None = None
        chi: float | None = None
        psi: float | None = None
        if component_family == "gaussian":
            w = np.ones(count)
        elif component_family in {"student_t", "skew_t"}:
            lmbda = -tail_df / 2.0
            chi, psi = unit_mean_gig_parameters(lmbda, 0.0)
            w = (tail_df - 2.0) / rng.chisquare(tail_df, size=count)
        elif component_family == "skew_normal":
            centered_half_normal = np.abs(rng.normal(size=count)) - math.sqrt(2.0 / math.pi)
            z[ids] = centers[component] + gaussian + centered_half_normal[:, None] * beta
            true_components.append(
                {
                    "component": component,
                    "family": component_family,
                    "weight": float(probabilities[component]),
                    "mean_center": centers[component].tolist(),
                    "sigma": sigma.tolist(),
                    "skew_vector": beta.tolist(),
                }
            )
            continue
        else:
            if component_family == "variance_gamma":
                lmbda = max(float(gh_lambda), 0.25)
                chi, psi = unit_mean_gig_parameters(lmbda, 0.0)
                w = rng.gamma(shape=lmbda, scale=1.0 / lmbda, size=count)
            else:
                if component_family == "hyperbolic":
                    lmbda = (latent_dim + 1.0) / 2.0
                elif component_family == "nig":
                    lmbda = -0.5
                else:
                    lmbda = float(gh_lambda)
                chi, psi = unit_mean_gig_parameters(lmbda, alpha_bar)
                w = geninvgauss.rvs(
                    lmbda,
                    alpha_bar,
                    scale=math.sqrt(chi / psi),
                    size=count,
                    random_state=rng,
                )

        location = centers[component] - beta
        z[ids] = location + w[:, None] * beta + np.sqrt(w)[:, None] * gaussian
        true_components.append(
            {
                "component": component,
                "family": component_family,
                "weight": float(probabilities[component]),
                "mu": location.tolist(),
                "sigma": sigma.tolist(),
                "beta": beta.tolist(),
                "lambda": lmbda,
                "chi": chi,
                "psi": psi,
            }
        )

    torch.manual_seed(int(seed) + 3001)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 3001)
    z_t = torch.from_numpy(z).float()
    w1 = torch.randn(latent_dim, hidden_width, generator=generator) / math.sqrt(latent_dim)
    b1 = 0.2 * torch.randn(hidden_width, generator=generator)
    w2 = torch.randn(hidden_width, hidden_width, generator=generator) / math.sqrt(hidden_width)
    b2 = 0.2 * torch.randn(hidden_width, generator=generator)
    w3 = torch.randn(hidden_width, input_dim, generator=generator) / math.sqrt(hidden_width)
    skip = 0.35 * torch.randn(latent_dim, input_dim, generator=generator)
    hidden1 = torch.tanh(z_t @ w1 + b1)
    hidden2 = torch.tanh(hidden1 @ w2 + b2)
    x = hidden2 @ w3 + z_t @ skip
    x = x + float(noise) * torch.randn(x.shape, generator=generator)
    truth = {
        "true_family": (
            family_by_component[0]
            if len(set(family_by_component)) == 1
            else "heterogeneous"
        ),
        "component_families": list(family_by_component),
        "skew_orientation": skew_orientation,
        "class_probs": probabilities.tolist(),
        "covariance_ratio": float(covariance_ratio),
        "alpha_bar": float(alpha_bar),
        "gh_lambda": float(gh_lambda),
        "components": true_components,
    }
    return _standardize(x).float(), torch.from_numpy(labels_np).long(), z_t, truth
