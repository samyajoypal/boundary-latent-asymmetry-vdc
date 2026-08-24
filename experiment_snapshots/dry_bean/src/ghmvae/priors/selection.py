from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ghmvae.priors.base import Prior, PriorFitter


ASYMMETRIC_CANDIDATES = frozenset(
    {
        "skew_normal",
        "skew_t",
        "variance_gamma",
        "hyperbolic",
        "nig",
        "full_gh",
    }
)
SYMMETRIC_CANDIDATES = frozenset({"gaussian", "student_t"})
SELECTION_RULES = frozenset(
    {"one_standard_error", "best_score", "shape_gated", "student_t_safe"}
)


@dataclass(frozen=True)
class PriorCandidate:
    name: str
    fitter: PriorFitter
    complexity_rank: int


@dataclass
class PriorSelectionResult:
    prior: Prior
    candidate_name: str
    diagnostics: dict[str, Any]


@torch.no_grad()
def prior_per_observation_score(prior: Prior, z_bank: Tensor, device: str) -> Tensor:
    if z_bank.ndim != 3:
        raise ValueError("z_bank must have shape (n_mc_samples, n_observations, latent_dim).")
    n_samples, n_observations, latent_dim = z_bank.shape
    prior.to(device)
    flat = z_bank.reshape(n_samples * n_observations, latent_dim).to(device)
    return prior.log_prob(flat).reshape(n_samples, n_observations).mean(dim=0).cpu()


@torch.no_grad()
def _mean_abs_marginal_skewness(z_bank: Tensor) -> float:
    flat = z_bank.reshape(-1, z_bank.shape[-1]).float().cpu()
    centered = flat - flat.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    skew = ((centered / scale) ** 3).mean(dim=0).abs()
    return float(skew.mean())


@torch.no_grad()
def prior_assignment_diagnostics(
    prior: Prior,
    z_bank: Tensor,
    device: str,
    component_mass_threshold: float,
) -> dict[str, float | int]:
    """Return label-free mixture-assignment diagnostics on a posterior bank."""

    responsibilities = getattr(prior, "responsibilities", None)
    if not callable(responsibilities):
        raise TypeError(f"Prior {prior.name!r} does not expose responsibilities.")
    n_samples, n_observations, latent_dim = z_bank.shape
    prior.to(device)
    flat = z_bank.reshape(n_samples * n_observations, latent_dim).to(device)
    draw_responsibilities = responsibilities(flat).reshape(
        n_samples, n_observations, -1
    ).detach().cpu()
    if not bool(torch.isfinite(draw_responsibilities).all()):
        raise ValueError("Candidate produced non-finite responsibilities.")
    averaged = draw_responsibilities.mean(dim=0)
    mass = averaged.mean(dim=0)
    conditional_entropy = -(
        averaged * averaged.clamp_min(1e-12).log()
    ).sum(dim=1).mean()
    marginal_entropy = -(mass * mass.clamp_min(1e-12).log()).sum()
    normalizer = math.log(max(int(mass.numel()), 2))
    hard = draw_responsibilities.argmax(dim=2)
    if n_samples <= 1:
        draw_stability = 1.0
    else:
        counts = F.one_hot(hard, num_classes=int(mass.numel())).sum(dim=0)
        draw_stability = float(counts.max(dim=1).values.float().mean() / n_samples)
    return {
        "active_components": int((mass >= float(component_mass_threshold)).sum()),
        "n_components": int(mass.numel()),
        "minimum_component_mass": float(mass.min()),
        "maximum_component_mass": float(mass.max()),
        "normalized_marginal_entropy": float(marginal_entropy / normalizer),
        "normalized_assignment_mi": float(
            (marginal_entropy - conditional_entropy) / normalizer
        ),
        "draw_assignment_stability": draw_stability,
    }


@torch.no_grad()
def prior_assignment_agreement_diagnostics(
    anchor_prior: Prior,
    candidate_prior: Prior,
    z_bank: Tensor,
    device: str,
    core_confidence_threshold: float = 0.8,
) -> dict[str, Any]:
    """Compare candidate assignments with the anchor after label alignment."""

    from scipy.optimize import linear_sum_assignment

    if z_bank.ndim != 3:
        raise ValueError("z_bank must have shape (n_mc_samples, n_observations, latent_dim).")
    sample_count, n_observations, latent_dim = z_bank.shape
    flat = z_bank.reshape(sample_count * n_observations, latent_dim).to(device)
    anchor_prior.to(device)
    candidate_prior.to(device)
    anchor = anchor_prior.responsibilities(flat).reshape(
        sample_count, n_observations, -1
    ).mean(dim=0).cpu()
    candidate = candidate_prior.responsibilities(flat).reshape(
        sample_count, n_observations, -1
    ).mean(dim=0).cpu()
    if anchor.shape[1] != candidate.shape[1]:
        raise ValueError("Anchor and candidate must have the same number of components.")
    if not bool(torch.isfinite(anchor).all() and torch.isfinite(candidate).all()):
        raise ValueError("Anchor or candidate produced non-finite responsibilities.")

    overlap = anchor.T @ candidate
    anchor_rows, candidate_columns = linear_sum_assignment(-overlap.numpy())
    aligned_columns = torch.empty(anchor.shape[1], dtype=torch.long)
    aligned_columns[torch.as_tensor(anchor_rows)] = torch.as_tensor(candidate_columns)
    aligned = candidate[:, aligned_columns]

    anchor_confidence, anchor_hard = anchor.max(dim=1)
    candidate_hard = aligned.argmax(dim=1)
    core = anchor_confidence >= float(core_confidence_threshold)
    agreement = candidate_hard == anchor_hard
    core_fraction = float(core.float().mean())
    core_retention = float(agreement[core].float().mean()) if bool(core.any()) else math.nan
    total_variation = 0.5 * (anchor - aligned).abs().sum(dim=1)
    return {
        "core_confidence_threshold": float(core_confidence_threshold),
        "core_fraction": core_fraction,
        "core_retention": core_retention,
        "hard_assignment_agreement": float(agreement.float().mean()),
        "mean_assignment_total_variation": float(total_variation.mean()),
        "core_mean_assignment_total_variation": (
            float(total_variation[core].mean()) if bool(core.any()) else math.nan
        ),
        "alignment_candidate_columns": aligned_columns.tolist(),
    }


@torch.no_grad()
def prior_boundary_assignment_diagnostics(
    anchor_prior: Prior,
    candidate_prior: Prior,
    z_bank: Tensor,
    device: str,
    boundary_fraction: float = 0.2,
) -> dict[str, Any]:
    """Measure whether assignment changes concentrate near anchor boundaries.

    Components are first aligned by maximum responsibility overlap. The anchor
    boundary is then defined prospectively as the observations in the lowest
    ``boundary_fraction`` of top-two posterior log-odds margins.
    """

    from scipy.optimize import linear_sum_assignment

    if z_bank.ndim != 3:
        raise ValueError("z_bank must have shape (n_mc_samples, n_observations, latent_dim).")
    if not 0.0 < float(boundary_fraction) < 1.0:
        raise ValueError("boundary_fraction must lie strictly between zero and one.")
    sample_count, n_observations, latent_dim = z_bank.shape
    if n_observations < 2:
        raise ValueError("At least two observations are required for boundary diagnostics.")
    flat = z_bank.reshape(sample_count * n_observations, latent_dim).to(device)
    anchor_prior.to(device)
    candidate_prior.to(device)
    anchor = anchor_prior.responsibilities(flat).reshape(
        sample_count, n_observations, -1
    ).mean(dim=0).cpu()
    candidate = candidate_prior.responsibilities(flat).reshape(
        sample_count, n_observations, -1
    ).mean(dim=0).cpu()
    if anchor.shape[1] != candidate.shape[1]:
        raise ValueError("Anchor and candidate must have the same number of components.")
    if anchor.shape[1] < 2:
        raise ValueError("Boundary diagnostics require at least two components.")
    if not bool(torch.isfinite(anchor).all() and torch.isfinite(candidate).all()):
        raise ValueError("Anchor or candidate produced non-finite responsibilities.")

    overlap = anchor.T @ candidate
    anchor_rows, candidate_columns = linear_sum_assignment(-overlap.numpy())
    aligned_columns = torch.empty(anchor.shape[1], dtype=torch.long)
    aligned_columns[torch.as_tensor(anchor_rows)] = torch.as_tensor(candidate_columns)
    aligned = candidate[:, aligned_columns]

    top_two = torch.topk(anchor, k=2, dim=1).values.clamp_min(1e-12)
    anchor_margin = (top_two[:, 0].log() - top_two[:, 1].log()).cpu()
    anchor_hard = anchor.argmax(dim=1)
    candidate_hard = aligned.argmax(dim=1)
    changed = candidate_hard != anchor_hard

    boundary_count = max(
        1,
        min(
            n_observations - 1,
            int(math.ceil(float(boundary_fraction) * n_observations)),
        ),
    )
    boundary = torch.zeros(n_observations, dtype=torch.bool)
    boundary[torch.argsort(anchor_margin)[:boundary_count]] = True
    interior = ~boundary
    overall_rate = float(changed.float().mean())
    boundary_rate = float(changed[boundary].float().mean())
    interior_rate = float(changed[interior].float().mean())
    disagreement_count = int(changed.sum())
    disagreement_share = (
        float(changed[boundary].sum() / disagreement_count)
        if disagreement_count > 0
        else math.nan
    )
    concentration = (
        boundary_rate / overall_rate if overall_rate > 0.0 else math.nan
    )
    ordered = torch.argsort(anchor_margin)
    margin_bins: list[dict[str, float | int]] = []
    for bin_index, indices in enumerate(torch.tensor_split(ordered, 10)):
        if indices.numel() == 0:
            continue
        margin_bins.append(
            {
                "margin_decile": int(bin_index + 1),
                "n_observations": int(indices.numel()),
                "anchor_margin_mean": float(anchor_margin[indices].mean()),
                "anchor_margin_minimum": float(anchor_margin[indices].min()),
                "anchor_margin_maximum": float(anchor_margin[indices].max()),
                "decision_change_rate": float(changed[indices].float().mean()),
            }
        )

    return {
        "status": "ok",
        "boundary_fraction": float(boundary_count / n_observations),
        "boundary_count": int(boundary_count),
        "n_observations": int(n_observations),
        "decision_change_count": disagreement_count,
        "decision_change_fraction": overall_rate,
        "boundary_decision_change_rate": boundary_rate,
        "interior_decision_change_rate": interior_rate,
        "boundary_disagreement_share": disagreement_share,
        "boundary_disagreement_concentration": concentration,
        "anchor_margin_boundary_max": float(anchor_margin[boundary].max()),
        "anchor_margin_interior_min": float(anchor_margin[interior].min()),
        "anchor_margin_changed_mean": (
            float(anchor_margin[changed].mean()) if disagreement_count > 0 else math.nan
        ),
        "anchor_margin_unchanged_mean": (
            float(anchor_margin[~changed].mean())
            if disagreement_count < n_observations
            else math.nan
        ),
        "margin_deciles": margin_bins,
        "alignment_candidate_columns": aligned_columns.tolist(),
    }


@torch.no_grad()
def prior_boundary_correction_diagnostics(
    anchor_prior: Prior,
    candidate_prior: Prior,
    z_bank: Tensor,
    component: int,
    device: str,
    boundary_log_odds_threshold: float = 2.0,
) -> dict[str, float | int]:
    """Locate a component enrichment relative to the anchor decision boundary.

    The statistic is label-free.  For component ``component``, it compares the
    anchor component log joint with its strongest rival and measures how much of
    the candidate log-density correction acts where the anchor is uncertain.
    Monte Carlo draws are averaged within observations before standard errors
    are computed, so posterior draws are not treated as independent data.
    """

    if z_bank.ndim != 3:
        raise ValueError("z_bank must have shape (n_mc_samples, n_observations, latent_dim).")
    if float(boundary_log_odds_threshold) <= 0.0:
        raise ValueError("boundary_log_odds_threshold must be positive.")
    sample_count, n_observations, latent_dim = z_bank.shape
    if n_observations < 2:
        raise ValueError("At least two observations are required for a standard error.")
    flat = z_bank.reshape(sample_count * n_observations, latent_dim).to(device)
    anchor_prior.to(device)
    candidate_prior.to(device)
    anchor_component = anchor_prior.component_log_prob(flat)
    candidate_component = candidate_prior.component_log_prob(flat)
    if anchor_component.shape != candidate_component.shape:
        raise ValueError("Anchor and candidate component scores must have equal shape.")
    n_components = int(anchor_component.shape[1])
    if n_components < 2:
        raise ValueError("Boundary diagnostics require at least two components.")
    if not 0 <= int(component) < n_components:
        raise ValueError(f"component must lie in [0, {n_components - 1}].")

    anchor_log_weights = getattr(anchor_prior, "log_weights", None)
    candidate_log_weights = getattr(candidate_prior, "log_weights", None)
    if not isinstance(anchor_log_weights, Tensor) or not isinstance(
        candidate_log_weights, Tensor
    ):
        raise TypeError("Anchor and candidate must expose tensor log_weights.")
    anchor_joint = (
        anchor_component + anchor_log_weights.to(anchor_component)
    ).reshape(sample_count, n_observations, n_components)
    candidate_joint = (
        candidate_component + candidate_log_weights.to(candidate_component)
    ).reshape(sample_count, n_observations, n_components)
    if not bool(torch.isfinite(anchor_joint).all() and torch.isfinite(candidate_joint).all()):
        raise ValueError("Anchor or candidate produced non-finite component scores.")

    rival_indices = [index for index in range(n_components) if index != int(component)]
    anchor_target = anchor_joint[:, :, int(component)]
    anchor_rival = anchor_joint[:, :, rival_indices].max(dim=2).values
    contrast = anchor_target - anchor_rival
    correction = candidate_joint[:, :, int(component)] - anchor_target
    candidate_contrast = contrast + correction
    anchor_responsibility = torch.softmax(anchor_joint, dim=2)[:, :, int(component)]
    candidate_responsibility = torch.softmax(candidate_joint, dim=2)[:, :, int(component)]

    boundary_weight_draw = 4.0 * anchor_responsibility * (1.0 - anchor_responsibility)
    boundary_weight = boundary_weight_draw.mean(dim=0).cpu()
    mixture_difference_draw = (
        torch.logsumexp(candidate_joint, dim=2)
        - torch.logsumexp(anchor_joint, dim=2)
    )
    weighted_difference = (
        boundary_weight_draw * mixture_difference_draw
    ).mean(dim=0).cpu()
    mean_weight = float(boundary_weight.mean())
    if mean_weight <= 1e-12:
        boundary_score_gain = math.nan
        boundary_score_standard_error = math.nan
        boundary_effective_size = 0.0
    else:
        estimate = weighted_difference.mean() / mean_weight
        influence = (weighted_difference - estimate * boundary_weight) / mean_weight
        boundary_score_gain = float(estimate)
        boundary_score_standard_error = float(
            influence.std(unbiased=True) / math.sqrt(n_observations)
        )
        boundary_effective_size = float(
            boundary_weight.sum().square()
            / boundary_weight.square().sum().clamp_min(1e-12)
        )

    absolute_correction = correction.abs()
    leverage = absolute_correction / (
        contrast.abs() + absolute_correction + 1e-8
    )
    boundary_weight_total = boundary_weight_draw.sum().clamp_min(1e-12)
    boundary_leverage = float(
        (boundary_weight_draw * leverage).sum() / boundary_weight_total
    )
    responsibility_shift = (candidate_responsibility - anchor_responsibility).abs()
    boundary_responsibility_shift = float(
        (boundary_weight_draw * responsibility_shift).sum() / boundary_weight_total
    )
    near_boundary = contrast.abs() <= float(boundary_log_odds_threshold)
    correction_total = absolute_correction.sum().clamp_min(1e-12)
    decision_change = (contrast >= 0.0) != (candidate_contrast >= 0.0)

    return {
        "component": int(component),
        "boundary_log_odds_threshold": float(boundary_log_odds_threshold),
        "boundary_weight_mean": mean_weight,
        "boundary_effective_sample_size": boundary_effective_size,
        "boundary_score_gain": boundary_score_gain,
        "boundary_score_standard_error": boundary_score_standard_error,
        "boundary_leverage": boundary_leverage,
        "boundary_responsibility_shift": boundary_responsibility_shift,
        "near_boundary_fraction": float(near_boundary.float().mean()),
        "near_boundary_correction_share": float(
            absolute_correction[near_boundary].sum() / correction_total
        ),
        "decision_change_fraction": float(decision_change.float().mean()),
        "mean_absolute_log_density_correction": float(absolute_correction.mean()),
    }


@torch.no_grad()
def prior_component_shape_diagnostics(
    anchor_prior: Prior,
    z_bank: Tensor,
    device: str,
    component_mass_threshold: float,
) -> dict[str, Any]:
    """Measure rotation-invariant residual skewness inside anchor components."""

    responsibilities = getattr(anchor_prior, "responsibilities", None)
    if not callable(responsibilities):
        raise TypeError(f"Prior {anchor_prior.name!r} does not expose responsibilities.")
    z = z_bank.mean(dim=0).float().cpu()
    n_samples, n_observations, latent_dim = z_bank.shape
    anchor_prior.to(device)
    flat = z_bank.reshape(n_samples * n_observations, latent_dim).to(device)
    weights = responsibilities(flat).reshape(n_samples, n_observations, -1).mean(dim=0).cpu()
    component_mass = weights.mean(dim=0)
    component_scores: list[float] = []
    component_weights: list[float] = []
    component_records: list[dict[str, float | int]] = []
    for component in range(weights.shape[1]):
        if float(component_mass[component]) < float(component_mass_threshold):
            continue
        weight = weights[:, component]
        total = weight.sum().clamp_min(1e-8)
        center = (weight[:, None] * z).sum(dim=0) / total
        residual = z - center
        covariance = residual.T @ (weight[:, None] * residual) / total
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        inverse_sqrt = eigenvectors @ torch.diag(
            eigenvalues.clamp_min(1e-6).rsqrt()
        ) @ eigenvectors.T
        whitened = residual @ inverse_sqrt
        variance = (weight[:, None] * residual.square()).sum(dim=0) / total
        third = (weight[:, None] * residual.pow(3)).sum(dim=0) / total
        coordinate_skew = third / variance.clamp_min(1e-8).pow(1.5)

        split_scores: list[float] = []
        for fit_parity, score_parity in ((0, 1), (1, 0)):
            fit_mask = torch.arange(z.shape[0]) % 2 == fit_parity
            score_mask = ~fit_mask
            fit_weight = weight[fit_mask]
            score_weight = weight[score_mask]
            if float(fit_weight.sum()) <= 1e-6 or float(score_weight.sum()) <= 1e-6:
                continue
            fit_whitened = whitened[fit_mask]
            direction = (
                fit_weight[:, None]
                * fit_whitened
                * fit_whitened.square().sum(dim=1, keepdim=True)
            ).sum(dim=0) / fit_weight.sum()
            direction_norm = torch.linalg.vector_norm(direction)
            if float(direction_norm) <= 1e-8:
                split_scores.append(0.0)
                continue
            projection = whitened[score_mask] @ (direction / direction_norm)
            projection_center = (
                score_weight * projection
            ).sum() / score_weight.sum()
            projection = projection - projection_center
            projection_variance = (
                score_weight * projection.square()
            ).sum() / score_weight.sum()
            projection_third = (
                score_weight * projection.pow(3)
            ).sum() / score_weight.sum()
            split_scores.append(
                float(
                    projection_third
                    / projection_variance.clamp_min(1e-8).pow(1.5)
                )
            )
        signed_invariant_skew = float(
            sum(split_scores) / max(len(split_scores), 1)
        )
        invariant_skew = abs(signed_invariant_skew)
        effective_n = float(total.square() / weight.square().sum().clamp_min(1e-8))
        normal_reference_se = math.sqrt(6.0 / max(effective_n, 1e-12))
        component_scores.append(invariant_skew)
        component_weights.append(float(component_mass[component]))
        component_records.append(
            {
                "component": int(component),
                "mass": float(component_mass[component]),
                "effective_sample_size": effective_n,
                "coordinate_abs_skew_mean": float(coordinate_skew.abs().mean()),
                "crossfit_directional_skew": signed_invariant_skew,
                "crossfit_directional_abs_skew": invariant_skew,
                "normal_reference_skew_standard_error": normal_reference_se,
                "normal_reference_skew_signal": (
                    invariant_skew / max(normal_reference_se, 1e-12)
                ),
            }
        )
    if not component_scores:
        return {
            "anchor_component_abs_skew_mean": math.nan,
            "anchor_component_abs_skew_max": math.nan,
            "components": [],
        }
    total_weight = sum(component_weights)
    weighted_mean = sum(
        score * weight for score, weight in zip(component_scores, component_weights)
    ) / max(total_weight, 1e-12)
    return {
        "anchor_component_abs_skew_mean": float(weighted_mean),
        "anchor_component_abs_skew_max": float(max(component_scores)),
        "components": component_records,
    }


def prior_regularity_diagnostics(
    prior: Prior,
    *,
    minimum_weight: float,
    minimum_covariance_eigenvalue: float,
    maximum_covariance_condition: float,
    minimum_interior_shape: float,
) -> dict[str, Any]:
    """Check the compact regular parameter region used by the theory."""

    base_prior = getattr(prior, "base_prior", prior)
    violations: list[str] = []
    log_weights = getattr(base_prior, "log_weights", None)
    if isinstance(log_weights, Tensor):
        weights = log_weights.detach().float().softmax(dim=0).cpu()
        if not bool(torch.isfinite(weights).all()):
            violations.append("nonfinite_mixture_weight")
        elif float(weights.min()) < float(minimum_weight):
            violations.append(
                f"minimum_weight={float(weights.min()):.6g}<{float(minimum_weight):.6g}"
            )
    components = getattr(base_prior, "components", None)
    if components is None:
        return {"regular": not violations, "violations": violations}
    minimum_eigenvalue = math.inf
    maximum_condition = 0.0
    minimum_positive_shape = math.inf
    interior_families = frozenset({"mghp", "mvhb", "mnig"})
    for index, component in enumerate(components):
        sigma = getattr(component, "sigma", None)
        if isinstance(sigma, Tensor):
            eigenvalues = torch.linalg.eigvalsh(sigma.detach().float().cpu())
            if not bool(torch.isfinite(eigenvalues).all()):
                violations.append(f"component_{index}_nonfinite_covariance")
            else:
                smallest = float(eigenvalues.min())
                largest = float(eigenvalues.max())
                condition = largest / max(smallest, 1e-30)
                minimum_eigenvalue = min(minimum_eigenvalue, smallest)
                maximum_condition = max(maximum_condition, condition)
                if smallest < float(minimum_covariance_eigenvalue):
                    violations.append(
                        f"component_{index}_minimum_eigenvalue={smallest:.6g}"
                    )
                if condition > float(maximum_covariance_condition):
                    violations.append(f"component_{index}_condition={condition:.6g}")
        family = str(getattr(component, "family", ""))
        if family in interior_families:
            chi = float(getattr(component, "chi", math.nan))
            psi = float(getattr(component, "psi", math.nan))
            if not math.isfinite(chi) or chi < float(minimum_interior_shape):
                violations.append(f"component_{index}_chi={chi:.6g}")
            if not math.isfinite(psi) or psi < float(minimum_interior_shape):
                violations.append(f"component_{index}_psi={psi:.6g}")
            if math.isfinite(chi) and chi > 0.0:
                minimum_positive_shape = min(minimum_positive_shape, chi)
            if math.isfinite(psi) and psi > 0.0:
                minimum_positive_shape = min(minimum_positive_shape, psi)
    return {
        "regular": not violations,
        "violations": violations,
        "minimum_covariance_eigenvalue": (
            minimum_eigenvalue if math.isfinite(minimum_eigenvalue) else None
        ),
        "maximum_covariance_condition": maximum_condition,
        "minimum_positive_interior_shape": (
            minimum_positive_shape if math.isfinite(minimum_positive_shape) else None
        ),
    }


def _held_out_mean(item: tuple[PriorCandidate, Prior, dict[str, Any], Tensor]) -> float:
    return float(item[2]["selection_mean_log_score"])


def _one_se_selection(
    successful: Sequence[tuple[PriorCandidate, Prior, dict[str, Any], Tensor]],
    threshold: float,
) -> tuple[PriorCandidate, Prior, dict[str, Any], Tensor]:
    eligible = [
        item
        for item in successful
        if float(item[2]["selection_mean_log_score"]) >= threshold
    ]
    if not eligible:
        return max(successful, key=_held_out_mean)
    return min(
        eligible,
        key=lambda item: (
            item[0].complexity_rank,
            -float(item[2]["selection_mean_log_score"]),
            item[0].name,
        ),
    )


def fit_and_select_prior(
    candidates: Sequence[PriorCandidate],
    optimization_bank: Tensor,
    selection_bank: Tensor,
    device: str = "cpu",
    selection_rule: str = "one_standard_error",
    shape_skew_threshold: float = 0.25,
    shape_score_margin: float = 0.01,
    safe_confidence_z: float = 0.0,
    safe_familywise_alpha: float = 0.05,
    safe_min_active_fraction: float = 0.8,
    safe_assignment_tolerance: float = 0.05,
    safe_component_mass_threshold: float = 0.01,
    safe_minimum_weight: float = 1e-4,
    safe_minimum_covariance_eigenvalue: float = 1e-5,
    safe_maximum_covariance_condition: float = 1e6,
    safe_minimum_interior_shape: float = 1e-4,
    safe_core_confidence_threshold: float = 0.8,
    safe_minimum_core_fraction: float = 0.05,
    safe_minimum_core_retention: float = 0.95,
) -> PriorSelectionResult:
    """Fit candidates and select by held-out latent log score."""

    if not candidates:
        raise ValueError("At least one prior candidate is required.")
    if selection_rule not in SELECTION_RULES:
        raise ValueError(f"selection_rule must be one of {sorted(SELECTION_RULES)}.")
    if optimization_bank.ndim != 3 or selection_bank.ndim != 3:
        raise ValueError("Optimization and selection banks must both have shape (S, N, D).")
    if optimization_bank.shape[-1] != selection_bank.shape[-1]:
        raise ValueError("Optimization and selection banks must have the same latent dimension.")
    if selection_bank.shape[1] < 2:
        raise ValueError("The selection bank needs at least two observations for a standard error.")
    if selection_rule == "student_t_safe":
        if not 0.0 < float(safe_familywise_alpha) < 1.0:
            raise ValueError("safe_familywise_alpha must lie strictly between zero and one.")
        if not 0.0 < float(safe_min_active_fraction) <= 1.0:
            raise ValueError("safe_min_active_fraction must lie in (0, 1].")
        if float(safe_assignment_tolerance) < 0.0:
            raise ValueError("safe_assignment_tolerance must be nonnegative.")
        if not 0.0 <= float(safe_component_mass_threshold) <= 1.0:
            raise ValueError("safe_component_mass_threshold must lie in [0, 1].")
        if float(safe_minimum_weight) < 0.0:
            raise ValueError("safe_minimum_weight must be nonnegative.")
        if float(safe_minimum_covariance_eigenvalue) <= 0.0:
            raise ValueError("safe_minimum_covariance_eigenvalue must be positive.")
        if float(safe_maximum_covariance_condition) < 1.0:
            raise ValueError("safe_maximum_covariance_condition must be at least one.")
        if float(safe_minimum_interior_shape) <= 0.0:
            raise ValueError("safe_minimum_interior_shape must be positive.")
        if not 0.0 <= float(safe_core_confidence_threshold) <= 1.0:
            raise ValueError("safe_core_confidence_threshold must lie in [0, 1].")
        if not 0.0 <= float(safe_minimum_core_fraction) <= 1.0:
            raise ValueError("safe_minimum_core_fraction must lie in [0, 1].")
        if not 0.0 <= float(safe_minimum_core_retention) <= 1.0:
            raise ValueError("safe_minimum_core_retention must lie in [0, 1].")

    fit_points = optimization_bank.reshape(-1, optimization_bank.shape[-1])
    successful: list[tuple[PriorCandidate, Prior, dict[str, Any], Tensor]] = []
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_start = time.perf_counter()
        try:
            fit_result = candidate.fitter.fit(fit_points)
            scores = prior_per_observation_score(
                fit_result.prior, selection_bank, device=device
            )
            if scores.numel() == 0 or not bool(torch.isfinite(scores).all()):
                raise ValueError("Candidate produced non-finite held-out log scores.")
            mean = float(scores.mean())
            standard_error = float(scores.std(unbiased=True) / math.sqrt(scores.numel()))
            if not math.isfinite(mean) or not math.isfinite(standard_error):
                raise ValueError(
                    "Candidate produced a non-finite held-out mean or standard error."
                )
            record = {
                "candidate": candidate.name,
                "complexity_rank": int(candidate.complexity_rank),
                "status": "ok",
                "selection_mean_log_score": mean,
                "selection_standard_error": standard_error,
                "fit_and_score_seconds": float(time.perf_counter() - candidate_start),
                "fit_diagnostics": fit_result.diagnostics,
            }
            if selection_rule == "student_t_safe":
                record["regularity"] = prior_regularity_diagnostics(
                    fit_result.prior,
                    minimum_weight=safe_minimum_weight,
                    minimum_covariance_eigenvalue=safe_minimum_covariance_eigenvalue,
                    maximum_covariance_condition=safe_maximum_covariance_condition,
                    minimum_interior_shape=safe_minimum_interior_shape,
                )
                record["assignment"] = prior_assignment_diagnostics(
                    fit_result.prior,
                    selection_bank,
                    device=device,
                    component_mass_threshold=safe_component_mass_threshold,
                )
            records.append(record)
            successful.append((candidate, fit_result.prior, record, scores))
        except Exception as exc:
            records.append(
                {
                    "candidate": candidate.name,
                    "complexity_rank": int(candidate.complexity_rank),
                    "status": "failed",
                    "fit_and_score_seconds": float(time.perf_counter() - candidate_start),
                    "error": repr(exc),
                }
            )

    if not successful:
        raise RuntimeError(f"No prior candidate fitted successfully: {records}")

    best = max(successful, key=lambda item: float(item[2]["selection_mean_log_score"]))
    best_mean = float(best[2]["selection_mean_log_score"])
    best_se = float(best[2]["selection_standard_error"])
    threshold = best_mean - best_se
    one_se_selected = _one_se_selection(successful, threshold)
    shape_gate_diagnostics: dict[str, Any] = {}
    if selection_rule == "best_score":
        selected = best
    elif selection_rule == "student_t_safe":
        anchor = next(
            (item for item in successful if item[0].name == "student_t"),
            None,
        )
        if anchor is None:
            raise ValueError("student_t_safe selection requires a student_t candidate.")
        anchor_assignment = anchor[2]["assignment"]
        component_skewness = prior_component_shape_diagnostics(
            anchor[1],
            selection_bank,
            device=device,
            component_mass_threshold=safe_component_mass_threshold,
        )
        safe_candidates: list[tuple[PriorCandidate, Prior, dict[str, Any], Tensor]] = []
        safe_records: list[dict[str, Any]] = []
        n_safe_comparisons = sum(
            item[0].name in ASYMMETRIC_CANDIDATES for item in successful
        )
        if float(safe_confidence_z) > 0.0:
            confidence_z = float(safe_confidence_z)
        else:
            alpha = min(max(float(safe_familywise_alpha), 1e-12), 1.0 - 1e-12)
            confidence_z = float(
                NormalDist().inv_cdf(1.0 - alpha / max(n_safe_comparisons, 1))
            )
        anchor_active = int(anchor_assignment["active_components"])
        minimum_active = max(
            1,
            int(math.ceil(float(safe_min_active_fraction) * anchor_active)),
        )
        for item in successful:
            if item[0].name == "student_t" or item[0].name not in ASYMMETRIC_CANDIDATES:
                continue
            difference = item[3] - anchor[3]
            difference_mean = float(difference.mean())
            difference_se = float(
                difference.std(unbiased=True) / math.sqrt(difference.numel())
            )
            lower_bound = difference_mean - confidence_z * difference_se
            assignment = item[2]["assignment"]
            regularity = item[2]["regularity"]
            agreement = prior_assignment_agreement_diagnostics(
                anchor[1],
                item[1],
                selection_bank,
                device=device,
                core_confidence_threshold=safe_core_confidence_threshold,
            )
            item[2]["anchor_assignment_agreement"] = agreement
            reasons: list[str] = []
            if not bool(regularity["regular"]):
                reasons.append("irregular_parameters")
            if lower_bound < float(shape_score_margin):
                reasons.append("insufficient_paired_score_gain")
            component_skew_max = float(
                component_skewness["anchor_component_abs_skew_max"]
            )
            if not math.isfinite(component_skew_max) or component_skew_max < float(
                shape_skew_threshold
            ):
                reasons.append("insufficient_component_skew")
            if int(assignment["active_components"]) < minimum_active:
                reasons.append("component_collapse")
            if float(agreement["core_fraction"]) < float(safe_minimum_core_fraction):
                reasons.append("insufficient_anchor_core")
            core_retention = float(agreement["core_retention"])
            if not math.isfinite(core_retention) or core_retention < float(
                safe_minimum_core_retention
            ):
                reasons.append("degraded_anchor_core")
            for metric in (
                "normalized_marginal_entropy",
                "normalized_assignment_mi",
                "draw_assignment_stability",
            ):
                if float(assignment[metric]) + float(safe_assignment_tolerance) < float(
                    anchor_assignment[metric]
                ):
                    reasons.append(f"degraded_{metric}")
            safe_record = {
                "candidate": item[0].name,
                "paired_score_gain": difference_mean,
                "paired_score_standard_error": difference_se,
                "paired_score_lower_bound": lower_bound,
                "eligible": not reasons,
                "rejection_reasons": reasons,
            }
            item[2]["safe_anchor_comparison"] = safe_record
            safe_records.append(safe_record)
            if not reasons:
                safe_candidates.append(item)
        if safe_candidates:
            safest_best = max(safe_candidates, key=_held_out_mean)
            safest_threshold = _held_out_mean(safest_best) - float(
                safest_best[2]["selection_standard_error"]
            )
            selected = _one_se_selection(safe_candidates, safest_threshold)
        else:
            selected = anchor
        shape_gate_diagnostics = {
            "safe_anchor_candidate": anchor[0].name,
            "safe_anchor_assignment": anchor_assignment,
            "safe_confidence_z": confidence_z,
            "safe_familywise_alpha": float(safe_familywise_alpha),
            "safe_n_familywise_comparisons": int(n_safe_comparisons),
            "safe_min_active_fraction": float(safe_min_active_fraction),
            "safe_minimum_active_components": minimum_active,
            "safe_assignment_tolerance": float(safe_assignment_tolerance),
            "safe_component_mass_threshold": float(safe_component_mass_threshold),
            "safe_core_confidence_threshold": float(safe_core_confidence_threshold),
            "safe_minimum_core_fraction": float(safe_minimum_core_fraction),
            "safe_minimum_core_retention": float(safe_minimum_core_retention),
            "safe_score_margin": float(shape_score_margin),
            "safe_component_skew_threshold": float(shape_skew_threshold),
            "safe_component_skewness": component_skewness,
            "safe_candidate_comparisons": safe_records,
            "safe_promoted": selected is not anchor,
        }
    elif selection_rule == "shape_gated":
        selected = one_se_selected
        best_asymmetric = max(
            (item for item in successful if item[0].name in ASYMMETRIC_CANDIDATES),
            key=_held_out_mean,
            default=None,
        )
        best_symmetric = max(
            (item for item in successful if item[0].name in SYMMETRIC_CANDIDATES),
            key=_held_out_mean,
            default=None,
        )
        latent_abs_skew = _mean_abs_marginal_skewness(selection_bank)
        asymmetric_margin = math.nan
        shape_gate_triggered = False
        student_t_boundary_diagnostics: dict[str, Any] | None = None
        student_t_score_margin = math.nan
        if best_asymmetric is not None and best_symmetric is not None:
            asymmetric_margin = _held_out_mean(best_asymmetric) - _held_out_mean(best_symmetric)
            shape_gate_triggered = (
                latent_abs_skew >= float(shape_skew_threshold)
                and asymmetric_margin >= float(shape_score_margin)
            )
            if shape_gate_triggered:
                selected = best_asymmetric
        student_t_anchor = next(
            (item for item in successful if item[0].name == "student_t"),
            None,
        )
        if best_asymmetric is not None and student_t_anchor is not None:
            student_t_score_margin = _held_out_mean(best_asymmetric) - _held_out_mean(
                student_t_anchor
            )
            try:
                student_t_boundary_diagnostics = prior_boundary_assignment_diagnostics(
                    student_t_anchor[1],
                    best_asymmetric[1],
                    selection_bank,
                    device=device,
                    boundary_fraction=0.2,
                )
            except Exception as exc:
                student_t_boundary_diagnostics = {
                    "status": "failed",
                    "error": repr(exc),
                }
        shape_gate_diagnostics = {
            "shape_gate_abs_skew": latent_abs_skew,
            "shape_skew_threshold": float(shape_skew_threshold),
            "shape_score_margin": float(shape_score_margin),
            "shape_gate_best_asymmetric_candidate": (
                best_asymmetric[0].name if best_asymmetric is not None else None
            ),
            "shape_gate_best_symmetric_candidate": (
                best_symmetric[0].name if best_symmetric is not None else None
            ),
            "shape_gate_asymmetric_margin": asymmetric_margin,
            "shape_gate_student_t_score_margin": student_t_score_margin,
            "shape_gate_student_t_boundary_diagnostics": (
                student_t_boundary_diagnostics
            ),
            "shape_gate_triggered": shape_gate_triggered,
        }
    else:
        selected = one_se_selected
    selected[2]["selected"] = True
    diagnostics = {
        "selection_rule": selection_rule,
        "score_unit": "observation_mean_over_mc_draws",
        "best_candidate": best[0].name,
        "best_mean_log_score": best_mean,
        "best_standard_error": best_se,
        "one_se_threshold": threshold,
        "one_se_selected_candidate": one_se_selected[0].name,
        "selected_candidate": selected[0].name,
        "n_candidates": len(candidates),
        "n_successful_candidates": len(successful),
        "candidates": records,
        **shape_gate_diagnostics,
    }
    return PriorSelectionResult(
        prior=selected[1],
        candidate_name=selected[0].name,
        diagnostics=diagnostics,
    )
