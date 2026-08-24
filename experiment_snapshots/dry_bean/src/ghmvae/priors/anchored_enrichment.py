from __future__ import annotations

import copy
import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from sklearn.neighbors import NearestNeighbors

from ghmvae.priors.base import Prior
from ghmvae.priors.fmvmm_adapter import _fit_timeout
from ghmvae.priors.selection import (
    prior_assignment_agreement_diagnostics,
    prior_assignment_diagnostics,
    prior_boundary_correction_diagnostics,
    prior_component_shape_diagnostics,
    prior_per_observation_score,
    prior_regularity_diagnostics,
)
from ghmvae.priors.torch_fmvmm import (
    MultivariateTComponent,
    TorchMixturePrior,
    component_from_fmvmm,
)
from ghmvae.priors.transformed import AffineTransformedPrior


ENRICHMENT_FAMILIES = {
    "mgst": ("skew_t", 2, "fmvmm.distributions.multivariate_genskewt"),
    "mvvg": ("variance_gamma", 3, "fmvmm.distributions.multivariate_vargamma"),
    "mvhb": ("hyperbolic", 4, "fmvmm.distributions.multivariate_hyperbolic"),
    "mnig": ("nig", 5, "fmvmm.distributions.multivariate_norminvgauss"),
    "mghp": ("full_gh", 6, "fmvmm.distributions.multivariate_genhyperbolic"),
}


@dataclass(frozen=True)
class ComponentwiseEnrichmentConfig:
    families: tuple[str, ...] = ("mgst", "mghp")
    familywise_alpha: float = 0.05
    confidence_z: float = 0.0
    boundary_confidence_z: float = 1.6448536269514722
    score_margin: float = 0.0
    component_skew_threshold: float = 0.25
    component_skew_signal_threshold: float = 2.0
    minimum_effective_component_size: float = 30.0
    minimum_anchor_normalized_entropy: float = 0.8
    minimum_active_fraction: float = 0.8
    assignment_tolerance: float = 0.05
    component_mass_threshold: float = 0.01
    core_confidence_threshold: float = 0.8
    minimum_core_fraction: float = 0.05
    minimum_core_retention: float = 0.95
    boundary_log_odds_threshold: float = 2.0
    minimum_boundary_effective_size: float = 10.0
    minimum_boundary_score_gain: float = 0.0
    minimum_boundary_leverage: float = 0.01
    minimum_decision_change_fraction: float = 0.005
    graph_refinement: bool = True
    graph_neighbors: int = 10
    graph_density_one_se_multiplier: float = 1.0
    minimum_optimization_graph_gain: float = 0.0
    minimum_weight: float = 1e-4
    minimum_covariance_eigenvalue: float = 1e-5
    maximum_covariance_condition: float = 1e6
    minimum_interior_shape: float = 1e-4
    max_fit_iterations: int = 25
    fit_relative_tolerance: float = 1e-4
    fit_absolute_tolerance: float = 1e-5
    fit_timeout_seconds: float = 0.0


@dataclass
class ComponentwiseEnrichmentResult:
    prior: Prior
    selected_family: str
    selected_component: int | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _GraphFitContext:
    anchor_log_joint: np.ndarray
    component_log_weight: float
    component: int
    sample_count: int
    observation_count: int
    source: np.ndarray
    target: np.ndarray


def _validate_config(config: ComponentwiseEnrichmentConfig) -> None:
    unknown = sorted(set(config.families) - set(ENRICHMENT_FAMILIES))
    if unknown:
        raise ValueError(f"Unsupported enrichment families: {unknown}")
    if not config.families:
        raise ValueError("At least one enrichment family is required.")
    if not 0.0 < float(config.familywise_alpha) < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one.")
    if float(config.boundary_confidence_z) < 0.0:
        raise ValueError("boundary_confidence_z must be nonnegative.")
    if not 0.0 < float(config.minimum_active_fraction) <= 1.0:
        raise ValueError("minimum_active_fraction must lie in (0, 1].")
    for name in (
        "component_mass_threshold",
        "core_confidence_threshold",
        "minimum_core_fraction",
        "minimum_core_retention",
    ):
        value = float(getattr(config, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
    if float(config.assignment_tolerance) < 0.0:
        raise ValueError("assignment_tolerance must be nonnegative.")
    if float(config.minimum_effective_component_size) <= 0.0:
        raise ValueError("minimum_effective_component_size must be positive.")
    if float(config.component_skew_signal_threshold) < 0.0:
        raise ValueError("component_skew_signal_threshold must be nonnegative.")
    if not 0.0 <= float(config.minimum_anchor_normalized_entropy) <= 1.0:
        raise ValueError("minimum_anchor_normalized_entropy must lie in [0, 1].")
    if float(config.boundary_log_odds_threshold) <= 0.0:
        raise ValueError("boundary_log_odds_threshold must be positive.")
    if float(config.minimum_boundary_effective_size) <= 0.0:
        raise ValueError("minimum_boundary_effective_size must be positive.")
    if float(config.minimum_boundary_leverage) < 0.0:
        raise ValueError("minimum_boundary_leverage must be nonnegative.")
    if not 0.0 <= float(config.minimum_decision_change_fraction) <= 1.0:
        raise ValueError("minimum_decision_change_fraction must lie in [0, 1].")
    if int(config.graph_neighbors) <= 0:
        raise ValueError("graph_neighbors must be positive.")
    if float(config.graph_density_one_se_multiplier) < 0.0:
        raise ValueError("graph_density_one_se_multiplier must be nonnegative.")
    if float(config.minimum_optimization_graph_gain) < 0.0:
        raise ValueError("minimum_optimization_graph_gain must be nonnegative.")


def _knn_edge_index(
    features: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Graph features must be a matrix.")
    if features.shape[0] < 2:
        raise ValueError("Graph refinement requires at least two observations.")
    scale = features.std(axis=0, keepdims=True)
    standardized = (features - features.mean(axis=0, keepdims=True)) / np.maximum(
        scale, 1e-8
    )
    neighbor_count = min(max(int(n_neighbors), 1), features.shape[0] - 1)
    neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(
        standardized
    ).kneighbors(standardized, return_distance=False)[:, 1:]
    source = np.repeat(np.arange(features.shape[0]), neighbor_count)
    return source, neighbors.reshape(-1)


def _soft_graph_assortativity_from_log_joint(
    log_joint: np.ndarray,
    sample_count: int,
    observation_count: int,
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return soft assignment assortativity after averaging posterior draws."""

    log_joint = np.asarray(log_joint, dtype=np.float64)
    expected_rows = int(sample_count) * int(observation_count)
    if log_joint.ndim != 2 or log_joint.shape[0] != expected_rows:
        raise ValueError("log_joint has an incompatible shape.")
    shifted = log_joint - np.max(log_joint, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-300)
    responsibilities = probabilities.reshape(
        int(sample_count), int(observation_count), -1
    ).mean(axis=0)
    edge_agreement = float(
        np.einsum(
            "ij,ij->i",
            responsibilities[source],
            responsibilities[target],
        ).mean()
    )
    mass = responsibilities.mean(axis=0)
    null_agreement = float(np.square(mass).sum())
    return float(
        (edge_agreement - null_agreement)
        / max(1.0 - null_agreement, 1e-12)
    )


def _canonical_gh_chi_psi(alpha_bar: float, lmbda: float) -> tuple[float, float]:
    """Map canonical GH shape parameters to a unit-mean GIG mixing law."""

    from scipy.special import kve

    alpha_bar = float(alpha_bar)
    lmbda = float(lmbda)
    if alpha_bar < 0.0:
        raise ValueError("alpha_bar must be nonnegative.")
    if alpha_bar <= 1e-12:
        if lmbda > 0.0:
            return 0.0, 2.0 * lmbda
        if lmbda < -1.0:
            return -2.0 * (lmbda + 1.0), 0.0
        raise ValueError(
            "The alpha_bar=0 boundary requires lambda>0 or lambda<-1."
        )
    denominator = float(kve(lmbda, alpha_bar))
    numerator = float(kve(lmbda + 1.0, alpha_bar))
    if not (
        math.isfinite(denominator)
        and math.isfinite(numerator)
        and denominator > 0.0
        and numerator > 0.0
    ):
        raise FloatingPointError("The canonical GH Bessel ratio is non-finite.")
    if lmbda >= 0.0:
        psi = alpha_bar * numerator / denominator
        chi = alpha_bar * alpha_bar / psi
    else:
        chi = alpha_bar * denominator / numerator
        psi = alpha_bar * alpha_bar / chi
    if not (math.isfinite(chi) and math.isfinite(psi) and chi >= 0.0 and psi >= 0.0):
        raise FloatingPointError("The canonical GH shape map is non-finite.")
    return float(chi), float(psi)


def _base_and_transform(
    anchor_prior: Prior,
) -> tuple[TorchMixturePrior, Tensor | None, Tensor | None]:
    if isinstance(anchor_prior, AffineTransformedPrior):
        base = anchor_prior.base_prior
        center = anchor_prior.center.detach().float().cpu()
        scale = anchor_prior.scale.detach().float().cpu()
    else:
        base = anchor_prior
        center = None
        scale = None
    if not isinstance(base, TorchMixturePrior):
        raise TypeError("Componentwise enrichment requires a torch-native FMVMM anchor.")
    if not all(isinstance(component, MultivariateTComponent) for component in base.components):
        raise TypeError("Componentwise enrichment requires a homogeneous Student-t anchor.")
    return base, center, scale


def _standardize(z: Tensor, center: Tensor | None, scale: Tensor | None) -> Tensor:
    z_cpu = z.detach().float().cpu()
    if center is None or scale is None:
        return z_cpu
    return (z_cpu - center) / scale


def _compose_candidate(
    anchor_base: TorchMixturePrior,
    replacement_component: int,
    replacement: torch.nn.Module,
    center: Tensor | None,
    scale: Tensor | None,
) -> Prior:
    components = [copy.deepcopy(component) for component in anchor_base.components]
    components[int(replacement_component)] = replacement
    family_names = [str(getattr(component, "family", "unknown")) for component in components]
    base = TorchMixturePrior(
        anchor_base.log_weights.detach().cpu(),
        components,
        name="anchored_" + "_".join(family_names),
    )
    if center is None or scale is None:
        return base
    return AffineTransformedPrior(base, center=center, scale=scale)


def _initial_fit_kwargs(
    family: str,
    component: MultivariateTComponent,
    config: ComponentwiseEnrichmentConfig,
) -> dict[str, Any]:
    mu = component.mu.detach().double().cpu().numpy()
    sigma = component.sigma.detach().double().cpu().numpy()
    gamma = np.zeros_like(mu)
    common: dict[str, Any] = {
        "nit": int(config.max_fit_iterations),
        "reltol": float(config.fit_relative_tolerance),
        "abstol": float(config.fit_absolute_tolerance),
        "silent": True,
        "fmu": mu,
        "fsigma": sigma,
        "fgamma": gamma,
    }
    df = max(float(component.df), 2.1)
    if family == "mgst":
        common["lmbda"] = min(-1.05, -(df + 2.0) / 2.0)
    elif family == "mvvg":
        common["lmbda"] = max(1.05, df / 2.0)
    elif family in {"mvhb", "mnig"}:
        common["alpha_bar"] = 1.0
    elif family == "mghp":
        common["lmbda"] = min(-1.05, -df / 2.0)
        common["alpha_bar"] = 0.5
    return common


def _weighted_ratio_gain(
    candidate_log_prob: np.ndarray,
    anchor_log_prob: np.ndarray,
    weights: np.ndarray,
    sample_count: int,
    observation_count: int,
) -> tuple[float, float]:
    """Estimate E[r h] / E[r], keeping posterior draws within observations."""

    difference = (candidate_log_prob - anchor_log_prob).reshape(
        int(sample_count), int(observation_count)
    )
    draw_weights = weights.reshape(int(sample_count), int(observation_count))
    numerator = (draw_weights * difference).mean(axis=0)
    denominator = draw_weights.mean(axis=0)
    mean_denominator = max(float(denominator.mean()), 1e-12)
    estimate = float(numerator.mean() / mean_denominator)
    influence = (numerator - estimate * denominator) / mean_denominator
    standard_error = float(
        np.std(influence, ddof=1) / math.sqrt(int(observation_count))
    )
    return estimate, standard_error


def _candidate_graph_score(
    component_log_prob: np.ndarray,
    context: _GraphFitContext,
) -> float:
    candidate_joint = np.array(context.anchor_log_joint, copy=True)
    candidate_joint[:, int(context.component)] = (
        np.asarray(component_log_prob, dtype=np.float64)
        + float(context.component_log_weight)
    )
    return _soft_graph_assortativity_from_log_joint(
        candidate_joint,
        context.sample_count,
        context.observation_count,
        context.source,
        context.target,
    )


def _fit_nested_gh_component(
    family: str,
    module: Any,
    component: MultivariateTComponent,
    points: np.ndarray,
    weights: np.ndarray,
    config: ComponentwiseEnrichmentConfig,
    graph_context: _GraphFitContext | None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Fit an exact-t-nested skew-t or canonical full-GH replacement."""

    from scipy.optimize import minimize

    anchor_df = max(float(component.df), 2.0001)
    anchor_mu = component.mu.detach().double().cpu().numpy()
    anchor_sigma = component.sigma.detach().double().cpu().numpy()
    anchor_canonical_sigma = (anchor_df / (anchor_df - 2.0)) * anchor_sigma
    zero = np.zeros_like(anchor_mu)
    scale = np.sqrt(np.clip(np.diag(anchor_canonical_sigma), 1e-8, None))
    fit_start = time.perf_counter()

    def component_values(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        value = np.asarray(value, dtype=float)
        if family == "mgst":
            df = 2.0 + math.exp(float(value[0]))
            gamma = value[1:]
            canonical_sigma = (df / (df - 2.0)) * anchor_sigma
            lmbda = -df / 2.0
            chi = df - 2.0
            psi = 0.0
            alpha_bar = 0.0
            values = module.logpdf(
                points,
                lmbda,
                chi,
                anchor_mu - gamma,
                canonical_sigma,
                gamma,
            )
        else:
            lmbda = float(value[0])
            alpha_bar = max(float(value[1]), 0.0)
            gamma = value[2:]
            chi, psi = _canonical_gh_chi_psi(alpha_bar, lmbda)
            canonical_sigma = anchor_canonical_sigma
            values = module.logpdf(
                points,
                lmbda,
                chi,
                psi,
                anchor_mu - gamma,
                canonical_sigma,
                gamma,
            )
            df = -2.0 * lmbda if lmbda < -1.0 else math.nan
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != points.shape[0] or not np.all(np.isfinite(values)):
            raise FloatingPointError("Nested GH candidate produced a non-finite score.")
        return values, {
            "df": float(df),
            "lmbda": float(lmbda),
            "alpha_bar": float(alpha_bar),
            "chi": float(chi),
            "psi": float(psi),
            "gamma": np.asarray(gamma, dtype=float),
            "sigma": np.asarray(canonical_sigma, dtype=float),
        }

    def objective(value: np.ndarray) -> float:
        try:
            values, _ = component_values(value)
        except Exception:
            return 1e100
        return -float(np.dot(weights, values))

    mean = np.average(points, axis=0, weights=weights)
    residual = points - mean
    covariance = (residual.T @ (weights[:, None] * residual)) / max(
        float(weights.sum()), 1e-12
    )
    covariance = covariance + 1e-6 * np.eye(covariance.shape[0])
    inverse = np.linalg.pinv(covariance)
    mahalanobis = np.einsum("ni,ij,nj->n", residual, inverse, residual)
    skew_direction = np.sum(
        weights[:, None] * residual * mahalanobis[:, None], axis=0
    )
    direction_norm = float(np.linalg.norm(skew_direction))
    gamma_start = (
        0.25 * float(np.mean(scale)) * skew_direction / direction_norm
        if direction_norm > 1e-12
        else zero
    )

    if family == "mgst":
        moderate_df = min(max(anchor_df, 8.0), 20.0)
        initial = np.concatenate(([math.log(anchor_df - 2.0)], zero))
        starts = [
            initial,
            np.concatenate(([math.log(moderate_df - 2.0)], zero)),
            np.concatenate(([math.log(moderate_df - 2.0)], gamma_start)),
            np.concatenate(([math.log(moderate_df - 2.0)], -gamma_start)),
        ]
        maximum_df = max(anchor_df * 10.0, 1e4)
        bounds = [
            (math.log(0.05), math.log(maximum_df - 2.0)),
            *[(-5.0 * item, 5.0 * item) for item in scale],
        ]
        fit_mode = "student_t_nested_tail_and_skew"
    else:
        anchor_lmbda = -anchor_df / 2.0
        hyperbolic_lmbda = (points.shape[1] + 1.0) / 2.0
        initial = np.concatenate(([anchor_lmbda, 0.0], zero))
        starts = [
            initial,
            np.concatenate(([anchor_lmbda, 0.25], zero)),
            np.concatenate(([anchor_lmbda, 0.5], gamma_start)),
            np.concatenate(([-0.5, 0.5], gamma_start)),
            np.concatenate(([hyperbolic_lmbda, 0.5], gamma_start)),
        ]
        bounds = [
            (-50.0, 50.0),
            (0.0, 50.0),
            *[(-5.0 * item, 5.0 * item) for item in scale],
        ]
        fit_mode = "student_t_closure_nested_canonical_full_gh"

    initial_values, _ = component_values(initial)
    initial_objective = objective(initial)
    density_results: list[Any] = []
    graph_results: list[Any] = []
    with _fit_timeout(config.fit_timeout_seconds):
        for start in starts:
            density_results.append(
                minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": int(config.max_fit_iterations),
                        "ftol": float(config.fit_relative_tolerance),
                        "gtol": float(config.fit_absolute_tolerance),
                        "maxls": 30,
                    },
                )
            )
        finite_density = [
            result for result in density_results if np.isfinite(result.fun)
        ]
        density_result = min(
            finite_density,
            key=lambda item: float(item.fun),
            default=None,
        )
        density_value = (
            np.asarray(density_result.x, dtype=float)
            if density_result is not None
            else initial
        )
        density_objective = objective(density_value)
        if (
            not math.isfinite(density_objective)
            or density_objective > initial_objective
        ):
            density_value = initial
            density_objective = initial_objective

        final_value = density_value
        final_objective = density_objective
        density_values, _ = component_values(density_value)
        graph_refinement_used = False
        graph_result = None
        anchor_graph_score = math.nan
        density_graph_score = math.nan
        final_graph_score = math.nan
        density_gain, density_gain_se = _weighted_ratio_gain(
            density_values,
            initial_values,
            weights,
            graph_context.sample_count if graph_context is not None else 1,
            graph_context.observation_count
            if graph_context is not None
            else points.shape[0],
        )
        density_floor = max(
            0.0,
            density_gain
            - float(config.graph_density_one_se_multiplier) * density_gain_se,
        )
        if bool(config.graph_refinement) and graph_context is not None:
            anchor_graph_score = _candidate_graph_score(
                initial_values, graph_context
            )
            density_graph_score = _candidate_graph_score(
                density_values, graph_context
            )

            def graph_objective(value: np.ndarray) -> float:
                try:
                    values, _ = component_values(value)
                    score = _candidate_graph_score(values, graph_context)
                except Exception:
                    return 1e6
                return -score if math.isfinite(score) else 1e6

            def density_constraint(value: np.ndarray) -> float:
                try:
                    values, _ = component_values(value)
                    gain, _ = _weighted_ratio_gain(
                        values,
                        initial_values,
                        weights,
                        graph_context.sample_count,
                        graph_context.observation_count,
                    )
                except Exception:
                    return -1e6
                return float(gain - density_floor)

            refinement_starts = [density_value]
            refinement_starts.extend(
                np.asarray(result.x, dtype=float)
                for result in finite_density[:2]
                if np.isfinite(result.fun)
            )
            for start in refinement_starts:
                graph_results.append(
                    minimize(
                        graph_objective,
                        start,
                        method="SLSQP",
                        bounds=bounds,
                        constraints={"type": "ineq", "fun": density_constraint},
                        options={
                            "maxiter": int(config.max_fit_iterations),
                            "ftol": float(config.fit_relative_tolerance),
                        },
                    )
                )
            feasible_graph: list[Any] = []
            for result in graph_results:
                if not np.isfinite(result.fun):
                    continue
                if density_constraint(np.asarray(result.x, dtype=float)) < -1e-6:
                    continue
                feasible_graph.append(result)
            graph_result = min(
                feasible_graph,
                key=lambda item: float(item.fun),
                default=None,
            )
            if graph_result is not None:
                candidate_value = np.asarray(graph_result.x, dtype=float)
                candidate_values, _ = component_values(candidate_value)
                candidate_graph_score = _candidate_graph_score(
                    candidate_values, graph_context
                )
                if candidate_graph_score > density_graph_score + float(
                    config.minimum_optimization_graph_gain
                ):
                    final_value = candidate_value
                    final_objective = objective(final_value)
                    final_graph_score = candidate_graph_score
                    graph_refinement_used = True
        if not math.isfinite(final_graph_score):
            final_graph_score = density_graph_score

    final_values, parameters = component_values(final_value)
    final_gain, final_gain_se = _weighted_ratio_gain(
        final_values,
        initial_values,
        weights,
        graph_context.sample_count if graph_context is not None else 1,
        graph_context.observation_count
        if graph_context is not None
        else points.shape[0],
    )
    gamma = parameters["gamma"]
    if family == "mgst":
        alpha = (
            parameters["lmbda"],
            parameters["chi"],
            anchor_mu - gamma,
            parameters["sigma"],
            gamma,
        )
    else:
        alpha = (
            parameters["lmbda"],
            parameters["chi"],
            parameters["psi"],
            anchor_mu - gamma,
            parameters["sigma"],
            gamma,
        )
    replacement = component_from_fmvmm(module, alpha)
    return replacement, {
        "family_code": family,
        "fit_mode": fit_mode,
        "optimizer_success": bool(
            graph_result.success
            if graph_refinement_used and graph_result is not None
            else density_result.success
            if density_result is not None
            else False
        ),
        "optimizer_message": str(
            graph_result.message
            if graph_refinement_used and graph_result is not None
            else density_result.message
            if density_result is not None
            else "no_finite_result"
        ),
        "weighted_component_log_score_gain": float(
            initial_objective - final_objective
        ),
        "weighted_component_mean_log_score_gain": final_gain,
        "weighted_component_mean_log_score_gain_standard_error": final_gain_se,
        "anchor_df": anchor_df,
        "fitted_df": parameters["df"],
        "fitted_lmbda": parameters["lmbda"],
        "fitted_alpha_bar": parameters["alpha_bar"],
        "fitted_chi": parameters["chi"],
        "fitted_psi": parameters["psi"],
        "skewness_norm": float(np.linalg.norm(gamma)),
        "optimization_graph_refinement_enabled": bool(config.graph_refinement),
        "optimization_graph_refinement_used": graph_refinement_used,
        "optimization_anchor_graph_assortativity": anchor_graph_score,
        "optimization_density_graph_assortativity": density_graph_score,
        "optimization_final_graph_assortativity": final_graph_score,
        "optimization_graph_assortativity_gain": (
            float(final_graph_score - anchor_graph_score)
            if math.isfinite(final_graph_score) and math.isfinite(anchor_graph_score)
            else math.nan
        ),
        "optimization_density_score_floor": density_floor,
        "fit_seconds": float(time.perf_counter() - fit_start),
    }


def _fit_replacement_component(
    family: str,
    component: MultivariateTComponent,
    points: np.ndarray,
    weights: np.ndarray,
    config: ComponentwiseEnrichmentConfig,
    graph_context: _GraphFitContext | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    display_name, _, module_name = ENRICHMENT_FAMILIES[family]
    module = importlib.import_module(module_name)
    if family in {"mgst", "mghp"}:
        replacement, diagnostics = _fit_nested_gh_component(
            family,
            module,
            component,
            points,
            weights,
            config,
            graph_context,
        )
        diagnostics["family"] = display_name
        return replacement, diagnostics
    kwargs = _initial_fit_kwargs(family, component, config)
    fit_start = time.perf_counter()
    with _fit_timeout(config.fit_timeout_seconds):
        alpha = module.fit_weighted(points, weights=weights, **kwargs)
    replacement = component_from_fmvmm(module, alpha)
    return replacement, {
        "family_code": family,
        "family": display_name,
        "fit_mode": "non_nested_catalogue_extension",
        "fit_seconds": float(time.perf_counter() - fit_start),
    }


@torch.no_grad()
def _component_conditional_score_difference(
    anchor_prior: Prior,
    candidate_prior: Prior,
    z_bank: Tensor,
    component: int,
    device: str,
) -> tuple[float, float, Tensor]:
    """Return a soft-component conditional log-score difference and ratio SE."""

    sample_count, n_observations, latent_dim = z_bank.shape
    flat = z_bank.reshape(sample_count * n_observations, latent_dim).to(device)
    anchor_prior.to(device)
    candidate_prior.to(device)
    anchor_component_log_prob = anchor_prior.component_log_prob(flat).reshape(
        sample_count, n_observations, -1
    )
    candidate_component_log_prob = candidate_prior.component_log_prob(flat).reshape(
        sample_count, n_observations, -1
    )
    anchor_responsibilities = anchor_prior.responsibilities(flat).reshape(
        sample_count, n_observations, -1
    )
    difference_draw = (
        candidate_component_log_prob[:, :, int(component)]
        - anchor_component_log_prob[:, :, int(component)]
    ).cpu()
    weight_draw = anchor_responsibilities[:, :, int(component)].cpu()
    numerator = (weight_draw * difference_draw).mean(dim=0)
    weights = weight_draw.mean(dim=0)
    mean_weight = weights.mean().clamp_min(1e-8)
    estimate = numerator.mean() / mean_weight
    influence = (numerator - estimate * weights) / mean_weight
    standard_error = influence.std(unbiased=True) / math.sqrt(n_observations)
    return float(estimate), float(standard_error), influence


def fit_componentwise_gh_enrichment(
    anchor_prior: Prior,
    optimization_bank: Tensor,
    selection_bank: Tensor,
    *,
    config: ComponentwiseEnrichmentConfig | None = None,
    fmvmm_path: str | None = None,
    device: str = "cpu",
) -> ComponentwiseEnrichmentResult:
    """Safely replace at most one Student-t component by a GH-family component."""

    config = config or ComponentwiseEnrichmentConfig()
    _validate_config(config)
    if optimization_bank.ndim != 3 or selection_bank.ndim != 3:
        raise ValueError("Posterior banks must have shape (S, N, D).")
    if optimization_bank.shape[-1] != selection_bank.shape[-1]:
        raise ValueError("Optimization and selection banks need the same latent dimension.")
    if selection_bank.shape[1] < 2:
        raise ValueError("The selection bank needs at least two observations.")
    if fmvmm_path is not None:
        path = str(Path(fmvmm_path).expanduser())
        if path not in sys.path:
            sys.path.insert(0, path)

    anchor_base, center, scale = _base_and_transform(anchor_prior)
    sample_count, n_optimization, latent_dim = optimization_bank.shape
    flat_optimization = optimization_bank.reshape(
        sample_count * n_optimization, latent_dim
    )
    standardized_points = _standardize(flat_optimization, center, scale).numpy()
    with torch.no_grad():
        anchor_prior.to(device)
        optimization_responsibilities = anchor_prior.responsibilities(
            flat_optimization.to(device)
        ).detach().cpu().numpy()
        anchor_base.to("cpu")
        standardized_tensor = torch.as_tensor(
            standardized_points, dtype=torch.float32
        )
        anchor_component_log_prob = (
            anchor_base.component_log_prob(standardized_tensor)
            .detach()
            .double()
            .cpu()
            .numpy()
        )
        anchor_log_weights = (
            anchor_base.log_weights.detach().double().cpu().numpy()
        )
    optimization_features = standardized_points.reshape(
        sample_count, n_optimization, latent_dim
    ).mean(axis=0)
    graph_source, graph_target = _knn_edge_index(
        optimization_features,
        config.graph_neighbors,
    )
    anchor_log_joint = anchor_component_log_prob + anchor_log_weights[None, :]

    anchor_scores = prior_per_observation_score(anchor_prior, selection_bank, device)
    anchor_assignment = prior_assignment_diagnostics(
        anchor_prior,
        selection_bank,
        device=device,
        component_mass_threshold=config.component_mass_threshold,
    )
    component_shapes = prior_component_shape_diagnostics(
        anchor_prior,
        selection_bank,
        device=device,
        component_mass_threshold=config.component_mass_threshold,
    )
    shape_by_component = {
        int(record["component"]): record for record in component_shapes["components"]
    }
    minimum_active = max(
        1,
        int(
            math.ceil(
                float(config.minimum_active_fraction)
                * int(anchor_assignment["active_components"])
            )
        ),
    )

    attempted = len(anchor_base.components) * len(config.families)
    if float(config.confidence_z) > 0.0:
        confidence_z = float(config.confidence_z)
    else:
        confidence_z = float(
            NormalDist().inv_cdf(
                1.0 - float(config.familywise_alpha) / max(attempted, 1)
            )
        )

    successful: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for component_index, anchor_component in enumerate(anchor_base.components):
        fit_weights = optimization_responsibilities[:, component_index]
        effective_size = float(
            fit_weights.sum() ** 2 / max(float(np.square(fit_weights).sum()), 1e-12)
        )
        for family in config.families:
            display_name, complexity_rank, _ = ENRICHMENT_FAMILIES[family]
            record: dict[str, Any] = {
                "component": int(component_index),
                "family_code": family,
                "family": display_name,
                "complexity_rank": int(complexity_rank),
                "optimization_effective_sample_size": effective_size,
            }
            candidate_start = time.perf_counter()
            try:
                if effective_size < float(config.minimum_effective_component_size):
                    raise ValueError(
                        "effective component size "
                        f"{effective_size:.3f}<{config.minimum_effective_component_size:.3f}"
                    )
                replacement, fit_diagnostics = _fit_replacement_component(
                    family,
                    anchor_component,
                    standardized_points,
                    fit_weights,
                    config,
                    _GraphFitContext(
                        anchor_log_joint=anchor_log_joint,
                        component_log_weight=float(
                            anchor_log_weights[component_index]
                        ),
                        component=component_index,
                        sample_count=sample_count,
                        observation_count=n_optimization,
                        source=graph_source,
                        target=graph_target,
                    ),
                )
                candidate_prior = _compose_candidate(
                    anchor_base,
                    component_index,
                    replacement,
                    center,
                    scale,
                )
                candidate_scores = prior_per_observation_score(
                    candidate_prior, selection_bank, device
                )
                if not bool(torch.isfinite(candidate_scores).all()):
                    raise ValueError("Candidate produced non-finite held-out scores.")
                mixture_difference = candidate_scores - anchor_scores
                component_difference_mean, component_difference_se, _ = (
                    _component_conditional_score_difference(
                        anchor_prior,
                        candidate_prior,
                        selection_bank,
                        component_index,
                        device,
                    )
                )
                assignment = prior_assignment_diagnostics(
                    candidate_prior,
                    selection_bank,
                    device=device,
                    component_mass_threshold=config.component_mass_threshold,
                )
                agreement = prior_assignment_agreement_diagnostics(
                    anchor_prior,
                    candidate_prior,
                    selection_bank,
                    device=device,
                    core_confidence_threshold=config.core_confidence_threshold,
                )
                boundary = prior_boundary_correction_diagnostics(
                    anchor_prior,
                    candidate_prior,
                    selection_bank,
                    component_index,
                    device=device,
                    boundary_log_odds_threshold=(
                        config.boundary_log_odds_threshold
                    ),
                )
                boundary_score_gain = float(boundary["boundary_score_gain"])
                boundary_score_se = float(
                    boundary["boundary_score_standard_error"]
                )
                boundary["boundary_score_lower_bound"] = (
                    boundary_score_gain
                    - float(config.boundary_confidence_z) * boundary_score_se
                )
                regularity = prior_regularity_diagnostics(
                    candidate_prior,
                    minimum_weight=config.minimum_weight,
                    minimum_covariance_eigenvalue=(
                        config.minimum_covariance_eigenvalue
                    ),
                    maximum_covariance_condition=config.maximum_covariance_condition,
                    minimum_interior_shape=config.minimum_interior_shape,
                )
                record.update(
                    {
                        **fit_diagnostics,
                        "status": "ok",
                        "selection_mean_log_score": float(candidate_scores.mean()),
                        "mixture_score_gain": float(mixture_difference.mean()),
                        "mixture_score_standard_error": float(
                            mixture_difference.std(unbiased=True)
                            / math.sqrt(mixture_difference.numel())
                        ),
                        "paired_score_scope": "anchor_component_conditional",
                        "paired_score_gain": component_difference_mean,
                        "paired_score_standard_error": component_difference_se,
                        "paired_score_lower_bound": (
                            component_difference_mean
                            - confidence_z * component_difference_se
                        ),
                        "assignment": assignment,
                        "anchor_assignment_agreement": agreement,
                        "boundary_relevance": boundary,
                        "regularity": regularity,
                        "component_shape": shape_by_component.get(component_index),
                        "fit_and_score_seconds": float(
                            time.perf_counter() - candidate_start
                        ),
                    }
                )
                successful.append(
                    {
                        "prior": candidate_prior,
                        "scores": candidate_scores,
                        "record": record,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "failed",
                        "error": repr(exc),
                        "fit_and_score_seconds": float(
                            time.perf_counter() - candidate_start
                        ),
                    }
                )
            records.append(record)

    eligible: list[dict[str, Any]] = []
    for item in successful:
        record = item["record"]
        assignment = record["assignment"]
        agreement = record["anchor_assignment_agreement"]
        regularity = record["regularity"]
        boundary = record["boundary_relevance"]
        shape = record.get("component_shape")
        reasons: list[str] = []
        if float(anchor_assignment["normalized_marginal_entropy"]) < float(
            config.minimum_anchor_normalized_entropy
        ):
            reasons.append("unsafe_anchor_partition")
        if not bool(regularity["regular"]):
            reasons.append("irregular_parameters")
        if float(record["paired_score_lower_bound"]) < float(config.score_margin):
            reasons.append("insufficient_paired_score_gain")
        if float(record["mixture_score_gain"]) < -1e-6:
            reasons.append("degraded_full_mixture_score")
        if float(boundary["boundary_effective_sample_size"]) < float(
            config.minimum_boundary_effective_size
        ):
            reasons.append("insufficient_boundary_support")
        boundary_score_lower_bound = float(
            boundary["boundary_score_lower_bound"]
        )
        if (
            not math.isfinite(boundary_score_lower_bound)
            or boundary_score_lower_bound
            < float(config.minimum_boundary_score_gain)
        ):
            reasons.append("degraded_boundary_score")
        if float(boundary["boundary_leverage"]) < float(
            config.minimum_boundary_leverage
        ):
            reasons.append("insufficient_boundary_relevance")
        if float(boundary["decision_change_fraction"]) < float(
            config.minimum_decision_change_fraction
        ):
            reasons.append("insufficient_decision_change")
        component_skew = (
            float(shape["crossfit_directional_abs_skew"])
            if shape is not None
            else math.nan
        )
        if not math.isfinite(component_skew) or component_skew < float(
            config.component_skew_threshold
        ):
            reasons.append("insufficient_component_skew")
        component_skew_signal = (
            float(shape["normal_reference_skew_signal"])
            if shape is not None
            else math.nan
        )
        if not math.isfinite(component_skew_signal) or component_skew_signal < float(
            config.component_skew_signal_threshold
        ):
            reasons.append("insufficient_component_skew_signal")
        selection_effective_size = (
            float(shape["effective_sample_size"])
            if shape is not None
            else math.nan
        )
        if not math.isfinite(selection_effective_size) or selection_effective_size < float(
            config.minimum_effective_component_size
        ):
            reasons.append("insufficient_selection_component_size")
        if int(assignment["active_components"]) < minimum_active:
            reasons.append("component_collapse")
        for metric in (
            "normalized_marginal_entropy",
            "normalized_assignment_mi",
            "draw_assignment_stability",
        ):
            if float(assignment[metric]) + float(config.assignment_tolerance) < float(
                anchor_assignment[metric]
            ):
                reasons.append(f"degraded_{metric}")
        if float(agreement["core_fraction"]) < float(config.minimum_core_fraction):
            reasons.append("insufficient_anchor_core")
        core_retention = float(agreement["core_retention"])
        if not math.isfinite(core_retention) or core_retention < float(
            config.minimum_core_retention
        ):
            reasons.append("degraded_anchor_core")
        record["eligible"] = not reasons
        record["rejection_reasons"] = reasons
        if not reasons:
            eligible.append(item)

    selected: dict[str, Any] | None = None
    if eligible:
        best = max(
            eligible,
            key=lambda item: float(item["record"]["selection_mean_log_score"]),
        )
        threshold = float(best["record"]["selection_mean_log_score"]) - float(
            best["record"]["mixture_score_standard_error"]
        )
        one_se = [
            item
            for item in eligible
            if float(item["record"]["selection_mean_log_score"]) >= threshold
        ]
        selected = min(
            one_se,
            key=lambda item: (
                int(item["record"]["complexity_rank"]),
                -float(item["record"]["selection_mean_log_score"]),
                int(item["record"]["component"]),
                str(item["record"]["family"]),
            ),
        )
        selected["record"]["selected"] = True

    selected_prior = anchor_prior if selected is None else selected["prior"]
    selected_family = "student_t" if selected is None else str(
        selected["record"]["family"]
    )
    selected_component = None if selected is None else int(
        selected["record"]["component"]
    )
    diagnostics = {
        "selection_rule": "boundary_targeted_student_t_anchored_gh_enrichment",
        "anchor_family": "student_t",
        "candidate_families": list(config.families),
        "anchor_mean_log_score": float(anchor_scores.mean()),
        "anchor_assignment": anchor_assignment,
        "anchor_component_shapes": component_shapes,
        "familywise_alpha": float(config.familywise_alpha),
        "confidence_z": confidence_z,
        "boundary_confidence_z": float(config.boundary_confidence_z),
        "n_familywise_comparisons": int(attempted),
        "n_successful_candidates": int(len(successful)),
        "n_eligible_candidates": int(len(eligible)),
        "minimum_active_components": int(minimum_active),
        "minimum_boundary_effective_size": float(
            config.minimum_boundary_effective_size
        ),
        "minimum_boundary_score_gain": float(config.minimum_boundary_score_gain),
        "minimum_boundary_leverage": float(config.minimum_boundary_leverage),
        "minimum_decision_change_fraction": float(
            config.minimum_decision_change_fraction
        ),
        "optimization_graph_refinement": bool(config.graph_refinement),
        "optimization_graph_neighbors": int(config.graph_neighbors),
        "optimization_graph_density_one_se_multiplier": float(
            config.graph_density_one_se_multiplier
        ),
        "minimum_optimization_graph_gain": float(
            config.minimum_optimization_graph_gain
        ),
        "minimum_anchor_normalized_entropy": float(
            config.minimum_anchor_normalized_entropy
        ),
        "component_skew_signal_threshold": float(
            config.component_skew_signal_threshold
        ),
        "selected_family": selected_family,
        "selected_component": selected_component,
        "promoted": selected is not None,
        "candidates": records,
    }
    return ComponentwiseEnrichmentResult(
        prior=selected_prior,
        selected_family=selected_family,
        selected_component=selected_component,
        diagnostics=diagnostics,
    )
