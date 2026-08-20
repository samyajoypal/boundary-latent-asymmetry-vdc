from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ghmvae.data.synthetic import (
    make_matched_decoder_mixture,
    make_nonlinear_mixture,
    make_parametric_mixture,
    make_skewed_heavy_mixture,
)
from ghmvae.experiments import json_safe
from scripts.run_realdata_benchmark import (
    DEFAULT_METHODS,
    _fit_ae_kmeans,
    _fit_dec_like,
    _fit_diag_prior_on_features,
    _fit_fmvmm_on_features,
    _fit_gmm,
    _fit_kmeans,
    _fit_mixture_aware_vae,
    _fit_prior_vae,
    _standard_vae_embedding,
    _write_outputs,
)


BASE_SCENARIOS = ("gaussian", "skew", "heavy", "skew_heavy", "contaminated")
OVERLAP_SCENARIOS = tuple(f"overlap_{name}" for name in BASE_SCENARIOS)
MATCHED_SCENARIOS = tuple(f"matched_{name}" for name in BASE_SCENARIOS)
PARAMETRIC_SCENARIOS = ("parametric", "parametric_contaminated")
SCENARIOS = BASE_SCENARIOS + OVERLAP_SCENARIOS + MATCHED_SCENARIOS + PARAMETRIC_SCENARIOS


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-6)


def _standardized_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    standardized = centered / np.clip(scale, 1e-8, None)
    skew = np.mean(standardized**3, axis=0)
    excess_kurtosis = np.mean(standardized**4, axis=0) - 3.0
    return skew, excess_kurtosis


def _latent_moment_diagnostics(
    latent: np.ndarray,
    labels: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, float]:
    clean = np.asarray(eval_mask, dtype=bool) & (np.asarray(labels) >= 0)
    latent_clean = np.asarray(latent, dtype=float)[clean]
    labels_clean = np.asarray(labels)[clean]
    if latent_clean.shape[0] < 4:
        return {
            "latent_global_abs_skew_mean": math.nan,
            "latent_global_excess_kurtosis_mean": math.nan,
            "latent_component_abs_skew_mean": math.nan,
            "latent_component_abs_skew_max": math.nan,
            "latent_component_excess_kurtosis_mean": math.nan,
            "latent_component_excess_kurtosis_max": math.nan,
        }

    global_skew, global_kurtosis = _standardized_moments(latent_clean)
    component_abs_skew: list[float] = []
    component_excess_kurtosis: list[float] = []
    for label in np.unique(labels_clean):
        component_values = latent_clean[labels_clean == label]
        if component_values.shape[0] < 4:
            continue
        skew, excess_kurtosis = _standardized_moments(component_values)
        component_abs_skew.append(float(np.mean(np.abs(skew))))
        component_excess_kurtosis.append(float(np.mean(excess_kurtosis)))
    return {
        "latent_global_abs_skew_mean": float(np.mean(np.abs(global_skew))),
        "latent_global_excess_kurtosis_mean": float(np.mean(global_kurtosis)),
        "latent_component_abs_skew_mean": float(np.mean(component_abs_skew)) if component_abs_skew else math.nan,
        "latent_component_abs_skew_max": float(np.max(component_abs_skew)) if component_abs_skew else math.nan,
        "latent_component_excess_kurtosis_mean": (
            float(np.mean(component_excess_kurtosis)) if component_excess_kurtosis else math.nan
        ),
        "latent_component_excess_kurtosis_max": (
            float(np.max(component_excess_kurtosis)) if component_excess_kurtosis else math.nan
        ),
    }


def _truth_skew_diagnostics(truth: dict[str, object]) -> dict[str, float]:
    vectors: list[np.ndarray] = []
    for component in truth.get("components", []):  # type: ignore[union-attr]
        if not isinstance(component, dict):
            continue
        vector = component.get("beta", component.get("skew_vector"))
        if vector is not None:
            vectors.append(np.asarray(vector, dtype=float))
    if not vectors:
        return {
            "true_skew_vector_norm_mean": 0.0,
            "true_skew_vector_norm_max": 0.0,
        }
    norms = np.asarray([np.linalg.norm(vector) for vector in vectors], dtype=float)
    return {
        "true_skew_vector_norm_mean": float(norms.mean()),
        "true_skew_vector_norm_max": float(norms.max()),
    }


def _append_unlabelled_outliers(
    x: torch.Tensor,
    y: torch.Tensor,
    rate: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    eval_mask = np.ones(int(y.numel()), dtype=bool)
    n_outliers = int(round(max(float(rate), 0.0) * int(y.numel())))
    if n_outliers <= 0:
        return x, y, eval_mask, 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 9001)
    low = x.min(dim=0).values - 2.5
    high = x.max(dim=0).values + 2.5
    outliers = low + (high - low) * torch.rand((n_outliers, x.shape[1]), generator=generator)
    x_all = torch.cat([x, outliers.float()], dim=0)
    y_all = torch.cat([y.long(), -torch.ones(n_outliers, dtype=torch.long)], dim=0)
    eval_mask = np.concatenate([eval_mask, np.zeros(n_outliers, dtype=bool)])
    return _standardize(x_all.float()), y_all, eval_mask, n_outliers


def _append_latent_outliers(latent: torch.Tensor, n_outliers: int, seed: int) -> torch.Tensor:
    if n_outliers <= 0:
        return latent.float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 9109)
    low = latent.min(dim=0).values - 3.0
    high = latent.max(dim=0).values + 3.0
    outliers = low + (high - low) * torch.rand((n_outliers, latent.shape[1]), generator=generator)
    return _standardize(torch.cat([latent.float(), outliers.float()], dim=0))


def _scenario_design(scenario: str, args: argparse.Namespace) -> dict[str, object]:
    is_overlap = scenario.startswith("overlap_")
    base = scenario.removeprefix("overlap_") if is_overlap else scenario
    if base not in BASE_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}.")

    default_center = 2.5 if base == "gaussian" else 2.8
    default_cluster = 0.35 if base == "gaussian" else 0.28
    if is_overlap:
        center_scale = args.overlap_center_scale
        cluster_scale = args.overlap_cluster_scale
        feature_mode = args.overlap_feature_mode
        nuisance_dims = args.overlap_nuisance_dims
        noise = args.overlap_noise if args.overlap_noise is not None else args.noise
    else:
        center_scale = args.center_scale if args.center_scale is not None else default_center
        cluster_scale = args.cluster_scale if args.cluster_scale is not None else default_cluster
        feature_mode = args.feature_mode
        nuisance_dims = args.nuisance_dims
        noise = args.noise

    return {
        "base": base,
        "is_overlap": is_overlap,
        "center_scale": float(center_scale),
        "cluster_scale": float(cluster_scale),
        "feature_mode": str(feature_mode),
        "nuisance_dims": int(nuisance_dims),
        "noise": float(noise),
    }


def _make_matched_scenario(
    scenario: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    base = scenario.removeprefix("matched_")
    if base not in BASE_SCENARIOS:
        raise ValueError(f"Unknown matched scenario {scenario!r}.")
    family = "skew_heavy" if base == "contaminated" else base
    x, labels, latent = make_matched_decoder_mixture(
        n_samples=args.n_samples,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        n_components=args.n_components,
        family=family,
        noise=args.matched_noise,
        center_scale=args.matched_center_scale,
        cluster_scale=args.matched_cluster_scale,
        skew_strength=args.skew_strength,
        tail_df=args.tail_df,
        hidden_width=args.matched_decoder_width,
        seed=seed,
    )
    outlier_rate = args.outlier_rate if base == "contaminated" else 0.0
    x, labels, eval_mask, n_outliers = _append_unlabelled_outliers(
        x,
        labels,
        outlier_rate,
        seed,
    )
    latent = _append_latent_outliers(latent, n_outliers, seed)
    tail_df: float | str = args.tail_df if family in {"heavy", "skew_heavy"} else "inf"
    skew_strength = args.skew_strength if family in {"skew", "skew_heavy"} else 0.0
    meta = {
        "dataset": f"simulation_{scenario}",
        "scenario": scenario,
        "scenario_base": base,
        "is_overlap": False,
        "is_matched": True,
        "n_samples": int(x.shape[0]),
        "n_eval_samples": int(eval_mask.sum()),
        "n_outliers": int(n_outliers),
        "input_dim": int(x.shape[1]),
        "n_classes": int(args.n_components),
        "true_latent_dim": int(latent.shape[1]),
        "feature_mode": "matched_decoder",
        "nuisance_dims": 0,
        "center_scale": float(args.matched_center_scale),
        "cluster_scale": float(args.matched_cluster_scale),
        "noise": float(args.matched_noise),
        "skew_strength": float(skew_strength),
        "tail_df": tail_df,
        "outlier_rate": float(outlier_rate),
    }
    return x.float(), labels.numpy().astype(np.int64), eval_mask, latent.numpy().astype(np.float32), meta


def _make_scenario(
    scenario: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if scenario in PARAMETRIC_SCENARIOS:
        class_probs = tuple(args.class_probs) if args.class_probs is not None else None
        x, labels, latent, truth = make_parametric_mixture(
            n_samples=args.n_samples,
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            n_components=args.n_components,
            family=args.dgp_family,
            noise=args.matched_noise,
            center_scale=args.matched_center_scale,
            cluster_scale=args.matched_cluster_scale,
            skew_strength=args.skew_strength,
            tail_df=args.tail_df,
            alpha_bar=args.dgp_alpha_bar,
            gh_lambda=args.dgp_lambda,
            covariance_ratio=args.covariance_ratio,
            class_probs=class_probs,
            hidden_width=args.matched_decoder_width,
            seed=seed,
            component_families=(
                tuple(args.dgp_component_families)
                if args.dgp_component_families is not None
                else None
            ),
            skew_orientation=args.dgp_skew_orientation,
        )
        outlier_rate = args.outlier_rate if scenario == "parametric_contaminated" else 0.0
        x, labels, eval_mask, n_outliers = _append_unlabelled_outliers(
            x, labels, outlier_rate, seed
        )
        latent = _append_latent_outliers(latent, n_outliers, seed)
        design_id = args.design_id or args.dgp_family
        meta = {
            "dataset": f"simulation_{scenario}_{design_id}",
            "design_id": str(design_id),
            "scenario": scenario,
            "scenario_base": args.dgp_family,
            "is_overlap": False,
            "is_matched": True,
            "is_parametric": True,
            "n_samples": int(x.shape[0]),
            "n_eval_samples": int(eval_mask.sum()),
            "n_outliers": int(n_outliers),
            "input_dim": int(x.shape[1]),
            "n_classes": int(args.n_components),
            "true_latent_dim": int(latent.shape[1]),
            "feature_mode": "matched_decoder",
            "nuisance_dims": 0,
            "center_scale": float(args.matched_center_scale),
            "cluster_scale": float(args.matched_cluster_scale),
            "noise": float(args.matched_noise),
            "skew_strength": float(args.skew_strength),
            "tail_df": float(args.tail_df),
            "outlier_rate": float(outlier_rate),
            "true_family": str(truth["true_family"]),
            "true_component_families": truth["component_families"],
            "dgp_skew_orientation": truth["skew_orientation"],
            "class_probs": truth["class_probs"],
            "covariance_ratio": float(truth["covariance_ratio"]),
            "dgp_alpha_bar": float(truth["alpha_bar"]),
            "dgp_lambda": float(truth["gh_lambda"]),
            "true_components": truth["components"],
            **_truth_skew_diagnostics(truth),
            **_latent_moment_diagnostics(
                latent.numpy().astype(np.float32),
                labels.numpy().astype(np.int64),
                eval_mask,
            ),
        }
        return (
            x.float(),
            labels.numpy().astype(np.int64),
            eval_mask,
            latent.numpy().astype(np.float32),
            meta,
        )
    if scenario.startswith("matched_"):
        return _make_matched_scenario(scenario, args, seed)
    design = _scenario_design(scenario, args)
    base = str(design["base"])
    generator_kwargs = {
        "n_samples": args.n_samples,
        "input_dim": args.input_dim,
        "n_components": args.n_components,
        "noise": float(design["noise"]),
        "center_scale": float(design["center_scale"]),
        "cluster_scale": float(design["cluster_scale"]),
        "feature_mode": str(design["feature_mode"]),
        "nuisance_dims": int(design["nuisance_dims"]),
        "seed": seed,
    }

    if base == "gaussian":
        x, labels, latent = make_nonlinear_mixture(
            **generator_kwargs,
        )
        x, labels, eval_mask, n_outliers = _append_unlabelled_outliers(x, labels, 0.0, seed)
        latent = _append_latent_outliers(latent, n_outliers, seed)
        scenario_params = {"skew_strength": 0.0, "tail_df": "inf", "outlier_rate": 0.0}
    elif base == "skew":
        x, labels, latent = make_skewed_heavy_mixture(
            skew_strength=args.skew_strength,
            tail_df=40.0,
            contamination=0.0,
            **generator_kwargs,
        )
        x, labels, eval_mask, n_outliers = _append_unlabelled_outliers(x, labels, 0.0, seed)
        latent = _append_latent_outliers(latent, n_outliers, seed)
        scenario_params = {"skew_strength": args.skew_strength, "tail_df": 40.0, "outlier_rate": 0.0}
    elif base == "heavy":
        x, labels, latent = make_skewed_heavy_mixture(
            skew_strength=0.0,
            tail_df=args.tail_df,
            contamination=0.0,
            **generator_kwargs,
        )
        x, labels, eval_mask, n_outliers = _append_unlabelled_outliers(x, labels, 0.0, seed)
        latent = _append_latent_outliers(latent, n_outliers, seed)
        scenario_params = {"skew_strength": 0.0, "tail_df": args.tail_df, "outlier_rate": 0.0}
    elif base == "skew_heavy":
        x, labels, latent = make_skewed_heavy_mixture(
            skew_strength=args.skew_strength,
            tail_df=args.tail_df,
            contamination=0.0,
            **generator_kwargs,
        )
        x, labels, eval_mask, n_outliers = _append_unlabelled_outliers(x, labels, 0.0, seed)
        latent = _append_latent_outliers(latent, n_outliers, seed)
        scenario_params = {"skew_strength": args.skew_strength, "tail_df": args.tail_df, "outlier_rate": 0.0}
    elif base == "contaminated":
        x, labels, latent = make_skewed_heavy_mixture(
            skew_strength=args.skew_strength,
            tail_df=args.tail_df,
            contamination=0.0,
            **generator_kwargs,
        )
        x, labels, eval_mask, n_outliers = _append_unlabelled_outliers(x, labels, args.outlier_rate, seed)
        latent = _append_latent_outliers(latent, n_outliers, seed)
        scenario_params = {
            "skew_strength": args.skew_strength,
            "tail_df": args.tail_df,
            "outlier_rate": args.outlier_rate,
        }
    else:
        raise ValueError(f"Unknown scenario {scenario!r}.")

    meta = {
        "dataset": f"simulation_{scenario}",
        "scenario": scenario,
        "scenario_base": base,
        "is_overlap": bool(design["is_overlap"]),
        "n_samples": int(x.shape[0]),
        "n_eval_samples": int(eval_mask.sum()),
        "n_outliers": int(n_outliers),
        "input_dim": int(x.shape[1]),
        "n_classes": int(args.n_components),
        "true_latent_dim": int(latent.shape[1]),
        "feature_mode": str(design["feature_mode"]),
        "nuisance_dims": int(design["nuisance_dims"]),
        "center_scale": float(design["center_scale"]),
        "cluster_scale": float(design["cluster_scale"]),
        "noise": float(design["noise"]),
        **scenario_params,
        **_latent_moment_diagnostics(
            latent.numpy().astype(np.float32),
            labels.numpy().astype(np.int64),
            eval_mask,
        ),
    }
    return x.float(), labels.numpy().astype(np.int64), eval_mask, latent.numpy().astype(np.float32), meta


def _run_one(
    scenario: str,
    seed: int,
    args: argparse.Namespace,
    artifact_root: Path | None = None,
) -> list[dict[str, object]]:
    x, y, eval_mask, z_true, meta = _make_scenario(scenario, args, seed)
    n_clusters = int(args.fit_n_components or meta["n_classes"])
    input_dim = int(meta["input_dim"])
    latent_dim = int(args.latent_dim)
    hidden_dims = tuple(args.hidden_dims)

    rows: list[dict[str, object]] = []
    dataset_hash = hashlib.sha256()
    dataset_hash.update(x.numpy().tobytes())
    dataset_hash.update(y.tobytes())
    dataset_hash.update(eval_mask.tobytes())
    common = {
        "seed": int(seed),
        "dataset_sha256": dataset_hash.hexdigest(),
        **meta,
        "latent_dim": latent_dim,
        "fit_n_components": n_clusters,
    }
    x_np = x.numpy()

    if artifact_root is not None:
        truth_dir = artifact_root / scenario / f"seed_{seed}"
        truth_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            truth_dir / "latent_truth.npz",
            latent=z_true,
            labels=y,
            eval_mask=eval_mask,
        )
        (truth_dir / "latent_truth_metadata.json").write_text(
            json.dumps(json_safe(common), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    for method in args.methods:
        try:
            if method == "raw_kmeans":
                result = _fit_kmeans(method, x_np, y, eval_mask, n_clusters, seed)
            elif method == "raw_gmm":
                result = _fit_gmm(method, x_np, y, eval_mask, n_clusters, seed)
            elif method == "ae_kmeans":
                result = _fit_ae_kmeans(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                )
            elif method in {"dec", "idec"}:
                result = _fit_dec_like(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                    use_reconstruction=(method == "idec"),
                )
            elif method in {"vae_kmeans", "vae_gmm"}:
                _, z_mu, history = _standard_vae_embedding(
                    x, input_dim, latent_dim, hidden_dims, args, seed
                )
                if method == "vae_kmeans":
                    result = _fit_kmeans(method, z_mu, y, eval_mask, n_clusters, seed)
                else:
                    result = _fit_gmm(method, z_mu, y, eval_mask, n_clusters, seed)
                result["pretrain_final_loss"] = float(history[-1].get("loss", math.nan))
            elif method == "latent_kmeans":
                result = _fit_kmeans(method, z_true, y, eval_mask, n_clusters, seed)
            elif method == "latent_gmm":
                result = _fit_gmm(method, z_true, y, eval_mask, n_clusters, seed)
            elif method == "latent_diag_gmm":
                result = _fit_diag_prior_on_features(method, z_true, y, eval_mask, n_clusters, args, seed)
            elif method.startswith("latent_fmvmm_"):
                result = _fit_fmvmm_on_features(method, z_true, y, eval_mask, n_clusters, args, seed)
            elif method in {"vae_diag_gmm"} or method.startswith("vae_fmvmm_"):
                _, z_mu, history = _standard_vae_embedding(
                    x, input_dim, latent_dim, hidden_dims, args, seed
                )
                if method == "vae_diag_gmm":
                    result = _fit_diag_prior_on_features(method, z_mu, y, eval_mask, n_clusters, args, seed)
                else:
                    result = _fit_fmvmm_on_features(method, z_mu, y, eval_mask, n_clusters, args, seed)
                result["pretrain_final_loss"] = float(history[-1].get("loss", math.nan))
            elif method == "diag_gmm_prior" or method.startswith("fmvmm_"):
                result = _fit_prior_vae(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                )
            elif method.startswith("vade_diag") or method.startswith("ghvade_"):
                result = _fit_mixture_aware_vae(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                    artifact_dir=(artifact_root / scenario / f"seed_{seed}" / method) if artifact_root else None,
                )
            else:
                raise ValueError(f"Unknown method {method!r}")
        except Exception as exc:
            result = {"method": method, "status": "failed", "error": repr(exc)}
        row = {**common, **result}
        rows.append(row)
        print(json.dumps(json_safe(row), sort_keys=True, allow_nan=False))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled simulation benchmarks for GH-mixture VAE clustering.")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--design-id", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--fit-n-components", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--feature-mode", type=str, default="direct", choices=["direct", "obfuscated"])
    parser.add_argument("--nuisance-dims", type=int, default=0)
    parser.add_argument("--center-scale", type=float, default=None)
    parser.add_argument("--cluster-scale", type=float, default=None)
    parser.add_argument("--overlap-center-scale", type=float, default=1.45)
    parser.add_argument("--overlap-cluster-scale", type=float, default=0.55)
    parser.add_argument("--overlap-feature-mode", type=str, default="obfuscated", choices=["direct", "obfuscated"])
    parser.add_argument("--overlap-nuisance-dims", type=int, default=8)
    parser.add_argument("--overlap-noise", type=float, default=0.12)
    parser.add_argument("--matched-center-scale", type=float, default=1.8)
    parser.add_argument("--matched-cluster-scale", type=float, default=0.45)
    parser.add_argument("--matched-noise", type=float, default=0.08)
    parser.add_argument("--matched-decoder-width", type=int, default=64)
    parser.add_argument("--skew-strength", type=float, default=1.2)
    parser.add_argument("--tail-df", type=float, default=2.5)
    parser.add_argument("--outlier-rate", type=float, default=0.12)
    parser.add_argument(
        "--dgp-family",
        choices=[
            "gaussian",
            "student_t",
            "skew_normal",
            "skew_t",
            "variance_gamma",
            "hyperbolic",
            "nig",
            "full_gh",
        ],
        default="gaussian",
    )
    parser.add_argument("--dgp-alpha-bar", type=float, default=1.0)
    parser.add_argument("--dgp-lambda", type=float, default=0.7)
    parser.add_argument(
        "--dgp-skew-orientation",
        choices=("radial", "clockwise_neighbor", "counterclockwise_neighbor"),
        default="radial",
    )
    parser.add_argument(
        "--dgp-component-families",
        nargs="+",
        choices=[
            "gaussian",
            "student_t",
            "skew_normal",
            "skew_t",
            "variance_gamma",
            "hyperbolic",
            "nig",
            "full_gh",
        ],
        default=None,
    )
    parser.add_argument("--covariance-ratio", type=float, default=1.0)
    parser.add_argument("--class-probs", nargs="+", type=float, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--ae-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=8e-4)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-warmstart-ae-epochs", type=int, default=0)
    parser.add_argument("--vae-warmstart-lr", type=float, default=1e-3)
    parser.add_argument("--vae-warmstart-logvar", type=float, default=-4.0)
    parser.add_argument("--dec-epochs", type=int, default=20)
    parser.add_argument("--dec-lr", type=float, default=1e-3)
    parser.add_argument("--dec-alpha", type=float, default=1.0)
    parser.add_argument("--idec-recon-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--recon-variance", type=float, default=1.0)
    parser.add_argument("--mixture-pretrain-epochs", type=int, default=40)
    parser.add_argument("--mixture-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--mixture-initial-logvar", type=float, default=-4.0)
    parser.add_argument(
        "--mixture-affine-gauge",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--mixture-epochs", type=int, default=30)
    parser.add_argument("--mixture-cycles", type=int, default=2)
    parser.add_argument("--mixture-lr", type=float, default=8e-4)
    parser.add_argument("--mixture-prior-lr", type=float, default=3e-4)
    parser.add_argument(
        "--mixture-latent-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty on posterior-mean drift from the deterministic pretraining "
            "representation; applied identically to all mixture-aware methods."
        ),
    )
    parser.add_argument(
        "--mixture-posterior-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty on KL(q_pretrain(z|x) || q_current(z|x)); applied "
            "identically to all mixture-aware methods."
        ),
    )
    parser.add_argument(
        "--mixture-posterior-anchor-max-validation-kl",
        type=float,
        default=0.025,
        help=(
            "Maximum mean held-out KL from the pretrained posterior for trust "
            "methods; nonpositive disables the hard checkpoint constraint."
        ),
    )
    parser.add_argument("--mixture-mc-samples", type=int, default=3)
    parser.add_argument("--mixture-refit-mc-samples", type=int, default=3)
    parser.add_argument("--mixture-eval-mc-samples", type=int, default=32)
    parser.add_argument("--mixture-validation-mc-samples", type=int, default=3)
    parser.add_argument("--mixture-selection-fraction", type=float, default=0.1)
    parser.add_argument(
        "--latent-signal-selection-fraction",
        type=float,
        default=0.3,
        help="Held-out fraction used only by the post-hoc latent-signal analysis.",
    )
    parser.add_argument(
        "--mixture-neural-validation-fraction", type=float, default=0.1
    )
    parser.add_argument("--prior-acceptance-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--prior-selection-rule",
        choices=("one_standard_error", "best_score", "shape_gated", "student_t_safe"),
        default="one_standard_error",
    )
    parser.add_argument("--shape-skew-threshold", type=float, default=0.25)
    parser.add_argument("--shape-score-margin", type=float, default=0.01)
    parser.add_argument(
        "--safe-confidence-z",
        type=float,
        default=0.0,
        help="One-sided z cutoff; nonpositive derives a Bonferroni cutoff.",
    )
    parser.add_argument("--safe-familywise-alpha", type=float, default=0.05)
    parser.add_argument("--safe-min-active-fraction", type=float, default=0.8)
    parser.add_argument("--safe-assignment-tolerance", type=float, default=0.05)
    parser.add_argument("--safe-component-mass-threshold", type=float, default=0.01)
    parser.add_argument("--safe-minimum-weight", type=float, default=1e-4)
    parser.add_argument(
        "--safe-minimum-covariance-eigenvalue", type=float, default=1e-5
    )
    parser.add_argument(
        "--safe-maximum-covariance-condition", type=float, default=1e6
    )
    parser.add_argument("--safe-minimum-interior-shape", type=float, default=1e-4)
    parser.add_argument("--safe-core-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--safe-minimum-core-fraction", type=float, default=0.05)
    parser.add_argument("--safe-minimum-core-retention", type=float, default=0.95)
    parser.add_argument(
        "--component-enrichment-families",
        nargs="+",
        choices=("mgst", "mvvg", "mvhb", "mnig", "mghp"),
        default=["mgst", "mghp"],
    )
    parser.add_argument("--component-minimum-effective-size", type=float, default=30.0)
    parser.add_argument(
        "--component-skew-signal-threshold", type=float, default=2.0
    )
    parser.add_argument(
        "--component-minimum-anchor-normalized-entropy", type=float, default=0.8
    )
    parser.add_argument(
        "--component-boundary-log-odds-threshold", type=float, default=2.0
    )
    parser.add_argument(
        "--component-boundary-confidence-z",
        type=float,
        default=1.6448536269514722,
    )
    parser.add_argument(
        "--component-minimum-boundary-effective-size", type=float, default=10.0
    )
    parser.add_argument(
        "--component-minimum-boundary-score-gain", type=float, default=0.0
    )
    parser.add_argument(
        "--component-minimum-boundary-leverage", type=float, default=0.01
    )
    parser.add_argument(
        "--component-minimum-decision-change-fraction", type=float, default=0.005
    )
    parser.add_argument(
        "--component-minimum-graph-assortativity-gain", type=float, default=0.005
    )
    parser.add_argument("--component-graph-neighbors", type=int, default=10)
    parser.add_argument(
        "--component-fit-graph-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--component-graph-density-one-se-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--component-minimum-optimization-graph-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument("--component-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--component-anchor-confidence-threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--adaptive-families",
        nargs="+",
        choices=("mvn", "mvt", "mvsn", "mvvg", "mvhb", "mnig", "mgst", "mghp"),
        default=["mvn", "mvt", "mvsn", "mvvg", "mvhb", "mnig", "mgst", "mghp"],
    )
    parser.add_argument("--mixture-temperature", type=float, default=1.0)
    parser.add_argument("--mixture-mi-weight", type=float, default=0.1)
    parser.add_argument("--mixture-grad-clip", type=float, default=10.0)
    parser.add_argument("--mixture-patience", type=int, default=10)
    parser.add_argument("--diag-gmm-iter", type=int, default=50)
    parser.add_argument("--fmvmm-iter", type=int, default=5)
    parser.add_argument("--fmvmm-tol", type=float, default=1e-3)
    parser.add_argument("--fmvmm-jobs", type=int, default=1)
    parser.add_argument("--gh-mstep-nit", type=int, default=30)
    parser.add_argument("--fmvmm-path", type=str, default=os.environ.get("FMVMM_PATH"))
    parser.add_argument(
        "--fmvmm-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fmvmm-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fmvmm-initialization", choices=["kmeans", "gmm"], default="gmm")
    parser.add_argument("--fmvmm-fit-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "simulation_benchmark")
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.torch_threads))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    rows: list[dict[str, object]] = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            rows.extend(_run_one(scenario, seed, args, artifact_root=output_dir / "artifacts"))
    _write_outputs(rows, output_dir)


if __name__ == "__main__":
    main()
