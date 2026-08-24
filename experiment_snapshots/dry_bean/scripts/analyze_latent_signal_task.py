from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from ghmvae.experiments import json_safe, write_json
from ghmvae.priors.fmvmm_adapter import FMVMMPriorFitter
from scripts.run_realdata_benchmark import _metrics, _prior_parameter_summary
from scripts.run_simulation_benchmark import _latent_moment_diagnostics
from scripts.simulation_protocol import ProtocolTask, completed_run_dir, load_tasks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILIES = ("mvn", "mvt", "mgst", "mnig", "mghp")
FAMILY_NAMES = {
    "mvn": "gaussian",
    "mvt": "student_t",
    "mvsn": "skew_normal",
    "mgst": "skew_t",
    "mvvg": "variance_gamma",
    "mvhb": "hyperbolic",
    "mnig": "nig",
    "mghp": "full_gh",
}


def align_candidate_labels(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Align candidate component labels to a reference without using truth labels."""

    reference = np.asarray(reference, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate labels must have matching shapes.")
    reference_values = np.unique(reference)
    candidate_values = np.unique(candidate)
    counts = np.zeros((len(reference_values), len(candidate_values)), dtype=np.int64)
    for row, reference_value in enumerate(reference_values):
        for column, candidate_value in enumerate(candidate_values):
            counts[row, column] = int(
                np.sum((reference == reference_value) & (candidate == candidate_value))
            )
    if counts.size == 0:
        return candidate.copy()
    rows, columns = linear_sum_assignment(counts.max() - counts)
    mapping = {
        int(candidate_values[column]): int(reference_values[row])
        for row, column in zip(rows, columns)
    }
    next_label = int(reference_values.max()) + 1
    aligned = np.empty_like(candidate)
    for value in candidate_values:
        mapped = mapping.get(int(value))
        if mapped is None:
            mapped = next_label
            next_label += 1
        aligned[candidate == value] = mapped
    return aligned


def normalized_affine_residual(reference: np.ndarray, representation: np.ndarray) -> float:
    """Return affine least-squares residual normalized by representation variance."""

    reference = np.asarray(reference, dtype=float)
    representation = np.asarray(representation, dtype=float)
    if reference.ndim != 2 or representation.ndim != 2:
        raise ValueError("reference and representation must be matrices.")
    if reference.shape[0] != representation.shape[0]:
        raise ValueError("reference and representation must have the same row count.")
    design = np.column_stack([reference, np.ones(reference.shape[0])])
    coefficients, _, _, _ = np.linalg.lstsq(design, representation, rcond=None)
    residual = representation - design @ coefficients
    denominator = float(
        np.mean((representation - representation.mean(axis=0, keepdims=True)) ** 2)
    )
    if denominator <= 1e-15:
        return 0.0 if float(np.mean(residual**2)) <= 1e-15 else math.inf
    return float(np.mean(residual**2) / denominator)


def knn_overlap(
    reference: np.ndarray,
    representation: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Mean fraction of oracle neighbours retained by a representation."""

    reference = np.asarray(reference, dtype=float)
    representation = np.asarray(representation, dtype=float)
    if reference.shape[0] != representation.shape[0]:
        raise ValueError("reference and representation must have the same row count.")
    if reference.shape[0] <= 1:
        return math.nan
    k = min(int(n_neighbors), reference.shape[0] - 1)
    reference_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(reference)
    representation_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(representation)
    reference_indices = reference_neighbors.kneighbors(
        reference, return_distance=False
    )[:, 1:]
    representation_indices = representation_neighbors.kneighbors(
        representation, return_distance=False
    )[:, 1:]
    overlaps = [
        len(set(first).intersection(second)) / k
        for first, second in zip(reference_indices, representation_indices)
    ]
    return float(np.mean(overlaps))


def _load_config(protocol_path: Path) -> dict[str, Any]:
    with protocol_path.open("rb") as handle:
        return tomllib.load(handle)


def _output_root(protocol_path: Path, config: dict[str, Any]) -> Path:
    path = Path(config["experiment"]["output_dir"])
    return path if path.is_absolute() else protocol_path.resolve().parents[1] / path


def _task_arguments(config: dict[str, Any], task: ProtocolTask) -> dict[str, Any]:
    return {**dict(config["arguments"]), **dict(task.arguments)}


def _artifact_paths(
    run_dir: Path,
    task: ProtocolTask,
    arguments: dict[str, Any],
) -> tuple[Path, Path]:
    scenarios = arguments.get("scenarios", ["parametric"])
    if len(scenarios) != 1:
        raise ValueError("Latent-signal tasks require exactly one scenario.")
    base = run_dir / "artifacts" / str(scenarios[0]) / f"seed_{task.seed}"
    return base / "latent_truth.npz", base / task.method / "embeddings.npz"


def _split_indices(n_observations: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_observations < 4:
        raise ValueError("At least four observations are required.")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(n_observations)
    n_selection = min(
        n_observations - 2,
        max(2, int(round(float(fraction) * n_observations))),
    )
    return permutation[n_selection:], permutation[:n_selection]


def _fitter(
    family: str,
    n_components: int,
    arguments: dict[str, Any],
    seed: int,
) -> FMVMMPriorFitter:
    fmvmm_path = os.environ.get("FMVMM_PATH") or str(arguments.get("fmvmm_path") or "")
    if family in {"mvn", "mvt"}:
        candidate_combinations = (tuple([family] * int(n_components)),)
        list_of_dist = (family,)
    else:
        candidate_combinations = tuple(
            tuple(family if index == enriched else "mvt" for index in range(n_components))
            for enriched in range(n_components)
        )
        list_of_dist = ("mvt", family)
    return FMVMMPriorFitter(
        n_components=int(n_components),
        fmvmm_path=fmvmm_path or None,
        list_of_dist=list_of_dist,
        candidate_combinations=candidate_combinations,
        max_iter=int(arguments.get("fmvmm_iter", 15)),
        tol=float(arguments.get("fmvmm_tol", 1e-4)),
        initialization=str(arguments.get("fmvmm_initialization", "gmm")),
        n_jobs=int(arguments.get("fmvmm_jobs", 1)),
        gh_mstep_kwargs={
            "nit": int(arguments.get("gh_mstep_nit", 50)),
            "reltol": 1e-4,
            "abstol": 1e-5,
            "silent": True,
        },
        differentiable=True,
        require_differentiable=True,
        standardize=bool(arguments.get("fmvmm_standardize", True)),
        random_seed=int(seed),
        fit_timeout_seconds=float(arguments.get("fmvmm_fit_timeout_seconds", 300.0)),
    )


def _fit_stage_family(
    family: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    selection_indices: np.ndarray,
    n_components: int,
    arguments: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    start = time.perf_counter()
    fit = _fitter(family, n_components, arguments, seed).fit(
        torch.from_numpy(features[train_indices]).float()
    )
    all_features = torch.from_numpy(features).float()
    selection_features = torch.from_numpy(features[selection_indices]).float()
    with torch.no_grad():
        responsibilities = fit.prior.responsibilities(all_features).cpu().numpy()
        selection_responsibilities = (
            fit.prior.responsibilities(selection_features).cpu().numpy()
        )
        selection_log_prob = fit.prior.log_prob(selection_features).cpu().numpy()
    predictions = responsibilities.argmax(axis=1)
    metrics = _metrics(labels, predictions)
    diagnostics = fit.diagnostics
    parameter_summary = _prior_parameter_summary(fit.prior)
    weights = np.asarray(parameter_summary.get("weights", []), dtype=float)
    active_components = int(metrics["n_assigned"])
    validity_reasons: list[str] = []
    if weights.size != int(n_components) or not np.isfinite(weights).all():
        validity_reasons.append("invalid_component_weights")
        min_component_weight = math.nan
        min_expected_component_count = math.nan
    else:
        min_component_weight = float(np.min(weights))
        min_expected_component_count = float(min_component_weight * len(train_indices))
        if min_expected_component_count < 1.0:
            validity_reasons.append("component_has_less_than_one_expected_fit_observation")
    if active_components != int(n_components):
        validity_reasons.append(
            f"active_predicted_components={active_components}!={int(n_components)}"
        )
    row = {
        "family_code": family,
        "family": FAMILY_NAMES[family],
        "status": "ok",
        "heldout_log_score_mean": float(np.mean(selection_log_prob)),
        "heldout_log_score_se": float(
            np.std(selection_log_prob, ddof=1) / math.sqrt(len(selection_log_prob))
        ),
        "fit_seconds": float(time.perf_counter() - start),
        "prior_name": fit.prior.name,
        "best_bic": diagnostics.get("best_bic", math.nan),
        "best_mixture": "+".join(diagnostics.get("best_mixture", [])),
        "fit_attempt": diagnostics.get("fit_attempt", ""),
        "model_structure": (
            "homogeneous" if family in {"mvn", "mvt"} else "one_component_enrichment"
        ),
        "fitted_component_count": int(weights.size),
        "active_predicted_components": active_components,
        "min_component_weight": min_component_weight,
        "min_expected_component_count": min_expected_component_count,
        "fit_valid": not validity_reasons,
        "fit_validity_reason": ";".join(validity_reasons),
        "prior_parameters_json": json.dumps(
            json_safe(parameter_summary), sort_keys=True, separators=(",", ":")
        ),
        **metrics,
    }
    arrays = {
        "predictions": predictions,
        "selection_predictions": selection_responsibilities.argmax(axis=1),
        "selection_responsibilities": selection_responsibilities,
        "selection_log_prob": selection_log_prob,
    }
    return row, arrays


def _paired_standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _add_student_t_comparison(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reference_row: dict[str, Any],
    reference_arrays: dict[str, np.ndarray],
) -> None:
    comparison_valid = bool(
        row.get("fit_valid", False) and reference_row.get("fit_valid", False)
    )
    row["student_t_reference_fit_valid"] = bool(
        reference_row.get("fit_valid", False)
    )
    row["comparison_valid"] = comparison_valid
    if not comparison_valid:
        row["comparison_invalid_reason"] = (
            "candidate_fit_invalid"
            if not row.get("fit_valid", False)
            else "student_t_reference_fit_invalid"
        )
        row.update(
            {
                "delta_log_score_vs_student_t": math.nan,
                "delta_log_score_vs_student_t_se": math.nan,
                "boundary_weighted_delta_log_score_vs_student_t": math.nan,
                "decision_change_fraction_vs_student_t": math.nan,
                "boundary_decision_change_fraction_vs_student_t": math.nan,
                "boundary_observation_fraction": math.nan,
                "delta_acc_vs_student_t": math.nan,
                "delta_ari_vs_student_t": math.nan,
                "delta_nmi_vs_student_t": math.nan,
            }
        )
        return
    score_delta = arrays["selection_log_prob"] - reference_arrays["selection_log_prob"]
    reference_responsibilities = reference_arrays["selection_responsibilities"]
    reference_top = np.max(reference_responsibilities, axis=1)
    boundary_weight = 4.0 * reference_top * (1.0 - reference_top)
    denominator = float(np.sum(boundary_weight))
    aligned = align_candidate_labels(
        reference_arrays["selection_predictions"],
        arrays["selection_predictions"],
    )
    changes = aligned != reference_arrays["selection_predictions"]
    boundary = boundary_weight >= 0.75
    row.update(
        {
            "delta_log_score_vs_student_t": float(np.mean(score_delta)),
            "delta_log_score_vs_student_t_se": _paired_standard_error(score_delta),
            "boundary_weighted_delta_log_score_vs_student_t": (
                float(np.sum(boundary_weight * score_delta) / denominator)
                if denominator > 0.0
                else math.nan
            ),
            "decision_change_fraction_vs_student_t": float(np.mean(changes)),
            "boundary_decision_change_fraction_vs_student_t": (
                float(np.mean(changes[boundary])) if np.any(boundary) else math.nan
            ),
            "boundary_observation_fraction": float(np.mean(boundary)),
            "delta_acc_vs_student_t": float(row["acc"] - reference_row["acc"]),
            "delta_ari_vs_student_t": float(row["ari"] - reference_row["ari"]),
            "delta_nmi_vs_student_t": float(row["nmi"] - reference_row["nmi"]),
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_task(
    protocol_path: Path,
    stage: str,
    tier: str,
    task_id: int,
    families: tuple[str, ...],
) -> Path:
    config = _load_config(protocol_path)
    tasks = load_tasks(protocol_path, stage, tier)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"task_id must be in [0, {len(tasks) - 1}].")
    task = tasks[task_id]
    run_dir = completed_run_dir(protocol_path, task)
    if run_dir is None:
        raise RuntimeError(f"Neural task {task.run_id} has not completed successfully.")
    arguments = _task_arguments(config, task)
    truth_path, embeddings_path = _artifact_paths(run_dir, task, arguments)
    if not truth_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Missing latent-signal artifacts: {truth_path} or {embeddings_path}."
        )

    truth = np.load(truth_path)
    embeddings = np.load(embeddings_path)
    labels = np.asarray(truth["labels"], dtype=np.int64)
    eval_mask = np.asarray(truth["eval_mask"], dtype=bool)
    oracle = np.asarray(truth["latent"], dtype=np.float32)[eval_mask]
    labels = labels[eval_mask]
    representations = {
        "oracle": oracle,
        "pretrained": np.asarray(embeddings["pretrain_mu"], dtype=np.float32)[eval_mask],
        "final": np.asarray(embeddings["mu"], dtype=np.float32)[eval_mask],
    }
    train_indices, selection_indices = _split_indices(
        len(labels),
        float(arguments.get("latent_signal_selection_fraction", 0.30)),
        int(task.seed) + 39001,
    )
    n_components = int(arguments.get("fit_n_components") or arguments["n_components"])
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for stage_name, features in representations.items():
        representation_diagnostics = {
            "affine_residual_from_oracle": normalized_affine_residual(oracle, features),
            "knn_overlap_with_oracle": knn_overlap(oracle, features, n_neighbors=10),
            "affine_residual_from_pretrained": normalized_affine_residual(
                representations["pretrained"], features
            ),
            "knn_overlap_with_pretrained": knn_overlap(
                representations["pretrained"], features, n_neighbors=10
            ),
            **_latent_moment_diagnostics(
                features,
                labels,
                np.ones(len(labels), dtype=bool),
            ),
        }
        fitted: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
        for family_index, family in enumerate(families):
            try:
                row, arrays = _fit_stage_family(
                    family,
                    features,
                    labels,
                    train_indices,
                    selection_indices,
                    n_components,
                    arguments,
                    int(task.seed) + 40001 + 101 * family_index,
                )
                fitted[family] = (row, arrays)
            except Exception as exc:
                row = {
                    "family_code": family,
                    "family": FAMILY_NAMES[family],
                    "status": "failed",
                    "error": repr(exc),
                }
            row.update(
                {
                    "task_id": int(task.task_id),
                    "protocol_stage": stage,
                    "protocol_cell_id": task.cell_id,
                    "seed": int(task.seed),
                    "backbone_method": task.method,
                    "representation_stage": stage_name,
                    "true_family": str(arguments.get("dgp_family", "")),
                    "true_component_families": "+".join(
                        str(value)
                        for value in arguments.get("dgp_component_families", [])
                    ),
                    "skew_orientation": str(arguments.get("dgp_skew_orientation", "")),
                    "skew_strength": float(arguments.get("skew_strength", 0.0)),
                    "affine_gauge": bool(arguments.get("mixture_affine_gauge", False)),
                    "n_observations": int(len(labels)),
                    "n_fit": int(len(train_indices)),
                    "n_selection": int(len(selection_indices)),
                    **representation_diagnostics,
                }
            )
            rows.append(row)

        if "mvt" in fitted:
            reference_row, reference_arrays = fitted["mvt"]
            for family, (row, arrays) in fitted.items():
                _add_student_t_comparison(row, arrays, reference_row, reference_arrays)
        successful = [
            row for row, _ in fitted.values() if bool(row.get("fit_valid", False))
        ]
        if successful:
            best = max(
                successful, key=lambda item: float(item["heldout_log_score_mean"])
            )
            selection = {
                "selection_status": "ok",
                "selected_family": best["family"],
                "selected_score": best["heldout_log_score_mean"],
            }
        else:
            selection = {
                "selection_status": "no_valid_fit",
                "selected_family": None,
                "selected_score": math.nan,
            }
        summaries[stage_name] = {**selection, **representation_diagnostics}

    output_dir = run_dir / "latent_signal"
    _write_csv(output_dir / "diagnostics.csv", rows)
    write_json(output_dir / "summary.json", summaries)
    write_json(
        output_dir / "manifest.json",
        {
            "status": "completed",
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": _sha256(protocol_path),
            "task_id": int(task_id),
            "run_id": task.run_id,
            "families": list(families),
            "truth_sha256": _sha256(truth_path),
            "embeddings_sha256": _sha256(embeddings_path),
            "n_rows": len(rows),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a common mixture catalogue at oracle, pretrained, and final latent stages."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--tier", choices=("all", "core", "stress"), default="all")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_NAMES),
        default=list(DEFAULT_FAMILIES),
        help=(
            "Homogeneous Gaussian/Student-t fits and label-free one-component "
            "Student-t enrichments for each skewed family."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_task(
        args.protocol.resolve(),
        args.stage,
        args.tier,
        args.task_id,
        tuple(args.families),
    )
    print(json.dumps(json_safe({"status": "completed", "output_dir": str(output_dir)})))


if __name__ == "__main__":
    main()
