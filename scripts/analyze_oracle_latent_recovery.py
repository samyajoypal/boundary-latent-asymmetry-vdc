from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from ghmvae.data import unlabeled_optimization_selection_split
from ghmvae.data.synthetic import make_parametric_mixture
from ghmvae.experiments import json_safe, write_json
from ghmvae.priors import PriorCandidate, fit_and_select_prior
from ghmvae.priors.fmvmm_adapter import FMVMMPriorFitter
from scripts.run_realdata_benchmark import ADAPTIVE_FAMILY_ORDER, _metrics
from scripts.simulation_protocol import DEFAULT_PROTOCOL, load_tasks


FAMILY_METHODS = {
    "mvn": "oracle_fmvmm_mvn",
    "mvt": "oracle_fmvmm_mvt",
    "mvsn": "oracle_fmvmm_mvsn",
    "mvvg": "oracle_fmvmm_mvvg",
    "mvhb": "oracle_fmvmm_mvhb",
    "mnig": "oracle_fmvmm_mnig",
    "mgst": "oracle_fmvmm_mgst",
    "mghp": "oracle_fmvmm_mghp",
}


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    true_index = {label: idx for idx, label in enumerate(true_labels)}
    pred_index = {label: idx for idx, label in enumerate(pred_labels)}
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[true_index[true], pred_index[pred]] += 1
    return matrix, true_labels, pred_labels


def matched_recovery_summary(
    z: np.ndarray,
    labels: np.ndarray,
    responsibilities: np.ndarray,
) -> dict[str, float | int]:
    """Summarize label-matched oracle-latent centroid and weight recovery."""

    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels)
    responsibilities = np.asarray(responsibilities, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must have shape (n_samples, latent_dim).")
    if responsibilities.ndim != 2 or responsibilities.shape[0] != z.shape[0]:
        raise ValueError("responsibilities must have shape (n_samples, n_components).")
    if labels.shape[0] != z.shape[0]:
        raise ValueError("labels and z must have the same number of samples.")

    pred = responsibilities.argmax(axis=1)
    matrix, true_labels, pred_labels = _confusion(labels, pred)
    if matrix.size == 0:
        return {
            "matched_components": 0,
            "centroid_error_mean": math.nan,
            "centroid_error_max": math.nan,
            "weight_mae": math.nan,
        }

    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    centroid_errors: list[float] = []
    weight_errors: list[float] = []
    for row, col in zip(row_ind, col_ind):
        true_label = true_labels[row]
        pred_label = pred_labels[col]
        resp_col = responsibilities[:, int(pred_label)]
        resp_sum = float(resp_col.sum())
        if resp_sum <= 0.0:
            continue
        true_mask = labels == true_label
        true_center = z[true_mask].mean(axis=0)
        fitted_center = (resp_col[:, None] * z).sum(axis=0) / resp_sum
        centroid_errors.append(float(np.linalg.norm(fitted_center - true_center)))
        weight_errors.append(abs(float(resp_col.mean()) - float(true_mask.mean())))

    return {
        "matched_components": int(len(centroid_errors)),
        "centroid_error_mean": mean(centroid_errors) if centroid_errors else math.nan,
        "centroid_error_max": max(centroid_errors) if centroid_errors else math.nan,
        "weight_mae": mean(weight_errors) if weight_errors else math.nan,
    }


def _load_protocol_arguments(protocol_path: Path) -> dict[str, Any]:
    with protocol_path.open("rb") as handle:
        config = tomllib.load(handle)
    return dict(config["arguments"])


def _cell_tasks(protocol_path: Path, stage: str, tier: str) -> list[tuple[str, str, int, dict[str, Any]]]:
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, str, int, dict[str, Any]]] = []
    for task in load_tasks(protocol_path, stage, tier):
        key = (task.cell_id, task.seed)
        if key in seen:
            continue
        seen.add(key)
        out.append((task.cell_id, task.tier, task.seed, dict(task.arguments)))
    return out


def _make_latents(base_args: dict[str, Any], overrides: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    args = {**base_args, **overrides}
    class_probs = tuple(args["class_probs"]) if args.get("class_probs") is not None else None
    _, labels, latent, truth = make_parametric_mixture(
        n_samples=int(args["n_samples"]),
        input_dim=int(args["input_dim"]),
        latent_dim=int(args["latent_dim"]),
        n_components=int(args["n_components"]),
        family=str(args["dgp_family"]),
        noise=float(args["matched_noise"]),
        center_scale=float(args["matched_center_scale"]),
        cluster_scale=float(args["matched_cluster_scale"]),
        skew_strength=float(args["skew_strength"]),
        tail_df=float(args["tail_df"]),
        alpha_bar=float(args["dgp_alpha_bar"]),
        gh_lambda=float(args["dgp_lambda"]),
        covariance_ratio=float(args["covariance_ratio"]),
        class_probs=class_probs,
        hidden_width=int(args["matched_decoder_width"]),
        seed=int(seed),
    )
    return latent.numpy().astype(np.float32), labels.numpy().astype(np.int64), truth


def _fitter(
    family: str,
    n_components: int,
    args: argparse.Namespace,
    seed: int,
) -> FMVMMPriorFitter:
    return FMVMMPriorFitter(
        n_components=int(n_components),
        fmvmm_path=args.fmvmm_path,
        list_of_dist=(family,),
        candidate_combinations=(tuple([family] * int(n_components)),),
        max_iter=int(args.fmvmm_iter),
        tol=float(args.fmvmm_tol),
        initialization=str(args.fmvmm_initialization),
        n_jobs=int(args.fmvmm_jobs),
        gh_mstep_kwargs={
            "nit": int(args.gh_mstep_nit),
            "reltol": 1e-4,
            "abstol": 1e-5,
            "silent": True,
        },
        differentiable=True,
        require_differentiable=True,
        standardize=bool(args.fmvmm_standardize),
        random_seed=int(seed) + 8101,
        fit_timeout_seconds=args.fit_timeout_seconds,
    )


def _adaptive_candidates(n_components: int, args: argparse.Namespace, seed: int) -> list[PriorCandidate]:
    candidates: list[PriorCandidate] = []
    for family in args.families:
        name, rank = ADAPTIVE_FAMILY_ORDER[family]
        candidates.append(
            PriorCandidate(
                name=name,
                fitter=_fitter(family, n_components, args, seed),
                complexity_rank=rank,
            )
        )
    return candidates


def _evaluate_prior(
    method: str,
    prior: Any,
    z: np.ndarray,
    labels: np.ndarray,
    selected_family: str,
    truth: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    z_t = torch.from_numpy(z).float()
    responsibilities = prior.responsibilities(z_t).detach().cpu().numpy()
    pred = responsibilities.argmax(axis=1)
    recovery = matched_recovery_summary(z, labels, responsibilities)
    true_family = str(truth["true_family"])
    return {
        "method": method,
        "status": "ok",
        "true_family": true_family,
        "selected_family": selected_family,
        "family_match": selected_family == true_family,
        "best_bic": diagnostics.get("best_bic", math.nan),
        "fit_attempt": diagnostics.get("fit_attempt", ""),
        "n_successful_candidates": diagnostics.get("n_successful_candidates", ""),
        **_metrics(labels, pred),
        **recovery,
    }


def _run_fixed_family(
    family: str,
    z: np.ndarray,
    labels: np.ndarray,
    truth: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    result = _fitter(family, int(args.n_components), args, seed).fit(torch.from_numpy(z).float())
    selected_family = ADAPTIVE_FAMILY_ORDER[family][0]
    return _evaluate_prior(
        FAMILY_METHODS[family],
        result.prior,
        z,
        labels,
        selected_family,
        truth,
        result.diagnostics,
    )


def _run_adaptive(
    z: np.ndarray,
    labels: np.ndarray,
    truth: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    z_t = torch.from_numpy(z).float()
    train_z, selection_z, _, _ = unlabeled_optimization_selection_split(
        z_t,
        selection_fraction=float(args.selection_fraction),
        seed=int(seed) + 8209,
    )
    selection = fit_and_select_prior(
        _adaptive_candidates(int(args.n_components), args, seed),
        train_z.unsqueeze(0),
        selection_z.unsqueeze(0),
        device="cpu",
        selection_rule=str(args.selection_rule),
        shape_skew_threshold=float(args.shape_skew_threshold),
        shape_score_margin=float(args.shape_score_margin),
    )
    return _evaluate_prior(
        "oracle_adaptive",
        selection.prior,
        z,
        labels,
        selection.candidate_name,
        truth,
        selection.diagnostics,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row if not isinstance(row[key], (dict, list, tuple))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_method: dict[str, dict[str, Any]] = {}
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row.get("method") == method]
        ok_method = [row for row in method_rows if row.get("status") == "ok"]
        by_method[method] = {
            "n_total": len(method_rows),
            "n_failed": len(method_rows) - len(ok_method),
            "failure_rate": (len(method_rows) - len(ok_method)) / max(len(method_rows), 1),
            "ari_mean": mean(float(row["ari"]) for row in ok_method) if ok_method else math.nan,
            "acc_mean": mean(float(row["acc"]) for row in ok_method) if ok_method else math.nan,
            "centroid_error_mean": (
                mean(float(row["centroid_error_mean"]) for row in ok_method)
                if ok_method
                else math.nan
            ),
            "weight_mae_mean": mean(float(row["weight_mae"]) for row in ok_method) if ok_method else math.nan,
        }
    adaptive = [row for row in ok_rows if row.get("method") == "oracle_adaptive"]
    by_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adaptive:
        grouped[str(row["true_family"])].append(row)
    for family, group in sorted(grouped.items()):
        by_family[family] = {
            "n": len(group),
            "ari_mean": mean(float(row["ari"]) for row in group),
            "family_match_rate": mean(1.0 if row["family_match"] else 0.0 for row in group),
        }
    return {
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "by_method": json_safe(by_method),
        "oracle_adaptive_family_match_rate": (
            mean(1.0 if row["family_match"] else 0.0 for row in adaptive) if adaptive else math.nan
        ),
        "oracle_adaptive_by_true_family": json_safe(by_family),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-latent family and recovery diagnostics for the frozen protocol.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--tier", choices=("all", "core", "stress"), default="core")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "oracle_latent_recovery")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument(
        "--selection-rule",
        choices=("one_standard_error", "best_score", "shape_gated"),
        default="one_standard_error",
    )
    parser.add_argument("--shape-skew-threshold", type=float, default=0.25)
    parser.add_argument("--shape-score-margin", type=float, default=0.01)
    parser.add_argument("--fmvmm-path", type=str, default=os.environ.get("FMVMM_PATH"))
    parser.add_argument("--fmvmm-iter", type=int, default=15)
    parser.add_argument("--fmvmm-tol", type=float, default=1e-4)
    parser.add_argument("--fmvmm-jobs", type=int, default=1)
    parser.add_argument("--gh-mstep-nit", type=int, default=100)
    parser.add_argument("--fmvmm-standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fmvmm-initialization", choices=("kmeans", "gmm"), default="gmm")
    parser.add_argument("--fit-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_METHODS),
        default=list(FAMILY_METHODS),
    )
    parser.add_argument("--adaptive-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    base_args = _load_protocol_arguments(protocol_path)
    if args.n_samples is not None:
        base_args["n_samples"] = int(args.n_samples)
    out_dir = args.output_dir / (args.run_id or f"{args.stage}_{args.tier}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    tasks = _cell_tasks(protocol_path, args.stage, args.tier)
    if args.max_tasks is not None:
        tasks = tasks[: int(args.max_tasks)]
    jsonl_path = out_dir / "oracle_latent_recovery.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for cell_id, tier, seed, overrides in tasks:
            z, labels, truth = _make_latents(base_args, overrides, seed)
            common = {
                "cell_id": cell_id,
                "tier": tier,
                "seed": int(seed),
                "n_samples": int(z.shape[0]),
                "latent_dim": int(z.shape[1]),
            }
            methods = ["adaptive"] if args.adaptive_only else list(args.families) + ["adaptive"]
            for method in methods:
                try:
                    if method == "adaptive":
                        result = _run_adaptive(z, labels, truth, args, seed)
                    else:
                        result = _run_fixed_family(method, z, labels, truth, args, seed)
                except Exception as exc:
                    result = {
                        "method": "oracle_adaptive" if method == "adaptive" else FAMILY_METHODS[method],
                        "status": "failed",
                        "true_family": truth["true_family"],
                        "error": repr(exc),
                    }
                row = {**common, **result}
                safe_row = json_safe(row)
                rows.append(row)
                jsonl.write(json.dumps(safe_row, sort_keys=True, allow_nan=False) + "\n")
                jsonl.flush()
                print(json.dumps(safe_row, sort_keys=True, allow_nan=False), flush=True)

    _write_csv(out_dir / "oracle_latent_recovery.csv", rows)
    write_json(out_dir / "oracle_latent_recovery_summary.json", _summary(rows))
    write_json(
        out_dir / "run_manifest.json",
        {
            "protocol": str(protocol_path),
            "stage": args.stage,
            "tier": args.tier,
            "families": list(args.families),
            "adaptive_only": bool(args.adaptive_only),
            "selection_rule": str(args.selection_rule),
            "shape_skew_threshold": float(args.shape_skew_threshold),
            "shape_score_margin": float(args.shape_score_margin),
            "n_tasks": len(tasks),
            "n_rows": len(rows),
        },
    )


if __name__ == "__main__":
    main()
