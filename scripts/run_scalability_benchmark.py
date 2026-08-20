from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ghmvae.experiments import json_safe
from ghmvae.priors import DiagonalGaussianMixtureFitter, PriorCandidate, fit_and_select_prior
from ghmvae.priors.fmvmm_adapter import FMVMMPriorFitter


METHODS = ("gaussian", "student_t", "adaptive")
CELLS = (
    ("n0500", 500, 5, 5),
    ("n1000", 1000, 5, 5),
    ("n3000", 3000, 5, 5),
    ("d02", 1000, 2, 5),
    ("d10", 1000, 10, 5),
    ("k03", 1000, 5, 3),
    ("k10", 1000, 5, 10),
)
FAMILIES = (
    ("gaussian", "mvn", 0),
    ("student_t", "mvt", 1),
    ("skew_normal", "mvsn", 2),
    ("variance_gamma", "mvvg", 3),
    ("hyperbolic", "mvhb", 4),
    ("nig", "mnig", 5),
    ("skew_t", "mgst", 6),
    ("full_gh", "mghp", 7),
)


@dataclass(frozen=True)
class Task:
    task_id: int
    cell_id: str
    n_samples: int
    latent_dim: int
    n_clusters: int
    method: str
    seed: int


def tasks(seeds: list[int]) -> list[Task]:
    output: list[Task] = []
    for cell_id, n_samples, latent_dim, n_clusters in CELLS:
        for method in METHODS:
            for seed in seeds:
                output.append(
                    Task(
                        task_id=len(output),
                        cell_id=cell_id,
                        n_samples=n_samples,
                        latent_dim=latent_dim,
                        n_clusters=n_clusters,
                        method=method,
                        seed=seed,
                    )
                )
    return output


def _latent_bank(task: Task) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(task.seed + 19001)
    directions = rng.normal(size=(task.n_clusters, task.latent_dim))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    means = 4.0 * directions
    labels = rng.integers(0, task.n_clusters, size=task.n_samples)
    points = means[labels] + rng.normal(scale=0.8, size=(task.n_samples, task.latent_dim))
    permutation = rng.permutation(task.n_samples)
    split = max(2, int(round(0.8 * task.n_samples)))
    optimization = torch.tensor(points[permutation[:split]], dtype=torch.float32).unsqueeze(0)
    selection = torch.tensor(points[permutation[split:]], dtype=torch.float32).unsqueeze(0)
    return optimization, selection


def _fmvmm_fitter(task: Task, family: str, args: argparse.Namespace) -> FMVMMPriorFitter:
    return FMVMMPriorFitter(
        n_components=task.n_clusters,
        fmvmm_path=args.fmvmm_path,
        list_of_dist=(family,),
        candidate_combinations=[tuple([family] * task.n_clusters)],
        max_iter=args.fmvmm_iter,
        tol=args.fmvmm_tol,
        initialization="gmm",
        n_jobs=1,
        gh_mstep_kwargs={"nit": args.gh_mstep_nit, "reltol": 1e-4, "abstol": 1e-5, "silent": True},
        differentiable=True,
        require_differentiable=True,
        standardize=True,
        random_seed=task.seed + 19301,
        fit_timeout_seconds=args.fit_timeout_seconds,
    )


def run_task(task: Task, args: argparse.Namespace) -> dict[str, object]:
    optimization, selection = _latent_bank(task)
    fit_points = optimization.reshape(-1, task.latent_dim)
    started = time.perf_counter()
    status = "ok"
    error = ""
    selected_family = task.method
    n_failed_candidates = 0
    diagnostics: dict[str, object] = {}
    try:
        if task.method == "gaussian":
            fitted = DiagonalGaussianMixtureFitter(
                n_components=task.n_clusters,
                n_iter=100,
                seed=task.seed + 19201,
            ).fit(fit_points)
            diagnostics = fitted.diagnostics
        elif task.method == "student_t":
            fitted = _fmvmm_fitter(task, "mvt", args).fit(fit_points)
            diagnostics = fitted.diagnostics
        else:
            candidates = [
                PriorCandidate(
                    name=name,
                    fitter=_fmvmm_fitter(task, family, args),
                    complexity_rank=rank,
                )
                for name, family, rank in FAMILIES
            ]
            fitted = fit_and_select_prior(
                candidates,
                optimization,
                selection,
                selection_rule="shape_gated",
                shape_skew_threshold=0.25,
                shape_score_margin=0.01,
            )
            diagnostics = fitted.diagnostics
            selected_family = fitted.candidate_name
            n_failed_candidates = sum(
                candidate.get("status") == "failed"
                for candidate in diagnostics.get("candidates", [])
            )
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    elapsed = time.perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = float(peak_rss / 1024.0 if sys.platform != "darwin" else peak_rss / (1024.0**2))
    return {
        "task_id": task.task_id,
        "cell_id": task.cell_id,
        "method": task.method,
        "seed": task.seed,
        "n_samples": task.n_samples,
        "latent_dim": task.latent_dim,
        "n_clusters": task.n_clusters,
        "status": status,
        "error": error,
        "selected_family": selected_family,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": peak_rss_mb,
        "n_failed_candidates": n_failed_candidates,
        "diagnostics": diagnostics,
    }


def _write(row: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"task_{int(row['task_id']):03d}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(json_safe(row), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark prior-fitting scalability.")
    parser.add_argument("command", choices=("count", "show", "run"))
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--fmvmm-path", default=os.environ.get("FMVMM_PATH"))
    parser.add_argument("--fmvmm-iter", type=int, default=15)
    parser.add_argument("--fmvmm-tol", type=float, default=1e-4)
    parser.add_argument("--gh-mstep-nit", type=int, default=100)
    parser.add_argument("--fit-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "scalability_v9")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_tasks = tasks(args.seeds)
    if args.command == "count":
        print(len(all_tasks))
        return
    if args.task_id is None or not 0 <= args.task_id < len(all_tasks):
        raise SystemExit(f"--task-id must be in [0, {len(all_tasks) - 1}].")
    task = all_tasks[args.task_id]
    if args.command == "show":
        print(json.dumps(task.__dict__, indent=2, sort_keys=True))
        return
    row = run_task(task, args)
    _write(row, args.output_dir)
    print(json.dumps(json_safe(row), sort_keys=True, allow_nan=False))
    raise SystemExit(0 if row["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
