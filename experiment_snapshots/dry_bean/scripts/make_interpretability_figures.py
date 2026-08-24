from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "conv_ghvade_adaptive"
COMPARATORS = ("conv_vade_diag", "conv_ghvade_mvt", PRIMARY)
DISPLAY = {
    "conv_vade_diag": "Gaussian VaDE",
    "conv_ghvade_mvt": "Student-t VaDE",
    PRIMARY: "Adaptive GH-VaDE",
}
CONDITION_DISPLAY = {
    "clean": "Clean",
    "uniform05": "Random 5%",
    "uniform10": "Random 10%",
    "uniform20": "Random 20%",
    "saltpepper10": "Salt-pepper 10%",
    "semantic10": "Semantic/near 10%",
}
DATASET_DISPLAY = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "usps": "USPS",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _representative_seeds(rows: list[dict[str, str]]) -> dict[str, int]:
    clean_primary = [
        row
        for row in rows
        if row.get("condition") == "clean"
        and row.get("method") == PRIMARY
        and row.get("status") == "ok"
    ]
    output: dict[str, int] = {}
    for dataset in sorted({row["dataset"] for row in clean_primary}):
        group = [row for row in clean_primary if row["dataset"] == dataset]
        middle = median(float(row["ari"]) for row in group)
        selected = min(group, key=lambda row: abs(float(row["ari"]) - middle))
        output[dataset] = int(selected["seed"])
    return output


def _artifact_dir(results_root: Path, dataset: str, method: str, seed: int) -> Path:
    run_id = f"final__{dataset}__{method}__seed{seed:02d}"
    return (
        results_root
        / "realdata_final_v9_clean"
        / run_id
        / "artifacts"
        / dataset
        / f"seed_{seed}"
        / method
    )


def _weighted_ellipse(ax: plt.Axes, points: np.ndarray, weights: np.ndarray, color: object) -> None:
    total = float(weights.sum())
    if total <= 1e-8:
        return
    center = np.sum(points * weights[:, None], axis=0) / total
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance + 1e-6 * np.eye(2))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 4.0 * np.sqrt(np.maximum(eigenvalues, 1e-8))
    ax.add_patch(
        Ellipse(
            center,
            width=float(width),
            height=float(height),
            angle=angle,
            fill=False,
            color=color,
            linewidth=1.0,
            alpha=0.9,
        )
    )


def _latent_figure(results_root: Path, output_dir: Path, dataset: str, seed: int) -> None:
    figure, axes = plt.subplots(1, len(COMPARATORS), figsize=(10.5, 3.2), constrained_layout=True)
    for ax, method in zip(axes, COMPARATORS):
        artifact = _artifact_dir(results_root, dataset, method, seed) / "embeddings.npz"
        if not artifact.exists():
            ax.text(0.5, 0.5, "artifact unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        with np.load(artifact) as data:
            embedding = np.asarray(data["mu"], dtype=float)
            responsibilities = np.asarray(data["responsibilities"], dtype=float)
            labels = np.asarray(data["labels"], dtype=int) if "labels" in data else responsibilities.argmax(axis=1)
            eval_mask = np.asarray(data["eval_mask"], dtype=bool) if "eval_mask" in data else np.ones(len(labels), dtype=bool)
        projection = PCA(n_components=2, random_state=0).fit_transform(embedding)
        clean = projection[eval_mask]
        clean_labels = labels[eval_mask]
        scatter = ax.scatter(
            clean[:, 0],
            clean[:, 1],
            c=clean_labels,
            cmap="tab10",
            s=5,
            alpha=0.38,
            linewidths=0,
        )
        colors = scatter.cmap(np.linspace(0.0, 1.0, responsibilities.shape[1]))
        for cluster, color in enumerate(colors):
            _weighted_ellipse(ax, projection, responsibilities[:, cluster], color)
        ax.set_title(DISPLAY[method], fontsize=10)
        ax.set_xlabel("latent PC1")
        ax.set_ylabel("latent PC2")
        ax.tick_params(labelsize=8)
    figure.suptitle(f"{dataset.replace('_', ' ').title()}, representative seed {seed}", fontsize=11)
    figure.savefig(output_dir / f"latent_contours_{dataset}.pdf", bbox_inches="tight")
    plt.close(figure)


def _parameter_rows(results_root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in sorted(results_root.glob("realdata_final_v9_*/*/artifacts/*/seed_*/conv_ghvade_adaptive/prior_metadata.json")):
        if not path.parents[4].name.startswith("final__"):
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        parts = path.parts
        condition_root = next(part for part in parts if part.startswith("realdata_final_v9_"))
        if condition_root == "realdata_final_v9_ablations":
            continue
        dataset = path.parents[2].name
        seed = int(path.parents[1].name.removeprefix("seed_"))
        summary = metadata.get("parameter_summary", {})
        weights = summary.get("weights", [])
        for component in summary.get("components", []):
            record = {
                "condition": condition_root.removeprefix("realdata_final_v9_"),
                "dataset": dataset,
                "seed": seed,
                "selected_family": metadata.get("selected_family"),
                "component": component.get("component"),
                "family": component.get("family"),
                "weight": weights[int(component.get("component", 0))] if weights else "",
            }
            record.update({key: value for key, value in component.items() if key not in record})
            output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _write_parameter_table(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["dataset"]), str(row["selected_family"]))
        groups.setdefault(key, []).append(row)
    lines = [
        r"\begin{longtable}{lllrrrrrr}",
        r"\caption{Adaptive-prior component parameters in successful final runs; $n_c$ is the number of fitted component instances.}\label{tab:supp-component-parameters}\\",
        r"\toprule",
        r"Condition & Dataset & Family & $n_c$ & Skew norm & $\nu$ & $\lambda$ & $\chi$ & $\psi$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Condition & Dataset & Family & $n_c$ & Skew norm & $\nu$ & $\lambda$ & $\chi$ & $\psi$ \\",
        r"\midrule",
        r"\endhead",
    ]

    def field_mean(group: list[dict[str, object]], names: tuple[str, ...]) -> str:
        values = [
            float(row[name])
            for row in group
            for name in names
            if name in row and row[name] not in (None, "") and math.isfinite(float(row[name]))
        ]
        return "--" if not values else f"{mean(values):.3f}"

    for (condition, dataset, family), group in sorted(groups.items()):
        family_display = family.replace("_", r"\_")
        condition_display = CONDITION_DISPLAY.get(condition, condition).replace("%", r"\%")
        skew = field_mean(group, ("a_norm", "gamma_norm", "skewness_norm"))
        lines.append(
            f"{condition_display} & "
            f"{DATASET_DISPLAY.get(dataset, dataset)} & {family_display} & {len(group)} & "
            f"{skew} & {field_mean(group, ('df',))} & {field_mean(group, ('lmbda',))} & "
            f"{field_mean(group, ('chi',))} & {field_mean(group, ('psi',))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{longtable}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create latent-contour and GH-parameter diagnostics.")
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_analysis" / "raw_results.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "pattern_recognition_submission" / "figures",
    )
    parser.add_argument(
        "--parameter-table",
        type=Path,
        default=ROOT / "pattern_recognition_submission" / "supp_component_parameter_table.tex",
    )
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_analysis" / "interpretability",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = _read(args.raw_results)
    seeds = _representative_seeds(rows)
    for dataset, seed in seeds.items():
        _latent_figure(args.results_root, args.figure_dir, dataset, seed)
    parameters = _parameter_rows(args.results_root)
    _write_csv(args.output_dir / "component_parameters.csv", parameters)
    _write_parameter_table(args.parameter_table, parameters)
    print(json.dumps({"representative_seeds": seeds, "n_component_rows": len(parameters)}, sort_keys=True))


if __name__ == "__main__":
    main()
