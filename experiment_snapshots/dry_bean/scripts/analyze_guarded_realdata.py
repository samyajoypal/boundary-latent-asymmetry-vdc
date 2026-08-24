from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_realdata_protocol import _paired_delta


PRIMARY = "conv_ghvade_guarded_adaptive"
BASELINES = {
    "raw_kmeans",
    "raw_gmm",
    "dec",
    "idec",
    "vae_kmeans",
    "vae_gmm",
    "vade_diag",
    "conv_vade_diag",
    "conv_vade_diag_mi",
}
CONDITIONS = {
    "clean": (
        "realdata_hard_image_clean_v1",
        "realdata_hard_image_guarded_clean_v1",
    ),
    "semantic10": (
        "realdata_hard_image_semantic10_v1",
        "realdata_hard_image_guarded_semantic10_v1",
    ),
    "saltpepper10": (
        "realdata_hard_image_saltpepper10_v1",
        "realdata_hard_image_guarded_saltpepper10_v1",
    ),
}


def _read_rows(result_root: Path) -> list[dict[str, str]]:
    path = result_root / "aggregate_pilot" / "raw_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_summary(rows: list[dict[str, str]], metric: str) -> dict[str, tuple[float, float, int]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok" or not row.get(metric):
            continue
        grouped.setdefault(row["method"], []).append(float(row[metric]))
    return {
        method: (mean(values), pstdev(values) if len(values) > 1 else 0.0, len(values))
        for method, values in grouped.items()
    }


def _compact_delta(comparison: dict[str, object], metric: str) -> dict[str, object]:
    return {
        "n_pairs": comparison["n_pairs"],
        "delta_mean": comparison.get(f"delta_{metric}_mean"),
        "delta_std": comparison.get(f"delta_{metric}_std"),
        "ci95_low": comparison.get(f"delta_{metric}_ci95_low"),
        "ci95_high": comparison.get(f"delta_{metric}_ci95_high"),
        "wins": comparison.get("wins"),
        "ties": comparison.get("ties"),
        "losses": comparison.get("losses"),
        "ttest_p": comparison.get("ttest_p"),
        "wilcoxon_p": comparison.get("wilcoxon_p"),
    }


def analyze(results_root: Path) -> dict[str, object]:
    records: list[dict[str, object]] = []
    guard_diagnostics: list[dict[str, object]] = []
    for condition, (original_name, guarded_name) in CONDITIONS.items():
        original_rows = _read_rows(results_root / original_name)
        guarded_rows = _read_rows(results_root / guarded_name)
        rows = original_rows + guarded_rows

        primary_rows = [row for row in guarded_rows if row.get("method") == PRIMARY]
        guard_diagnostics.append(
            {
                "condition": condition,
                "n_completed": len(primary_rows),
                "n_guard_triggered": sum(row.get("guard_triggered", "").lower() == "true" for row in primary_rows),
                "selected_families": dict(Counter(row.get("selected_family", "") for row in primary_rows)),
                "initial_families": dict(
                    Counter(row.get("guard_initial_selected_family", "") for row in primary_rows)
                ),
            }
        )

        for dataset in ("cifar10", "cifar100_20"):
            dataset_rows = [row for row in rows if row.get("dataset") == dataset and row.get("status") == "ok"]
            for metric in ("acc", "ari", "nmi"):
                summaries = _metric_summary(dataset_rows, metric)
                baseline_methods = [method for method in summaries if method in BASELINES]
                proposed_methods = [method for method in summaries if method not in BASELINES]
                best_baseline = max(baseline_methods, key=lambda method: summaries[method][0])
                best_proposed = max(proposed_methods, key=lambda method: summaries[method][0])
                primary_mean, primary_std, primary_n = summaries[PRIMARY]
                baseline_mean, baseline_std, baseline_n = summaries[best_baseline]
                proposed_mean, proposed_std, proposed_n = summaries[best_proposed]
                record: dict[str, object] = {
                    "condition": condition,
                    "dataset": dataset,
                    "metric": metric,
                    "primary_method": PRIMARY,
                    "primary_mean": primary_mean,
                    "primary_std": primary_std,
                    "primary_n": primary_n,
                    "best_baseline": best_baseline,
                    "best_baseline_mean": baseline_mean,
                    "best_baseline_std": baseline_std,
                    "best_baseline_n": baseline_n,
                    "best_proposed": best_proposed,
                    "best_proposed_mean": proposed_mean,
                    "best_proposed_std": proposed_std,
                    "best_proposed_n": proposed_n,
                    "primary_vs_best_baseline": _compact_delta(
                        _paired_delta(dataset_rows, dataset, PRIMARY, best_baseline, metric), metric
                    ),
                }
                if "conv_ghvade_mvt" in summaries:
                    record["primary_vs_fixed_mvt"] = _compact_delta(
                        _paired_delta(dataset_rows, dataset, PRIMARY, "conv_ghvade_mvt", metric), metric
                    )
                records.append(record)
    return {
        "primary_method": PRIMARY,
        "records": records,
        "guard_diagnostics": guard_diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze guarded real-data pilots against frozen baselines.")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "realdata_hard_image_guarded_analysis_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
