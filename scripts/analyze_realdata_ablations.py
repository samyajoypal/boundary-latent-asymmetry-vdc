from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

from scripts.analyze_final_realdata import holm_adjust


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "conv_ghvade_adaptive"
METRICS = ("acc", "ari", "nmi")
CLEAN_ABLATIONS = {
    "family_gaussian": "conv_vade_diag",
    "family_student_t": "conv_ghvade_mvt",
    "family_skew_normal": "conv_ghvade_mvsn",
    "family_skew_t": "conv_ghvade_mgst",
    "mutual_information": "conv_ghvade_adaptive_mi",
    "encoder_mlp_clean": "ghvade_adaptive",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row.get("protocol_cell_id", ""), row["method"]), []).append(row)
    output: list[dict[str, object]] = []
    for (cell, method), group in sorted(groups.items()):
        item: dict[str, object] = {"cell": cell, "method": method, "n": len(group)}
        for metric in (*METRICS, "runtime_seconds"):
            values = [float(row[metric]) for row in group if row.get(metric)]
            if values:
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def paired_difference(
    reference: list[dict[str, str]],
    variant: list[dict[str, str]],
    metric: str,
) -> dict[str, object]:
    reference_by_seed = {int(row["seed"]): float(row[metric]) for row in reference if row.get(metric)}
    variant_by_seed = {int(row["seed"]): float(row[metric]) for row in variant if row.get(metric)}
    seeds = sorted(set(reference_by_seed) & set(variant_by_seed))
    deltas = [reference_by_seed[seed] - variant_by_seed[seed] for seed in seeds]
    output: dict[str, object] = {"metric": metric, "n_pairs": len(deltas)}
    if not deltas:
        return output
    output.update(
        {
            "delta_mean": mean(deltas),
            "delta_std": stdev(deltas) if len(deltas) > 1 else 0.0,
            "wins": sum(value > 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
        }
    )
    if len(deltas) > 1:
        if max(deltas) - min(deltas) <= 1e-12:
            output["ci95_low"] = mean(deltas)
            output["ci95_high"] = mean(deltas)
            output["ttest_p"] = 1.0 if deltas[0] == 0 else 0.0
            output["wilcoxon_p"] = (
                1.0 if deltas[0] == 0 else float(stats.wilcoxon(deltas).pvalue)
            )
        else:
            sem = stats.sem(deltas)
            low, high = stats.t.interval(0.95, len(deltas) - 1, loc=mean(deltas), scale=sem)
            output["ci95_low"] = float(low)
            output["ci95_high"] = float(high)
            output["ttest_p"] = float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)
            try:
                output["wilcoxon_p"] = float(stats.wilcoxon(deltas).pvalue)
            except ValueError:
                output["wilcoxon_p"] = math.nan
    return output


def _comparison_rows(
    compact_rows: list[dict[str, str]],
    robust_reference: list[dict[str, str]],
    clean_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    robust_reference = [
        row
        for row in robust_reference
        if row["dataset"] == "mnist" and row["method"] == PRIMARY and int(row["seed"]) < 10
    ]
    for cell in sorted({row.get("protocol_cell_id", "") for row in compact_rows}):
        variants = [row for row in compact_rows if row.get("protocol_cell_id", "") == cell]
        for metric in METRICS:
            item = paired_difference(robust_reference, variants, metric)
            item.update(
                {
                    "condition": "uniform_10",
                    "dataset": "mnist",
                    "ablation": cell,
                    "reference": PRIMARY,
                    "variant": variants[0]["method"] if variants else "",
                }
            )
            output.append(item)

    for dataset in sorted({row["dataset"] for row in clean_rows}):
        clean_reference = [
            row for row in clean_rows if row["dataset"] == dataset and row["method"] == PRIMARY
        ]
        for name, method in CLEAN_ABLATIONS.items():
            variants = [
                row for row in clean_rows if row["dataset"] == dataset and row["method"] == method
            ]
            if not variants:
                continue
            for metric in METRICS:
                item = paired_difference(clean_reference, variants, metric)
                item.update(
                    {
                        "condition": "clean",
                        "dataset": dataset,
                        "ablation": name,
                        "reference": PRIMARY,
                        "variant": method,
                    }
                )
                output.append(item)

    for field in ("ttest_p", "wilcoxon_p"):
        p_values = [
            float(row[field]) if field in row and math.isfinite(float(row[field])) else None
            for row in output
        ]
        for row, adjusted in zip(output, holm_adjust(p_values)):
            row[f"holm_{field}"] = adjusted
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the paired v9 real-data ablations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_ablations" / "aggregate_final" / "raw_results.csv",
    )
    parser.add_argument(
        "--robust-reference",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_uniform10" / "aggregate_final" / "raw_results.csv",
    )
    parser.add_argument(
        "--clean-input",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_analysis" / "raw_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_ablations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compact_rows = _read(args.input)
    robust_reference = _read(args.robust_reference)
    clean_rows = [row for row in _read(args.clean_input) if row.get("condition") == "clean"]
    summary = _summary(compact_rows)
    comparisons = _comparison_rows(compact_rows, robust_reference, clean_rows)
    _write(args.output_dir / "ablation_summary.csv", summary)
    _write(args.output_dir / "ablation_paired_comparisons.csv", comparisons)
    print(
        json.dumps(
            {
                "n_compact_rows": len(compact_rows),
                "n_groups": len(summary),
                "n_paired_comparisons": len(comparisons),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
