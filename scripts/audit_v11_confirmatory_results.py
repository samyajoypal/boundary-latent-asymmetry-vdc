from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_realdata_protocol import _paired_delta


METRICS = ("acc", "ari", "nmi")
REAL_PROTOCOLS = (
    "realdata_v11_tabular_confirmatory",
    "realdata_v11_image_confirmatory",
    "realdata_v11_robustness_confirmatory",
    "realdata_v11_image_ablations",
)
SIMULATION_PROTOCOL = "simulation_protocol_v11_confirmatory"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _cell(row: dict[str, str]) -> str:
    return row.get("protocol_cell_id") or "default"


def _matched_student_t(method: str) -> str | None:
    marker = "trust_component_adaptive"
    return method.replace(marker, "trust_mvt") if marker in method else None


def _holm_adjust(values: list[float | None]) -> list[float | None]:
    indexed = [(index, value) for index, value in enumerate(values) if value is not None]
    adjusted: list[float | None] = [None] * len(values)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(sorted(indexed, key=lambda item: item[1])):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def _add_adjusted_p_values(rows: list[dict[str, Any]]) -> None:
    for metric in METRICS:
        indices = [index for index, row in enumerate(rows) if row.get("metric") == metric]
        values = [_number(rows[index].get("wilcoxon_p")) for index in indices]
        for index, adjusted in zip(indices, _holm_adjust(values)):
            rows[index]["holm_wilcoxon_p"] = adjusted


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _absolute_summary(
    protocol: str, rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(_cell(row), row.get("dataset", ""), row.get("method", ""))].append(row)
    absolute: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for (cell_id, dataset, method), group in sorted(groups.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        item: dict[str, Any] = {
            "protocol": protocol,
            "protocol_cell_id": cell_id,
            "dataset": dataset,
            "method": method,
            "n_ok": len(ok),
            "n_failed": len(group) - len(ok),
            "n_total": len(group),
            "failure_rate": (len(group) - len(ok)) / len(group),
        }
        for metric in METRICS:
            values = [value for row in ok if (value := _number(row.get(metric))) is not None]
            if values:
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        absolute.append(item)
        if item["n_failed"]:
            failures.append(item.copy())
    return absolute, failures


def _error_category(error: str) -> str:
    lowered = error.lower()
    if "no kernel image is available for execution on the device" in lowered:
        return "cuda_architecture_incompatible"
    if "cuda" in lowered or "acceleratorerror" in lowered:
        return "cuda_other"
    if (
        "non-finite" in lowered
        or "not finite" in lowered
        or "finite inputs" in lowered
        or "finite computation graph" in lowered
    ):
        return "numerical_nonfinite"
    if "outside the declared regular parameter region" in lowered:
        return "regularity_rejection"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if not lowered.strip():
        return "missing_error_message"
    return "other"


def _failure_causes(protocol: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in rows:
        if row.get("status") == "ok":
            continue
        key = (
            _cell(row),
            row.get("dataset", ""),
            row.get("method", ""),
            _error_category(row.get("error", "")),
        )
        counts[key] += 1
    return [
        {
            "protocol": protocol,
            "protocol_cell_id": cell_id,
            "dataset": dataset,
            "method": method,
            "error_category": category,
            "count": count,
        }
        for (cell_id, dataset, method, category), count in sorted(counts.items())
    ]


def _primary_real_comparisons(protocol: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    cells = sorted({(_cell(row), row.get("dataset", "")) for row in rows})
    for cell_id, dataset in cells:
        methods = {
            row.get("method", "")
            for row in rows
            if _cell(row) == cell_id and row.get("dataset", "") == dataset
        }
        treatments = sorted(method for method in methods if _matched_student_t(method) in methods)
        for treatment in treatments:
            control = _matched_student_t(treatment)
            assert control is not None
            for metric in METRICS:
                item = _paired_delta(rows, dataset, treatment, control, metric, cell_id)
                item["protocol"] = protocol
                comparisons.append(item)
    _add_adjusted_p_values(comparisons)
    for row in comparisons:
        metric = str(row["metric"])
        low = _number(row.get(f"delta_{metric}_ci95_low"))
        adjusted = _number(row.get("holm_wilcoxon_p"))
        row["strict_win"] = (
            int(row.get("n_pairs", 0)) >= 10
            and low is not None
            and low > 0.0
            and adjusted is not None
            and adjusted < 0.05
        )
    return comparisons


def _simulation_comparisons(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = dict(row)
        item["protocol_cell_id"] = row.get("design_id", "")
        normalized.append(item)
    treatment = "ghvade_trust_component_adaptive"
    control = "ghvade_trust_mvt"
    comparisons = []
    for design_id in sorted({row.get("design_id", "") for row in rows}):
        design_rows = [row for row in normalized if row.get("design_id") == design_id]
        datasets = sorted({row.get("dataset", "") for row in design_rows})
        if len(datasets) != 1:
            raise ValueError(f"Expected one dataset name for design {design_id!r}, found {datasets!r}")
        methods = {row.get("method", "") for row in design_rows}
        if treatment not in methods or control not in methods:
            continue
        for metric in METRICS:
            item = _paired_delta(design_rows, datasets[0], treatment, control, metric, design_id)
            item["protocol"] = SIMULATION_PROTOCOL
            item["design_id"] = design_id
            comparisons.append(item)
    _add_adjusted_p_values(comparisons)
    for row in comparisons:
        metric = str(row["metric"])
        low = _number(row.get(f"delta_{metric}_ci95_low"))
        adjusted = _number(row.get("holm_wilcoxon_p"))
        row["strict_win"] = (
            int(row.get("n_pairs", 0)) >= 20
            and low is not None
            and low > 0.0
            and adjusted is not None
            and adjusted < 0.05
        )
    return comparisons


def _simulation_ranks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(row.get("design_id", ""), row.get("method", ""))].append(row)
    output: list[dict[str, Any]] = []
    for design_id in sorted({key[0] for key in groups}):
        methods = sorted({key[1] for key in groups if key[0] == design_id})
        for metric in METRICS:
            means = []
            for method in methods:
                values = [
                    value
                    for row in groups[(design_id, method)]
                    if (value := _number(row.get(metric))) is not None
                ]
                if values:
                    means.append((method, mean(values)))
            means.sort(key=lambda item: (-item[1], item[0]))
            for rank, (method, value) in enumerate(means, start=1):
                if method == "ghvade_trust_component_adaptive":
                    output.append(
                        {
                            "design_id": design_id,
                            "metric": metric,
                            "primary_mean": value,
                            "primary_rank": rank,
                            "n_methods": len(means),
                            "best_method": means[0][0],
                            "best_mean": means[0][1],
                            "primary_is_best": rank == 1,
                        }
                    )
                    break
    return output


def _find_raw_results(results_root: Path, protocol: str, aggregate_name: str) -> Path:
    return results_root / protocol / aggregate_name / "raw_results.csv"


def _report(
    real_comparisons: list[dict[str, Any]],
    real_failures: list[dict[str, Any]],
    real_failure_causes: list[dict[str, Any]],
    simulation_comparisons: list[dict[str, Any]],
    simulation_ranks: list[dict[str, Any]],
) -> dict[str, Any]:
    real_ari = [row for row in real_comparisons if row.get("metric") == "ari"]
    simulation_ari = [row for row in simulation_comparisons if row.get("metric") == "ari"]
    valid_clean_image = [
        row
        for row in real_ari
        if row.get("protocol") == "realdata_v11_image_confirmatory"
        and int(row.get("n_pairs", 0)) >= 10
    ]
    valid_robustness = [
        row
        for row in real_ari
        if row.get("protocol") == "realdata_v11_robustness_confirmatory"
        and int(row.get("n_pairs", 0)) >= 10
    ]
    simulation_wins = [row for row in simulation_ari if row.get("strict_win")]
    real_wins = [row for row in real_ari if row.get("strict_win")]
    primary_best = [
        row
        for row in simulation_ranks
        if row.get("metric") == "ari" and row.get("primary_is_best")
    ]
    submission_ready = bool(simulation_wins and real_wins and valid_clean_image and valid_robustness)
    failure_counts: dict[str, int] = defaultdict(int)
    for row in real_failure_causes:
        failure_counts[str(row["error_category"])] += int(row["count"])
    return {
        "primary_method": "Student-t-anchored component-adaptive GH-VAE",
        "primary_endpoint": "ARI",
        "simulation_designs_with_strict_primary_vs_student_t_win": len(simulation_wins),
        "simulation_designs_where_primary_has_best_mean_ari_among_all_methods": len(primary_best),
        "real_cells_with_strict_primary_vs_student_t_win": len(real_wins),
        "valid_clean_image_cells_with_at_least_10_pairs": len(valid_clean_image),
        "valid_robustness_cells_with_at_least_10_pairs": len(valid_robustness),
        "real_method_cells_with_failures": len(real_failures),
        "real_failed_runs_by_cause": dict(sorted(failure_counts.items())),
        "submission_evidence_gate_passed": submission_ready,
        "decision": (
            "Evidence gate passed."
            if submission_ready
            else "Evidence gate failed; do not insert a superiority claim into the manuscript."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# V11 Confirmatory Evidence Audit",
        "",
        f"**Decision:** {report['decision']}",
        "",
        "| Submission criterion | Result |",
        "|---|---:|",
        (
            "| Simulation designs with a strict primary-vs-Student-t ARI win | "
            f"{report['simulation_designs_with_strict_primary_vs_student_t_win']} |"
        ),
        (
            "| Simulation designs where the primary has best mean ARI among all methods | "
            f"{report['simulation_designs_where_primary_has_best_mean_ari_among_all_methods']} |"
        ),
        (
            "| Real cells with a strict primary-vs-Student-t ARI win | "
            f"{report['real_cells_with_strict_primary_vs_student_t_win']} |"
        ),
        (
            "| Valid clean-image cells with at least 10 pairs | "
            f"{report['valid_clean_image_cells_with_at_least_10_pairs']} |"
        ),
        (
            "| Valid robustness cells with at least 10 pairs | "
            f"{report['valid_robustness_cells_with_at_least_10_pairs']} |"
        ),
        f"| Real method-cells with failures | {report['real_method_cells_with_failures']} |",
        "",
        "A strict win requires at least 20 paired seeds in simulation or 10 in real data,",
        "a positive 95% paired-difference interval, and Holm-adjusted Wilcoxon p < 0.05.",
        "",
        "## Failed Runs by Cause",
        "",
    ]
    lines.extend(
        f"- `{category}`: {count}"
        for category, count in report["real_failed_runs_by_cause"].items()
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed audit of the frozen v11 confirmatory evidence.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_absolute: list[dict[str, Any]] = []
    real_failures: list[dict[str, Any]] = []
    real_failure_causes: list[dict[str, Any]] = []
    real_comparisons: list[dict[str, Any]] = []
    for protocol in REAL_PROTOCOLS:
        path = _find_raw_results(args.results_root, protocol, "aggregate_final")
        rows = _read_csv(path)
        absolute, failures = _absolute_summary(protocol, rows)
        real_absolute.extend(absolute)
        real_failures.extend(failures)
        real_failure_causes.extend(_failure_causes(protocol, rows))
        real_comparisons.extend(_primary_real_comparisons(protocol, rows))

    simulation_path = _find_raw_results(
        args.results_root, SIMULATION_PROTOCOL, "aggregate_final_core"
    )
    simulation_rows = _read_csv(simulation_path)
    simulation_comparisons = _simulation_comparisons(simulation_rows)
    simulation_ranks = _simulation_ranks(simulation_rows)
    report = _report(
        real_comparisons,
        real_failures,
        real_failure_causes,
        simulation_comparisons,
        simulation_ranks,
    )

    _write_csv(args.output_dir / "real_absolute_results.csv", real_absolute)
    _write_csv(args.output_dir / "real_failure_summary.csv", real_failures)
    _write_csv(args.output_dir / "real_failure_causes.csv", real_failure_causes)
    _write_csv(args.output_dir / "real_primary_vs_student_t.csv", real_comparisons)
    _write_csv(args.output_dir / "simulation_primary_vs_student_t.csv", simulation_comparisons)
    _write_csv(args.output_dir / "simulation_primary_ranks.csv", simulation_ranks)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    _write_markdown(args.output_dir / "report.md", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
