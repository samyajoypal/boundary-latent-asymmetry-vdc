from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


METHOD_LABELS = {
    "raw_kmeans": "K-means",
    "raw_gmm": "Gaussian mixture",
    "dec": "DEC",
    "idec": "IDEC",
    "vade_diag": "Gaussian VaDE",
    "ghvade_mvt": "Student-t GH-VaDE",
    "ghvade_adaptive": "Adaptive GH-VaDE",
}
METHOD_ORDER = tuple(METHOD_LABELS)
METRICS = ("acc", "ari", "nmi")


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _read_result_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("final__dry_bean__*__seed*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                rows.append({"run_dir": path.parent.name, **item})
    return rows


def _require_pilot_decision(
    path: Path,
    *,
    analysis_mode: str,
    amendment_path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Blinded pilot gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if analysis_mode == "confirmatory" and gate.get("gate_passed") is not True:
        raise RuntimeError(
            "Confirmatory analysis is locked because the blinded pilot gate did not pass."
        )
    if analysis_mode == "descriptive_fallback":
        if gate.get("gate_passed") is not False:
            raise RuntimeError(
                "Descriptive fallback mode requires a recorded failed pilot gate."
            )
        if not amendment_path.exists():
            raise FileNotFoundError(
                f"The pre-label protocol amendment is missing: {amendment_path}"
            )
    return gate


def _mean_sd(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    return mean(finite), stdev(finite) if len(finite) > 1 else math.nan


def _confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size < 2:
        return math.nan, math.nan
    standard_error = float(finite.std(ddof=1) / math.sqrt(finite.size))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=finite.size - 1))
    center = float(finite.mean())
    return center - critical * standard_error, center + critical * standard_error


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    finite = sorted(
        ((name, value) for name, value in p_values.items() if math.isfinite(value)),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {name: math.nan for name in p_values}
    running = 0.0
    count = len(finite)
    for rank, (name, value) in enumerate(finite):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = [
            row for row in rows if row.get("method") == method and row.get("status") == "ok"
        ]
        summary: dict[str, Any] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_ok": len(method_rows),
            "n_expected": 20,
        }
        for metric in METRICS:
            values = [_number(row.get(metric)) for row in method_rows]
            metric_mean, metric_sd = _mean_sd(values)
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_sd"] = metric_sd
        runtime_mean, runtime_sd = _mean_sd(
            [_number(row.get("runtime_seconds")) for row in method_rows]
        )
        summary["runtime_seconds_mean"] = runtime_mean
        summary["runtime_seconds_sd"] = runtime_sd
        summaries.append(summary)
    return summaries


def _paired_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("method")), int(row.get("seed"))): row
        for row in rows
        if row.get("status") == "ok" and row.get("method") in METHOD_ORDER
    }
    contrasts: list[dict[str, Any]] = []
    t_p_values: dict[str, float] = {}
    for comparator in ("ghvade_mvt", "vade_diag"):
        for metric in METRICS:
            differences: list[float] = []
            seeds: list[int] = []
            for seed in range(4200, 4220):
                adaptive = by_key.get(("ghvade_adaptive", seed))
                reference = by_key.get((comparator, seed))
                if adaptive is None or reference is None:
                    continue
                difference = _number(adaptive.get(metric)) - _number(reference.get(metric))
                if math.isfinite(difference):
                    differences.append(difference)
                    seeds.append(seed)
            difference_mean, difference_sd = _mean_sd(differences)
            ci_low, ci_high = _confidence_interval(differences)
            if len(differences) >= 2:
                test = stats.ttest_1samp(differences, popmean=0.0, nan_policy="omit")
                p_value = float(test.pvalue)
            else:
                p_value = math.nan
            key = f"{comparator}:{metric}"
            if comparator == "ghvade_mvt":
                t_p_values[metric] = p_value
            contrasts.append(
                {
                    "contrast": f"ghvade_adaptive-minus-{comparator}",
                    "comparator": comparator,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "seeds": ";".join(str(seed) for seed in seeds),
                    "mean_difference": difference_mean,
                    "sd_difference": difference_sd,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "two_sided_p": p_value,
                    "holm_p": math.nan,
                    "adaptive_wins": sum(value > 0.0 for value in differences),
                    "ties": sum(value == 0.0 for value in differences),
                    "reference_wins": sum(value < 0.0 for value in differences),
                }
            )
    adjusted = _holm_adjust(t_p_values)
    for row in contrasts:
        if row["comparator"] == "ghvade_mvt":
            row["holm_p"] = adjusted[str(row["metric"])]
    return contrasts


def _selection_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adaptive = [
        row for row in rows if row.get("method") == "ghvade_adaptive" and row.get("status") == "ok"
    ]
    counts = Counter(str(row.get("selected_family", "unknown")) for row in adaptive)
    return [
        {
            "selected_family": family,
            "count": count,
            "frequency": count / max(len(adaptive), 1),
        }
        for family, count in sorted(counts.items())
    ]


def _boundary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") != "ghvade_adaptive" or row.get("status") != "ok":
            continue
        diagnostics = row.get("fit_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        boundary = diagnostics.get("shape_gate_student_t_boundary_diagnostics")
        boundary = boundary if isinstance(boundary, dict) else {}
        for item in boundary.get("margin_deciles", []):
            if isinstance(item, dict):
                records.append(
                    {
                        "seed": row.get("seed", ""),
                        "selected_family": row.get("selected_family", ""),
                        **item,
                    }
                )
    return records


def _plot_boundary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    deciles = sorted({int(row["margin_decile"]) for row in rows})
    means: list[float] = []
    errors: list[float] = []
    for decile in deciles:
        values = [
            _number(row.get("decision_change_rate"))
            for row in rows
            if int(row["margin_decile"]) == decile
        ]
        values = [value for value in values if math.isfinite(value)]
        means.append(mean(values) if values else math.nan)
        errors.append(
            stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        )
    fig, axis = plt.subplots(figsize=(5.4, 3.5))
    axis.errorbar(deciles, means, yerr=errors, marker="o", color="#9c2f2f", capsize=3)
    axis.set_xlabel("Student-t margin decile (lowest first)")
    axis.set_ylabel("Assignment-change frequency")
    axis.set_xticks(deciles)
    axis.grid(axis="y", color="#d6d6d6", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the confirmed Paper 2 Dry Bean study.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/paper2_dry_bean_confirmatory"),
    )
    parser.add_argument(
        "--pilot-gate",
        type=Path,
        default=Path("results/paper2_dry_bean_blind_pilot/audit/pilot_gate.json"),
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("confirmatory", "descriptive_fallback"),
        default="confirmatory",
    )
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        default=Path(
            "papers/paper2_latent_geometry/DRY_BEAN_PROTOCOL_AMENDMENT.md"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper2_dry_bean_confirmatory/analysis"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = _require_pilot_decision(
        args.pilot_gate,
        analysis_mode=args.analysis_mode,
        amendment_path=args.protocol_amendment,
    )
    rows = _read_result_rows(args.result_root)
    expected = {(method, seed) for method in METHOD_ORDER for seed in range(4200, 4220)}
    observed = {(str(row.get("method")), int(row.get("seed"))) for row in rows}
    missing = sorted(expected - observed)
    duplicates = len(rows) - len(observed)
    if missing or duplicates:
        raise RuntimeError(
            "The frozen 140-task result matrix is incomplete; external metrics "
            f"remain sealed (missing={len(missing)}, duplicates={duplicates})."
        )
    blinded_rows = sum(bool(row.get("external_metrics_blinded")) for row in rows)
    if blinded_rows:
        raise RuntimeError("Confirmatory rows unexpectedly retain external-metric blinding.")

    method_summary = _method_summary(rows)
    paired = _paired_contrasts(rows)
    selection = _selection_summary(rows)
    boundary = _boundary_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "dry_bean_results_long.csv", rows)
    _write_csv(args.output_dir / "dry_bean_method_summary.csv", method_summary)
    _write_csv(args.output_dir / "dry_bean_paired_contrasts.csv", paired)
    _write_csv(args.output_dir / "dry_bean_selection_counts.csv", selection)
    _write_csv(args.output_dir / "dry_bean_boundary_deciles.csv", boundary)
    _plot_boundary(args.output_dir / "dry_bean_boundary_margin.png", boundary)

    summary = {
        "pilot_gate_passed": gate.get("gate_passed"),
        "analysis_mode": args.analysis_mode,
        "n_rows": len(rows),
        "n_expected": len(expected),
        "missing_tasks": [f"{method}:seed{seed}" for method, seed in missing],
        "duplicate_rows": duplicates,
        "method_summary": method_summary,
        "paired_contrasts": paired,
        "selection_counts": selection,
    }
    clean = _json_clean(summary)
    (args.output_dir / "dry_bean_confirmatory_summary.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(clean, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
