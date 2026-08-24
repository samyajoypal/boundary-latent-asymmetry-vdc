from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from scipy import stats


ADAPTIVE = "ghvade_adaptive"
TAIL_SELECT = "ghvade_tail_select"
CONSERVATIVE = "ghvade_conservative"
SELECTOR_METHODS = (ADAPTIVE, TAIL_SELECT, CONSERVATIVE)
PRIMARY_SELECTOR_PRIORITY = (TAIL_SELECT, CONSERVATIVE, ADAPTIVE)
GAUSSIAN = "vade_diag"
STUDENT_T = "ghvade_mvt"
FIXED_METHODS = (
    GAUSSIAN,
    STUDENT_T,
    "ghvade_gh_family",
    "ghvade_mvsn",
    "ghvade_mgst",
    "ghvade_mvvg",
    "ghvade_mvhb",
    "ghvade_mnig",
    "ghvade_mghp",
)


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selector_methods(rows: list[dict[str, str]]) -> list[str]:
    present = {row.get("method", "") for row in rows}
    ordered = [method for method in PRIMARY_SELECTOR_PRIORITY if method in present]
    ordered.extend(method for method in SELECTOR_METHODS if method in present and method not in ordered)
    return ordered


def _comparison_label(selector: str, control_family: str) -> str:
    if selector == ADAPTIVE:
        return f"adaptive_vs_{control_family}"
    return f"{selector.removeprefix('ghvade_')}_vs_{control_family}"


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    sem = stats.sem(values)
    low, high = stats.t.interval(0.95, len(values) - 1, loc=mean(values), scale=sem)
    return float(low), float(high)


def _holm(rows: list[dict[str, object]], p_key: str, out_key: str) -> None:
    eligible = [(index, float(row[p_key])) for index, row in enumerate(rows) if math.isfinite(float(row[p_key]))]
    eligible.sort(key=lambda item: item[1])
    running = 0.0
    total = len(eligible)
    for rank, (index, p_value) in enumerate(eligible):
        running = max(running, min(1.0, (total - rank) * p_value))
        rows[index][out_key] = running


def paired_effects(
    rows: list[dict[str, str]],
    treatment: str,
    control: str,
    designs: Iterable[str],
    comparison_family: str,
) -> list[dict[str, object]]:
    designs = set(designs)
    lookup = {
        (row.get("design_id", ""), row.get("method", ""), int(row.get("seed", -1))): row
        for row in rows
        if row.get("status") == "ok" and row.get("design_id") in designs
    }
    output: list[dict[str, object]] = []
    for design in sorted(designs):
        deltas: list[float] = []
        seeds = sorted({seed for d, _, seed in lookup if d == design})
        for seed in seeds:
            treatment_row = lookup.get((design, treatment, seed))
            control_row = lookup.get((design, control, seed))
            if treatment_row is None or control_row is None:
                continue
            treatment_ari = _number(treatment_row, "ari")
            control_ari = _number(control_row, "ari")
            if treatment_ari is not None and control_ari is not None:
                deltas.append(treatment_ari - control_ari)
        low, high = _ci95(deltas)
        t_p = float(stats.ttest_1samp(deltas, 0.0).pvalue) if len(deltas) > 1 else math.nan
        try:
            wilcoxon_p = float(stats.wilcoxon(deltas).pvalue) if len(deltas) > 1 else math.nan
        except ValueError:
            wilcoxon_p = 1.0
        output.append(
            {
                "comparison_family": comparison_family,
                "design_id": design,
                "treatment": treatment,
                "control": control,
                "n_pairs": len(deltas),
                "ari_delta_mean": mean(deltas) if deltas else math.nan,
                "ari_delta_std": stdev(deltas) if len(deltas) > 1 else math.nan,
                "ari_delta_ci95_low": low,
                "ari_delta_ci95_high": high,
                "ttest_p": t_p,
                "wilcoxon_p": wilcoxon_p,
                "wins": sum(delta > 0.0 for delta in deltas),
            }
        )
    _holm(output, "wilcoxon_p", "wilcoxon_holm_p")
    return output


def _method_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(row.get("design_id", ""), row.get("method", ""))].append(row)
    output: list[dict[str, object]] = []
    for (design, method), group in sorted(groups.items()):
        item: dict[str, object] = {"design_id": design, "method": method, "n_runs": len(group)}
        for metric in ("ari", "acc", "nmi"):
            values = [value for row in group if (value := _number(row, metric)) is not None]
            if values:
                low, high = _ci95(values)
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values) if len(values) > 1 else math.nan
                item[f"{metric}_ci95_low"] = low
                item[f"{metric}_ci95_high"] = high
        output.append(item)
    return output


def _selection_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in rows:
        if row.get("status") == "ok" and row.get("method") in SELECTOR_METHODS:
            groups[
                (
                    row.get("method", ""),
                    row.get("design_id", ""),
                    row.get("true_family", ""),
                    row.get("selected_family", ""),
                )
            ] += 1
    return [
        {
            "method": method,
            "design_id": design,
            "true_family": truth,
            "selected_family": selected,
            "count": count,
        }
        for (method, design, truth, selected), count in sorted(groups.items())
    ]


def _failure_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("method", "")].append(row)
    return [
        {
            "method": method,
            "n_total": len(group),
            "n_failed": sum(row.get("status") != "ok" for row in group),
            "failure_rate": sum(row.get("status") != "ok" for row in group) / len(group),
        }
        for method, group in sorted(groups.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selector_decision(
    selector: str,
    summaries: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    core_designs: list[str],
) -> dict[str, object]:
    summary_lookup = {(str(row["design_id"]), str(row["method"])): row for row in summaries}
    non_gaussian = [design for design in core_designs if design != "g00"]
    tied = 0
    evaluated = 0
    for design in non_gaussian:
        selected = summary_lookup.get((design, selector), {}).get("ari_mean")
        fixed = [
            summary_lookup.get((design, method), {}).get("ari_mean")
            for method in FIXED_METHODS
        ]
        fixed = [float(value) for value in fixed if value is not None]
        if selected is not None and fixed:
            evaluated += 1
            tied += float(selected) >= max(fixed) - 0.02

    comparison_lookup = {
        (str(row["comparison_family"]), str(row["design_id"])): row for row in comparisons
    }
    gaussian_key = _comparison_label(selector, "gaussian")
    student_key = _comparison_label(selector, "student_t")
    heavy = [
        float(comparison_lookup[(gaussian_key, design)]["ari_delta_mean"])
        for design in ("t_df05", "t_df03")
        if (gaussian_key, design) in comparison_lookup
    ]
    skew_designs = [design for design in core_designs if design.startswith(("sn_", "st_"))]
    skew_effects = [
        float(comparison_lookup[(student_key, design)]["ari_delta_mean"])
        for design in skew_designs
        if (student_key, design) in comparison_lookup
    ]
    gaussian_effect = comparison_lookup.get((gaussian_key, "g00"), {}).get("ari_delta_mean")
    return {
        "best_or_tied_fraction": tied / evaluated if evaluated else math.nan,
        "best_or_tied_pass": bool(evaluated and tied / evaluated >= 0.8),
        "heavy_tail_mean_ari_gain": mean(heavy) if heavy else math.nan,
        "heavy_tail_material_pass": bool(heavy and mean(heavy) >= 0.03),
        "skew_cells_gain_at_least_003_fraction": (
            sum(value >= 0.03 for value in skew_effects) / len(skew_effects) if skew_effects else math.nan
        ),
        "skew_material_pass": bool(
            skew_effects and sum(value >= 0.03 for value in skew_effects) / len(skew_effects) >= 0.5
        ),
        "gaussian_control_ari_delta": gaussian_effect,
        "gaussian_noninferiority_pass": bool(
            gaussian_effect is not None and float(gaussian_effect) > -0.03
        ),
    }


def _decision(
    selector_methods: list[str],
    summaries: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    summary_lookup = {(str(row["design_id"]), str(row["method"])): row for row in summaries}
    core_designs = sorted(
        {design for design, _ in summary_lookup if not design.startswith(("contam_", "unequal_", "imbalance_", "overlap", "underfit_", "overfit_"))}
    )
    selector_decisions = {
        method: _selector_decision(method, summaries, comparisons, core_designs)
        for method in selector_methods
    }
    primary = selector_methods[0] if selector_methods else None
    primary_decision = selector_decisions.get(primary, {})
    max_failure = max((float(row["failure_rate"]) for row in failures), default=math.nan)
    output: dict[str, object] = {
        "primary_selector": primary,
        "selector_decisions": selector_decisions,
        "maximum_method_failure_rate": max_failure,
        "failure_rate_pass": bool(math.isfinite(max_failure) and max_failure <= 0.05),
    }
    if primary_decision:
        output.update(
            {
                "adaptive_best_or_tied_fraction": primary_decision["best_or_tied_fraction"],
                "adaptive_best_or_tied_pass": primary_decision["best_or_tied_pass"],
                "heavy_tail_mean_ari_gain": primary_decision["heavy_tail_mean_ari_gain"],
                "heavy_tail_material_pass": primary_decision["heavy_tail_material_pass"],
                "skew_cells_gain_at_least_003_fraction": primary_decision["skew_cells_gain_at_least_003_fraction"],
                "skew_material_pass": primary_decision["skew_material_pass"],
                "gaussian_control_ari_delta": primary_decision["gaussian_control_ari_delta"],
                "gaussian_noninferiority_pass": primary_decision["gaussian_noninferiority_pass"],
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen ARI-first simulation analysis.")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "final"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read(args.csv_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = _method_summary(rows)
    selector_methods = _selector_methods(rows)
    core = sorted({row.get("design_id", "") for row in rows if row.get("protocol_tier") == "core"})
    non_gaussian = [design for design in core if design != "g00"]
    skew = [design for design in core if design.startswith(("sn_", "st_"))]
    comparisons = []
    for selector in selector_methods:
        comparisons += paired_effects(rows, selector, GAUSSIAN, core, _comparison_label(selector, "gaussian"))
        comparisons += paired_effects(rows, selector, STUDENT_T, skew, _comparison_label(selector, "student_t"))
    failures = _failure_summary(rows)
    selections = _selection_summary(rows)
    decision = _decision(selector_methods, summaries, comparisons, failures)

    _write_csv(args.output_dir / "method_summary.csv", summaries)
    _write_csv(args.output_dir / "paired_ari_comparisons.csv", comparisons)
    _write_csv(args.output_dir / "family_selection.csv", selections)
    _write_csv(args.output_dir / "failures.csv", failures)
    payload = {
        "stage": args.stage,
        "confirmatory": args.stage == "final",
        "n_rows": len(rows),
        "n_core_designs": len(core),
        "n_non_gaussian_core_designs": len(non_gaussian),
        "decision": decision,
    }
    safe_payload = _json_safe(payload)
    (args.output_dir / "decision.json").write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
