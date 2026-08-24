from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ASYMMETRIC_FAMILIES = frozenset(
    {"skew_normal", "variance_gamma", "hyperbolic", "nig", "skew_t", "full_gh"}
)
EXTERNAL_METRICS = frozenset({"acc", "ari", "nmi", "purity"})


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _external_metric_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).casefold()
            path = f"{prefix}.{key}" if prefix else str(key)
            if name in EXTERNAL_METRICS or any(
                name.endswith(f"_{metric}") for metric in EXTERNAL_METRICS
            ):
                paths.append(path)
            paths.extend(_external_metric_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_external_metric_paths(item, f"{prefix}[{index}]"))
    return paths


def _candidate_scores(diagnostics: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in diagnostics.get("candidates", []):
        if not isinstance(candidate, dict) or candidate.get("status") != "ok":
            continue
        score = _number(candidate.get("selection_mean_log_score"))
        if math.isfinite(score):
            scores[str(candidate.get("candidate", ""))] = score
    return scores


def _read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("*/results.json")):
        if not result_path.parent.name.startswith(
            "pilot__dry_bean__ghvade_adaptive__seed"
        ):
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict) and item.get("method") == "ghvade_adaptive":
                rows.append({"run_dir": result_path.parent.name, **item})
    return rows


def _audit_row(item: dict[str, Any]) -> dict[str, Any]:
    diagnostics = item.get("fit_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    boundary = diagnostics.get("shape_gate_student_t_boundary_diagnostics")
    boundary = boundary if isinstance(boundary, dict) else {}
    regularity = item.get("final_prior_regularity")
    regularity = regularity if isinstance(regularity, dict) else {}
    scores = _candidate_scores(diagnostics)
    student_score = scores.get("student_t", math.nan)
    selected_family = str(item.get("selected_family", ""))
    best_asymmetric = str(diagnostics.get("shape_gate_best_asymmetric_candidate", ""))
    best_asymmetric_score = scores.get(best_asymmetric, math.nan)
    score_gain = (
        best_asymmetric_score - student_score
        if math.isfinite(best_asymmetric_score) and math.isfinite(student_score)
        else math.nan
    )
    candidate_deltas = [
        abs(score - student_score)
        for score in scores.values()
        if math.isfinite(student_score)
    ]
    max_candidate_delta = max(candidate_deltas, default=math.nan)
    leaks = _external_metric_paths(item)

    reasons: list[str] = []
    if item.get("status") != "ok":
        reasons.append("task_not_ok")
    if item.get("external_metrics_blinded") is not True:
        reasons.append("blinding_not_active")
    if leaks:
        reasons.append("external_metric_leak")
    if regularity.get("regular") is not True:
        reasons.append("irregular_final_prior")
    if int(item.get("active_components", 0) or 0) != 7:
        reasons.append("not_all_components_active")
    if int(item.get("n_assigned", 0) or 0) != 7:
        reasons.append("not_all_components_assigned")
    if not math.isfinite(student_score) or not math.isfinite(best_asymmetric_score):
        reasons.append("missing_finite_anchor_score")
    if not math.isfinite(max_candidate_delta) or max_candidate_delta > 5.0:
        reasons.append("candidate_score_sanity_failure")
    if boundary.get("status") != "ok":
        reasons.append("missing_boundary_diagnostic")

    return {
        "run_dir": item.get("run_dir", ""),
        "seed": item.get("seed", ""),
        "status": item.get("status", ""),
        "external_metrics_blinded": item.get("external_metrics_blinded", False),
        "external_metric_leak_count": len(leaks),
        "selected_family": selected_family,
        "selected_is_asymmetric": selected_family in ASYMMETRIC_FAMILIES,
        "shape_gate_triggered": bool(diagnostics.get("shape_gate_triggered", False)),
        "best_asymmetric_candidate": best_asymmetric,
        "student_t_score": student_score,
        "best_asymmetric_score": best_asymmetric_score,
        "best_asymmetric_minus_student_t": score_gain,
        "max_candidate_abs_delta_from_student_t": max_candidate_delta,
        "active_components": item.get("active_components", ""),
        "n_assigned": item.get("n_assigned", ""),
        "final_prior_regular": regularity.get("regular", False),
        "decision_change_fraction": _number(boundary.get("decision_change_fraction")),
        "boundary_disagreement_concentration": _number(
            boundary.get("boundary_disagreement_concentration")
        ),
        "boundary_disagreement_share": _number(
            boundary.get("boundary_disagreement_share")
        ),
        "valid": not reasons,
        "invalid_reasons": ";".join(reasons),
    }


def evaluate_gate(rows: list[dict[str, Any]], expected_runs: int = 5) -> dict[str, Any]:
    audited = [_audit_row(row) for row in rows]
    valid = [row for row in audited if row["valid"]]
    promoted = [
        row
        for row in valid
        if row["selected_is_asymmetric"] and row["shape_gate_triggered"]
    ]
    boundary_supported = [
        row
        for row in promoted
        if _number(row["decision_change_fraction"]) >= 0.005
        and _number(row["boundary_disagreement_concentration"]) > 1.5
    ]
    promoted_gains = [
        _number(row["best_asymmetric_minus_student_t"])
        for row in promoted
        if math.isfinite(_number(row["best_asymmetric_minus_student_t"]))
    ]
    median_gain = median(promoted_gains) if promoted_gains else math.nan
    checks = {
        "all_expected_tasks_present": len(audited) == int(expected_runs),
        "no_external_metric_leak": all(
            int(row["external_metric_leak_count"]) == 0 for row in audited
        ),
        "at_least_four_valid_runs": len(valid) >= 4,
        "at_least_three_asymmetric_promotions": len(promoted) >= 3,
        "median_score_gain_at_least_0_01": (
            math.isfinite(median_gain) and median_gain >= 0.01
        ),
        "at_least_three_boundary_supported_promotions": len(boundary_supported) >= 3,
    }
    return {
        "gate_passed": all(checks.values()),
        "checks": checks,
        "expected_runs": int(expected_runs),
        "n_rows": len(audited),
        "n_valid": len(valid),
        "n_asymmetric_promotions": len(promoted),
        "n_boundary_supported_promotions": len(boundary_supported),
        "median_best_asymmetric_minus_student_t": median_gain,
        "audited_rows": audited,
    }


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, gate: dict[str, Any]) -> None:
    outcome = "PASS" if gate["gate_passed"] else "FAIL"
    lines = [
        "# Dry Bean Blinded Pilot Gate",
        "",
        f"Prospective gate: **{outcome}**.",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'pass' if passed else 'fail'} |")
    lines.extend(
        [
            "",
            f"Valid runs: {gate['n_valid']}/{gate['n_rows']}.",
            f"Asymmetric promotions: {gate['n_asymmetric_promotions']}.",
            f"Boundary-supported promotions: {gate['n_boundary_supported_promotions']}.",
            "",
            "External labels were not consulted by this audit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the blinded Paper 2 Dry Bean pilot.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/paper2_dry_bean_blind_pilot"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper2_dry_bean_blind_pilot/audit"),
    )
    parser.add_argument("--expected-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.result_root)
    gate = evaluate_gate(rows, expected_runs=args.expected_runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "pilot_run_audit.csv", gate["audited_rows"])
    clean = _json_clean(gate)
    (args.output_dir / "pilot_gate.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "pilot_gate.md", gate)
    print(json.dumps(clean, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
