from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ASYMMETRIC_FAMILIES = {
    "skew_normal",
    "skew_t",
    "variance_gamma",
    "hyperbolic",
    "nig",
    "full_gh",
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_mean(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _number(value)) is not None]
    return mean(finite) if finite else None


def _read_items(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    items: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in sorted(root.rglob("results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append({"path": str(path), "error": repr(exc)})
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if isinstance(record, dict):
                items.append({"result_path": str(path), **record})
    return items, failures


def _selector_events(
    item: dict[str, Any],
) -> Iterable[tuple[str, int, dict[str, Any]]]:
    """Yield final and transactionally rejected selector events.

    A Student-t fallback replaces the public ``prior_update_history`` with the
    matched reference history. The attempted GH history is intentionally kept
    under ``componentwise_attempt_prior_update_history`` and must be audited as
    well; otherwise every tournament veto is incorrectly counted as a run in
    which the selector never promoted a candidate.
    """

    histories = (
        ("returned", item.get("prior_update_history")),
        ("componentwise_attempt", item.get("componentwise_attempt_prior_update_history")),
    )
    for source, history in histories:
        if not isinstance(history, list):
            continue
        for event_index, event in enumerate(history):
            if isinstance(event, dict) and isinstance(event.get("candidates"), list):
                yield source, event_index, event


def _extract(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for item in items:
        for history_source, event_index, event in _selector_events(item):
            candidates = [candidate for candidate in event["candidates"] if isinstance(candidate, dict)]
            selected_family = str(event.get("selected_family") or item.get("selected_family") or "")
            run_row = {
                "result_path": item.get("result_path", ""),
                "dataset": item.get("dataset", ""),
                "design_id": item.get("design_id", ""),
                "seed": item.get("seed", ""),
                "status": item.get("status", ""),
                "true_family": item.get("true_family", ""),
                "true_component_families": json.dumps(item.get("true_component_families", [])),
                "event_index": event_index,
                "history_source": history_source,
                "returned_after_tournament": not bool(
                    item.get("componentwise_fallback_triggered", False)
                ),
                "tournament_score_gain": item.get(
                    "componentwise_tournament_score_gain", ""
                ),
                "selection_rule": event.get("selection_rule", ""),
                "selected_family": selected_family,
                "selected_component": event.get("selected_component", ""),
                "promoted": bool(event.get("promoted", False)),
                "selected_is_asymmetric": selected_family in ASYMMETRIC_FAMILIES,
                "n_candidates": len(candidates),
                "n_successful_candidates": event.get("n_successful_candidates", ""),
                "n_eligible_candidates": event.get("n_eligible_candidates", ""),
                "minimum_decision_change_fraction": event.get("minimum_decision_change_fraction", ""),
                "minimum_boundary_leverage": event.get("minimum_boundary_leverage", ""),
                "minimum_boundary_score_gain": event.get("minimum_boundary_score_gain", ""),
                "confidence_z": event.get("confidence_z", ""),
                "familywise_alpha": event.get("familywise_alpha", ""),
                "anchor_component_abs_skew_max": (
                    event.get("anchor_component_shapes", {}).get("anchor_component_abs_skew_max", "")
                    if isinstance(event.get("anchor_component_shapes"), dict)
                    else ""
                ),
            }
            run_rows.append(run_row)

            decision_threshold = _number(event.get("minimum_decision_change_fraction"))
            boundary_gain_threshold = _number(event.get("minimum_boundary_score_gain"))
            for candidate in candidates:
                boundary = candidate.get("boundary_relevance")
                boundary = boundary if isinstance(boundary, dict) else {}
                shape = candidate.get("component_shape")
                shape = shape if isinstance(shape, dict) else {}
                agreement = candidate.get("anchor_assignment_agreement")
                agreement = agreement if isinstance(agreement, dict) else {}
                regularity = candidate.get("regularity")
                regularity = regularity if isinstance(regularity, dict) else {}
                reasons_value = candidate.get("rejection_reasons", [])
                reasons = [str(reason) for reason in reasons_value] if isinstance(reasons_value, list) else []
                decision_change = _number(boundary.get("decision_change_fraction"))
                boundary_lcb = _number(boundary.get("boundary_score_lower_bound"))
                paired_lcb = _number(candidate.get("paired_score_lower_bound"))
                row = {
                    "result_path": item.get("result_path", ""),
                    "dataset": item.get("dataset", ""),
                    "design_id": item.get("design_id", ""),
                    "seed": item.get("seed", ""),
                    "true_family": item.get("true_family", ""),
                    "selected_family": selected_family,
                    "promoted": bool(event.get("promoted", False)),
                    "event_index": event_index,
                    "history_source": history_source,
                    "family": candidate.get("family", ""),
                    "family_code": candidate.get("family_code", ""),
                    "component": candidate.get("component", ""),
                    "complexity_rank": candidate.get("complexity_rank", ""),
                    "status": candidate.get("status", ""),
                    "eligible": bool(candidate.get("eligible", False)),
                    "rejection_reasons": ";".join(reasons),
                    "n_rejection_reasons": len(reasons),
                    "paired_score_gain": candidate.get("paired_score_gain", ""),
                    "paired_score_lower_bound": candidate.get("paired_score_lower_bound", ""),
                    "paired_score_standard_error": candidate.get("paired_score_standard_error", ""),
                    "mixture_score_gain": candidate.get("mixture_score_gain", ""),
                    "mixture_score_standard_error": candidate.get("mixture_score_standard_error", ""),
                    "weighted_component_mean_log_score_gain": candidate.get(
                        "weighted_component_mean_log_score_gain", ""
                    ),
                    "weighted_component_mean_log_score_gain_standard_error": candidate.get(
                        "weighted_component_mean_log_score_gain_standard_error", ""
                    ),
                    "boundary_score_gain": boundary.get("boundary_score_gain", ""),
                    "boundary_score_lower_bound": boundary.get("boundary_score_lower_bound", ""),
                    "boundary_score_standard_error": boundary.get("boundary_score_standard_error", ""),
                    "boundary_leverage": boundary.get("boundary_leverage", ""),
                    "boundary_responsibility_shift": boundary.get("boundary_responsibility_shift", ""),
                    "decision_change_fraction": boundary.get("decision_change_fraction", ""),
                    "boundary_effective_sample_size": boundary.get("boundary_effective_sample_size", ""),
                    "component_abs_skew": shape.get("crossfit_directional_abs_skew", ""),
                    "component_skew_signal": shape.get("normal_reference_skew_signal", ""),
                    "fitted_skewness_norm": candidate.get("skewness_norm", ""),
                    "fitted_alpha_bar": candidate.get("fitted_alpha_bar", ""),
                    "fitted_chi": candidate.get("fitted_chi", ""),
                    "fitted_psi": candidate.get("fitted_psi", ""),
                    "fitted_lambda": candidate.get("fitted_lmbda", ""),
                    "fitted_df": candidate.get("fitted_df", ""),
                    "hard_assignment_agreement": agreement.get("hard_assignment_agreement", ""),
                    "mean_assignment_total_variation": agreement.get("mean_assignment_total_variation", ""),
                    "regular": regularity.get("regular", ""),
                    "regularity_violations": ";".join(
                        str(value) for value in regularity.get("violations", [])
                    ),
                    "passes_decision_gate": (
                        decision_change is not None
                        and decision_threshold is not None
                        and decision_change >= decision_threshold
                    ),
                    "positive_paired_score_lcb": paired_lcb is not None and paired_lcb > 0.0,
                    "passes_boundary_score_gate": (
                        boundary_lcb is not None
                        and boundary_gain_threshold is not None
                        and boundary_lcb >= boundary_gain_threshold
                    ),
                    "rejected_only_for_decision_change": reasons == ["insufficient_decision_change"],
                }
                candidate_rows.append(row)
    return run_rows, candidate_rows


def _group_summary(
    run_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_design: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_design[str(row.get("design_id", ""))].append(row)
    design_rows = []
    for design_id, group in sorted(by_design.items()):
        design_rows.append(
            {
                "design_id": design_id,
                "n_runs": len(group),
                "n_promoted": sum(bool(row["promoted"]) for row in group),
                "n_selected_asymmetric": sum(bool(row["selected_is_asymmetric"]) for row in group),
                "selected_family_counts": json.dumps(
                    dict(sorted(Counter(str(row["selected_family"]) for row in group).items()))
                ),
                "mean_anchor_component_abs_skew_max": _finite_mean(
                    row.get("anchor_component_abs_skew_max") for row in group
                ),
            }
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_family[str(row.get("family", ""))].append(row)
    family_rows = []
    for family, group in sorted(by_family.items()):
        family_rows.append(
            {
                "family": family,
                "n_candidates": len(group),
                "n_eligible": sum(bool(row["eligible"]) for row in group),
                "n_positive_paired_score_lcb": sum(bool(row["positive_paired_score_lcb"]) for row in group),
                "n_passing_boundary_score_gate": sum(bool(row["passes_boundary_score_gate"]) for row in group),
                "n_passing_decision_gate": sum(bool(row["passes_decision_gate"]) for row in group),
                "mean_paired_score_gain": _finite_mean(row.get("paired_score_gain") for row in group),
                "mean_boundary_score_gain": _finite_mean(row.get("boundary_score_gain") for row in group),
                "mean_decision_change_fraction": _finite_mean(
                    row.get("decision_change_fraction") for row in group
                ),
                "max_decision_change_fraction": max(
                    (value for row in group if (value := _number(row.get("decision_change_fraction"))) is not None),
                    default=None,
                ),
            }
        )
    return design_rows, family_rows


def _summary(
    items: list[dict[str, Any]],
    read_failures: list[dict[str, str]],
    run_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rejection_counts = Counter(
        reason
        for row in candidate_rows
        for reason in str(row.get("rejection_reasons", "")).split(";")
        if reason
    )
    return {
        "n_result_items": len(items),
        "n_read_failures": len(read_failures),
        "n_selector_events": len(run_rows),
        "n_promoted": sum(bool(row["promoted"]) for row in run_rows),
        "n_selected_asymmetric": sum(bool(row["selected_is_asymmetric"]) for row in run_rows),
        "selected_family_counts": dict(sorted(Counter(str(row["selected_family"]) for row in run_rows).items())),
        "n_candidates": len(candidate_rows),
        "n_eligible_candidates": sum(bool(row["eligible"]) for row in candidate_rows),
        "n_candidates_positive_paired_score_lcb": sum(
            bool(row["positive_paired_score_lcb"]) for row in candidate_rows
        ),
        "n_candidates_passing_boundary_score_gate": sum(
            bool(row["passes_boundary_score_gate"]) for row in candidate_rows
        ),
        "n_candidates_passing_decision_gate": sum(bool(row["passes_decision_gate"]) for row in candidate_rows),
        "n_rejected_only_for_decision_change": sum(
            bool(row["rejected_only_for_decision_change"]) for row in candidate_rows
        ),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# V11 Candidate Promotion Audit",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Result items | {summary['n_result_items']} |",
        f"| Selector events | {summary['n_selector_events']} |",
        f"| Promoted runs | {summary['n_promoted']} |",
        f"| Asymmetric selections | {summary['n_selected_asymmetric']} |",
        f"| Candidate fits | {summary['n_candidates']} |",
        f"| Eligible candidates | {summary['n_eligible_candidates']} |",
        f"| Positive paired-score lower bound | {summary['n_candidates_positive_paired_score_lcb']} |",
        f"| Passing boundary-score gate | {summary['n_candidates_passing_boundary_score_gate']} |",
        f"| Passing decision-change gate | {summary['n_candidates_passing_decision_gate']} |",
        f"| Rejected only for decision change | {summary['n_rejected_only_for_decision_change']} |",
        "",
        "## Selected Families",
        "",
    ]
    lines.extend(f"- `{family}`: {count}" for family, count in summary["selected_family_counts"].items())
    lines.extend(["", "## Rejection Reasons", ""])
    lines.extend(f"- `{reason}`: {count}" for reason, count in summary["rejection_reason_counts"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v11 Student-t-anchored GH candidate promotion decisions.")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items, read_failures = _read_items(args.result_root)
    run_rows, candidate_rows = _extract(items)
    design_rows, family_rows = _group_summary(run_rows, candidate_rows)
    summary = _summary(items, read_failures, run_rows, candidate_rows)

    _write_csv(args.output_dir / "selector_runs.csv", run_rows)
    _write_csv(args.output_dir / "candidate_fits.csv", candidate_rows)
    _write_csv(args.output_dir / "design_summary.csv", design_rows)
    _write_csv(args.output_dir / "family_summary.csv", family_rows)
    _write_csv(args.output_dir / "read_failures.csv", read_failures)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
