from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ASYMMETRIC_FAMILIES = {
    "skew_normal",
    "skew_t",
    "variance_gamma",
    "hyperbolic",
    "nig",
    "full_gh",
}

TRUE_TO_CANDIDATE = {
    "gaussian": "gaussian",
    "student_t": "student_t",
    "skew_normal": "skew_normal",
    "skew_t": "skew_t",
    "variance_gamma": "variance_gamma",
    "hyperbolic": "hyperbolic",
    "nig": "nig",
    "full_gh": "full_gh",
}


def _number(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _read_result_items(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic script
            rows.append({"status": "failed", "run_dir": str(path.parent), "error": repr(exc)})
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                rows.append({"run_dir": str(path.parent), **item})
    return rows


def _candidate_scores(diagnostics: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in diagnostics.get("candidates", []):
        if not isinstance(candidate, dict) or candidate.get("status") != "ok":
            continue
        name = str(candidate.get("candidate", ""))
        scores[name] = _number(candidate.get("selection_mean_log_score"))
    return scores


def _selector_rows(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for item in items:
        diagnostics = item.get("fit_diagnostics")
        if not isinstance(diagnostics, dict) or "candidates" not in diagnostics:
            continue
        scores = _candidate_scores(diagnostics)
        selected = str(item.get("selected_family", diagnostics.get("selected_candidate", "")))
        best = str(diagnostics.get("best_candidate", ""))
        true_family = str(item.get("true_family", ""))
        true_candidate = TRUE_TO_CANDIDATE.get(true_family, "")
        selected_score = scores.get(selected, math.nan)
        best_score = scores.get(best, math.nan)
        true_score = scores.get(true_candidate, math.nan)
        gaussian_score = scores.get("gaussian", math.nan)
        student_score = scores.get("student_t", math.nan)
        best_minus_selected = best_score - selected_score if math.isfinite(best_score) and math.isfinite(selected_score) else math.nan
        best_minus_gaussian = best_score - gaussian_score if math.isfinite(best_score) and math.isfinite(gaussian_score) else math.nan
        best_minus_student = best_score - student_score if math.isfinite(best_score) and math.isfinite(student_score) else math.nan
        true_minus_gaussian = true_score - gaussian_score if math.isfinite(true_score) and math.isfinite(gaussian_score) else math.nan
        true_minus_student = true_score - student_score if math.isfinite(true_score) and math.isfinite(student_score) else math.nan
        row = {
            "run_dir": item.get("run_dir", ""),
            "design_id": item.get("design_id", ""),
            "seed": item.get("seed", ""),
            "method": item.get("method", ""),
            "status": item.get("status", ""),
            "true_family": true_family,
            "selected_family": selected,
            "best_candidate": best,
            "selection_rule": diagnostics.get("selection_rule", ""),
            "best_mean_log_score": _number(diagnostics.get("best_mean_log_score")),
            "best_standard_error": _number(diagnostics.get("best_standard_error")),
            "one_se_threshold": _number(diagnostics.get("one_se_threshold")),
            "n_successful_candidates": diagnostics.get("n_successful_candidates", ""),
            "selected_score": selected_score,
            "true_family_score": true_score,
            "gaussian_score": gaussian_score,
            "student_t_score": student_score,
            "best_minus_selected": best_minus_selected,
            "best_minus_gaussian": best_minus_gaussian,
            "best_minus_student_t": best_minus_student,
            "true_family_minus_gaussian": true_minus_gaussian,
            "true_family_minus_student_t": true_minus_student,
            "selected_is_asymmetric": selected in ASYMMETRIC_FAMILIES,
            "best_is_asymmetric": best in ASYMMETRIC_FAMILIES,
            "true_family_is_asymmetric": true_family in ASYMMETRIC_FAMILIES,
            "one_se_simplified": selected != best,
        }
        rows.append(row)
        for candidate, score in sorted(scores.items()):
            long_rows.append(
                {
                    "design_id": row["design_id"],
                    "seed": row["seed"],
                    "true_family": true_family,
                    "selected_family": selected,
                    "best_candidate": best,
                    "candidate": candidate,
                    "selection_mean_log_score": score,
                    "delta_from_best": score - best_score if math.isfinite(best_score) else math.nan,
                }
            )
    return rows, long_rows


def _finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else math.nan


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    asym = [row for row in ok if row.get("true_family_is_asymmetric")]
    modstrong = [
        row
        for row in asym
        if any(str(row.get("design_id", "")).endswith(suffix) for suffix in ("s150", "s300"))
    ]
    hidden = [
        row
        for row in asym
        if row.get("best_is_asymmetric") and not row.get("selected_is_asymmetric")
    ]
    by_true_selected = Counter((str(row["true_family"]), str(row["selected_family"])) for row in ok)
    by_true_best = Counter((str(row["true_family"]), str(row["best_candidate"])) for row in ok)
    return {
        "n_selector_runs": len(rows),
        "n_ok": len(ok),
        "selected_asymmetric_rate_asym_true": mean(bool(row["selected_is_asymmetric"]) for row in asym) if asym else math.nan,
        "best_asymmetric_rate_asym_true": mean(bool(row["best_is_asymmetric"]) for row in asym) if asym else math.nan,
        "selected_asymmetric_rate_modstrong_asym": (
            mean(bool(row["selected_is_asymmetric"]) for row in modstrong) if modstrong else math.nan
        ),
        "best_asymmetric_rate_modstrong_asym": (
            mean(bool(row["best_is_asymmetric"]) for row in modstrong) if modstrong else math.nan
        ),
        "one_se_hidden_asym_best_rate": len(hidden) / len(asym) if asym else math.nan,
        "mean_best_minus_selected": _finite_mean([float(row["best_minus_selected"]) for row in ok]),
        "mean_best_minus_gaussian_asym_true": _finite_mean([float(row["best_minus_gaussian"]) for row in asym]),
        "mean_best_minus_student_t_asym_true": _finite_mean([float(row["best_minus_student_t"]) for row in asym]),
        "selected_counts": dict(Counter(str(row["selected_family"]) for row in ok)),
        "best_counts": dict(Counter(str(row["best_candidate"]) for row in ok)),
        "selected_by_true_family": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_true_selected.items())},
        "best_by_true_family": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_true_best.items())},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    def pct(value: Any) -> str:
        value = _number(value)
        return "NA" if not math.isfinite(value) else f"{100.0 * value:.1f}%"

    def num(value: Any) -> str:
        value = _number(value)
        return "NA" if not math.isfinite(value) else f"{value:.4f}"

    lines = [
        "# Adaptive Candidate Diagnostics",
        "",
        "This audit reads per-run selector diagnostics and separates the candidate with the",
        "highest held-out latent likelihood from the candidate selected by the deployed rule.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Selector runs | {summary['n_selector_runs']} |",
        f"| OK runs | {summary['n_ok']} |",
        f"| Selected asymmetric rate, asymmetric truth | {pct(summary['selected_asymmetric_rate_asym_true'])} |",
        f"| Best asymmetric rate, asymmetric truth | {pct(summary['best_asymmetric_rate_asym_true'])} |",
        f"| Selected asymmetric rate, moderate/strong asymmetric truth | {pct(summary['selected_asymmetric_rate_modstrong_asym'])} |",
        f"| Best asymmetric rate, moderate/strong asymmetric truth | {pct(summary['best_asymmetric_rate_modstrong_asym'])} |",
        f"| Asymmetric best hidden by one-SE rate | {pct(summary['one_se_hidden_asym_best_rate'])} |",
        f"| Mean best-selected log-score gap | {num(summary['mean_best_minus_selected'])} |",
        f"| Mean best-Gaussian log-score gap, asymmetric truth | {num(summary['mean_best_minus_gaussian_asym_true'])} |",
        f"| Mean best-Student-t log-score gap, asymmetric truth | {num(summary['mean_best_minus_student_t_asym_true'])} |",
        "",
        "## Selected Counts",
        "",
    ]
    for family, count in sorted(summary["selected_counts"].items()):
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Best-Candidate Counts", ""])
    for family, count in sorted(summary["best_counts"].items()):
        lines.append(f"- `{family}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit adaptive GH-VAE candidate diagnostics.")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = _read_result_items(args.result_root)
    rows, long_rows = _selector_rows(items)
    summary = _summary(rows)
    _write_csv(args.output_dir / "adaptive_candidate_diagnostics.csv", rows)
    _write_csv(args.output_dir / "adaptive_candidate_scores_long.csv", long_rows)
    (args.output_dir / "adaptive_candidate_summary.json").write_text(
        json.dumps(_json_clean(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "adaptive_candidate_summary.md", summary)
    print(json.dumps(_json_clean(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
