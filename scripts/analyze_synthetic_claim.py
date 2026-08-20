from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional reporting
    stats = None


GAUSSIAN = "vade_diag"
STUDENT_T = "ghvade_mvt"
CATALOGUE = "ghvade_gh_family"
CONSERVATIVE = "ghvade_conservative"
TAIL_SELECT = "ghvade_tail_select"
ASYMMETRIC_FIXED = ("ghvade_mvsn", "ghvade_mvvg", "ghvade_mvhb", "ghvade_mnig", "ghvade_mghp")
GH_FIXED = ("ghvade_mvt", "ghvade_mvvg", "ghvade_mvhb", "ghvade_mnig", "ghvade_mghp")
STABLE_FIXED = ("ghvade_mvt", "ghvade_mvsn", "ghvade_mvvg", "ghvade_mvhb", "ghvade_mnig", "ghvade_mghp")
FAMILY_MATCH = {
    "gaussian": GAUSSIAN,
    "student_t": STUDENT_T,
    "skew_normal": "ghvade_mvsn",
    "variance_gamma": "ghvade_mvvg",
    "hyperbolic": "ghvade_mvhb",
    "nig": "ghvade_mnig",
    "full_gh": "ghvade_mghp",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out = {}
    for row in rows:
        if row.get("status") == "ok":
            out[(row["design_id"], row["method"], row["seed"])] = row
    return out


def _design_meta(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    meta = {}
    for row in rows:
        design_id = row.get("design_id", "")
        if design_id and design_id not in meta:
            meta[design_id] = {
                "design_id": design_id,
                "tier": row.get("protocol_tier", ""),
                "true_family": row.get("true_family", ""),
                "scenario": row.get("scenario", ""),
                "skew_strength": row.get("skew_strength", ""),
                "tail_df": row.get("tail_df", ""),
                "outlier_rate": row.get("outlier_rate", ""),
            }
    return meta


def _method_means(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        value = _float(row, metric)
        if value is not None:
            grouped[(row["design_id"], row["method"])].append(value)
    return {key: mean(values) for key, values in grouped.items() if values}


def _paired_delta(
    by_key: dict[tuple[str, str, str], dict[str, str]],
    design_id: str,
    treatment: str,
    control: str,
    metric: str,
) -> dict[str, object]:
    seeds = sorted(
        {
            seed
            for did, method, seed in by_key
            if did == design_id and method in {treatment, control}
        },
        key=int,
    )
    deltas = []
    for seed in seeds:
        t = by_key.get((design_id, treatment, seed))
        c = by_key.get((design_id, control, seed))
        if t is None or c is None:
            continue
        t_value = _float(t, metric)
        c_value = _float(c, metric)
        if t_value is not None and c_value is not None:
            deltas.append(t_value - c_value)
    out: dict[str, object] = {
        "design_id": design_id,
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "n_pairs": len(deltas),
    }
    if deltas:
        out[f"delta_{metric}_mean"] = mean(deltas)
        out[f"delta_{metric}_std"] = pstdev(deltas) if len(deltas) > 1 else 0.0
        out["wins"] = sum(value > 0 for value in deltas)
        out["ties"] = sum(value == 0 for value in deltas)
        out["losses"] = sum(value < 0 for value in deltas)
        if len(deltas) > 1 and max(deltas) - min(deltas) <= 1e-12:
            out[f"delta_{metric}_ci95_low"] = mean(deltas)
            out[f"delta_{metric}_ci95_high"] = mean(deltas)
            out["ttest_p"] = 1.0 if deltas[0] == 0 else 0.0
            out["wilcoxon_p"] = 1.0 if deltas[0] == 0 else math.nan
        elif len(deltas) > 1 and stats is not None:
            sem = stats.sem(deltas)
            if sem > 0:
                ci = stats.t.interval(0.95, len(deltas) - 1, loc=mean(deltas), scale=sem)
                out[f"delta_{metric}_ci95_low"] = float(ci[0])
                out[f"delta_{metric}_ci95_high"] = float(ci[1])
            out["ttest_p"] = float(stats.ttest_rel(deltas, [0.0] * len(deltas)).pvalue)
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(deltas).pvalue)
            except ValueError:
                out["wilcoxon_p"] = math.nan
    return out


def _best_method(
    means: dict[tuple[str, str], float],
    design_id: str,
    candidates: tuple[str, ...],
) -> str | None:
    available = [(method, means[(design_id, method)]) for method in candidates if (design_id, method) in means]
    if not available:
        return None
    return max(available, key=lambda item: item[1])[0]


def _comparison_rows(rows: list[dict[str, str]], metric: str) -> list[dict[str, object]]:
    by_key = _group_rows(rows)
    meta = _design_meta(rows)
    means = _method_means(rows, metric)
    out = []
    for design_id, item in sorted(meta.items()):
        if item["tier"] != "core":
            continue
        true_family = item["true_family"]
        rows_to_add: list[tuple[str, str, str]] = []
        if (design_id, CATALOGUE) in means:
            rows_to_add.append(("catalogue_vs_gaussian", CATALOGUE, GAUSSIAN))
            if design_id != "g00" and (design_id, STUDENT_T) in means:
                rows_to_add.append(("catalogue_vs_student_t", CATALOGUE, STUDENT_T))
        if (design_id, CONSERVATIVE) in means:
            rows_to_add.append(("conservative_vs_gaussian", CONSERVATIVE, GAUSSIAN))
            if design_id != "g00" and (design_id, STUDENT_T) in means:
                rows_to_add.append(("conservative_vs_student_t", CONSERVATIVE, STUDENT_T))
        if (design_id, TAIL_SELECT) in means:
            rows_to_add.append(("tail_select_vs_gaussian", TAIL_SELECT, GAUSSIAN))
            if design_id != "g00" and (design_id, STUDENT_T) in means:
                rows_to_add.append(("tail_select_vs_student_t", TAIL_SELECT, STUDENT_T))
        best_gh = _best_method(means, design_id, GH_FIXED)
        best_stable = _best_method(means, design_id, STABLE_FIXED)
        best_asym = _best_method(means, design_id, ASYMMETRIC_FIXED)
        if best_gh is not None:
            rows_to_add.append(("best_gh_fixed_vs_gaussian", best_gh, GAUSSIAN))
        if best_stable is not None:
            rows_to_add.append(("best_stable_fixed_vs_gaussian", best_stable, GAUSSIAN))
        if best_asym is not None and true_family in {"skew_normal", "skew_t", "variance_gamma", "hyperbolic", "nig", "full_gh"}:
            rows_to_add.append(("best_asymmetric_fixed_vs_student_t", best_asym, STUDENT_T))
        matched = FAMILY_MATCH.get(true_family)
        if matched is not None and matched != GAUSSIAN:
            rows_to_add.append(("family_matched_vs_gaussian", matched, GAUSSIAN))
            if matched != STUDENT_T:
                rows_to_add.append(("family_matched_vs_student_t", matched, STUDENT_T))
        for claim_type, treatment, control in rows_to_add:
            comparison = _paired_delta(by_key, design_id, treatment, control, metric)
            comparison.update(item)
            comparison["claim_type"] = claim_type
            comparison["scope"] = (
                "primary_unsupervised"
                if claim_type.startswith("catalogue_")
                else "mechanistic_oracle"
                if claim_type.startswith(("best_", "family_matched"))
                else "primary"
            )
            out.append(comparison)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _claim_summary(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("metric") == metric and f"delta_{metric}_mean" in row:
            grouped[str(row["claim_type"])].append(row)
    summary = []
    for claim_type, group in sorted(grouped.items()):
        deltas = [float(row[f"delta_{metric}_mean"]) for row in group]
        summary.append(
            {
                "claim_type": claim_type,
                "metric": metric,
                "n_designs": len(group),
                "mean_design_delta": mean(deltas),
                "median_design_delta": sorted(deltas)[len(deltas) // 2],
                "positive_design_fraction": sum(value > 0 for value in deltas) / len(deltas),
                "material_gain_fraction_003": sum(value >= 0.03 for value in deltas) / len(deltas),
            }
        )
    return summary


def _markdown(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Synthetic Claim Audit",
        "",
        "This audit separates deployable selector evidence from mechanistic fixed-family evidence.",
        "`best_*` and `family_matched_*` rows use simulation design information or post-hoc fixed-family",
        "comparisons and therefore cannot be claimed as an unsupervised deployed selector.",
        "",
        "| Claim type | Metric | Designs | Mean delta | Positive fraction | >=0.03 fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['claim_type']} | {row['metric']} | {row['n_designs']} | "
            f"{float(row['mean_design_delta']):.4f} | "
            f"{float(row['positive_design_fraction']):.3f} | "
            f"{float(row['material_gain_fraction_003']):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit defensible synthetic claims from protocol results.")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.csv_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for metric in ("ari", "acc", "nmi"):
        comparisons.extend(_comparison_rows(rows, metric))
    summary = []
    for metric in ("ari", "acc", "nmi"):
        summary.extend(_claim_summary(comparisons, metric))
    _write_csv(args.output_dir / "synthetic_claim_comparisons.csv", comparisons)
    _write_csv(args.output_dir / "synthetic_claim_summary.csv", summary)
    (args.output_dir / "synthetic_claim_audit.md").write_text(_markdown(summary), encoding="utf-8")
    print((args.output_dir / "synthetic_claim_audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
