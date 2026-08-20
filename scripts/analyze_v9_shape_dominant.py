from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional reporting dependency
    stats = None


ADAPTIVE = "ghvade_adaptive"
GAUSSIAN = "vade_diag"
STUDENT_T = "ghvade_mvt"
ASYMMETRIC_FAMILIES = {
    "skew_normal",
    "skew_t",
    "variance_gamma",
    "hyperbolic",
    "nig",
    "full_gh",
}
ASYMMETRIC_SELECTED = {
    "skew_normal",
    "skew_t",
    "variance_gamma",
    "hyperbolic",
    "nig",
    "full_gh",
}
METRICS = ("ari", "acc", "nmi")
COMPARISONS = (
    (ADAPTIVE, GAUSSIAN),
    (ADAPTIVE, STUDENT_T),
    (STUDENT_T, GAUSSIAN),
)


def _clean_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ci95(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().astype(float)
    n = len(values)
    if n < 2:
        return math.nan, math.nan
    sd = values.std(ddof=1)
    if not math.isfinite(sd) or sd == 0:
        mean = float(values.mean())
        return mean, mean
    sem = sd / math.sqrt(n)
    if stats is not None:
        low, high = stats.t.interval(0.95, n - 1, loc=float(values.mean()), scale=sem)
        return float(low), float(high)
    mean = float(values.mean())
    return mean - 1.96 * sem, mean + 1.96 * sem


def _p_values(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().astype(float)
    if stats is None or len(values) < 2:
        return math.nan, math.nan
    if float(values.abs().max()) == 0.0:
        return 1.0, 1.0
    t_p = float(stats.ttest_1samp(values, 0.0).pvalue)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w_p = float(stats.wilcoxon(values).pvalue)
    except ValueError:
        w_p = 1.0
    return t_p, w_p


def _paired_delta(
    ok: pd.DataFrame,
    metric: str,
    treatment: str,
    control: str,
) -> pd.DataFrame:
    keys = ["design_id", "seed", "true_family", "skew_strength"]
    left = ok.loc[ok["method"] == treatment, keys + [metric]].rename(columns={metric: "treatment_value"})
    right = ok.loc[ok["method"] == control, keys + [metric]].rename(columns={metric: "control_value"})
    paired = left.merge(right, on=keys, how="inner")
    paired["metric"] = metric
    paired["treatment"] = treatment
    paired["control"] = control
    paired["delta"] = paired["treatment_value"].astype(float) - paired["control_value"].astype(float)
    return paired


def _strata_masks(paired: pd.DataFrame) -> dict[str, pd.Series]:
    true_family = paired["true_family"].astype(str)
    design_id = paired["design_id"].astype(str)
    asym = true_family.isin(ASYMMETRIC_FAMILIES)
    return {
        "gaussian_control": design_id.eq("g_overlap"),
        "student_t_control": design_id.eq("t_overlap_df05"),
        "asymmetric_cells": asym,
        "all_cells": design_id.notna(),
    }


def _paired_summary(ok: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        for treatment, control in COMPARISONS:
            paired = _paired_delta(ok, metric, treatment, control)
            for stratum, mask in _strata_masks(paired).items():
                values = paired.loc[mask, "delta"].dropna().astype(float)
                if values.empty:
                    continue
                low, high = _ci95(values)
                t_p, wilcoxon_p = _p_values(values)
                rows.append(
                    {
                        "stratum": stratum,
                        "metric": metric,
                        "treatment": treatment,
                        "control": control,
                        "n_pairs": int(len(values)),
                        "delta_mean": float(values.mean()),
                        "delta_std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                        "delta_median": float(values.median()),
                        "delta_ci95_low": low,
                        "delta_ci95_high": high,
                        "positive_fraction": float((values > 0).mean()),
                        "material_ge_003_fraction": float((values >= 0.03).mean()),
                        "ttest_p": t_p,
                        "wilcoxon_p": wilcoxon_p,
                    }
                )
    return pd.DataFrame(rows)


def _method_summary(ok: pd.DataFrame) -> pd.DataFrame:
    grouped = ok.groupby("method", dropna=False)
    rows: list[dict[str, object]] = []
    for method, group in grouped:
        row: dict[str, object] = {"method": method, "n_runs": int(len(group))}
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method")


def _design_means(ok: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (design_id, true_family, skew_strength), group in ok.groupby(
        ["design_id", "true_family", "skew_strength"], dropna=False
    ):
        row: dict[str, object] = {
            "design_id": design_id,
            "true_family": true_family,
            "skew_strength": _clean_float(skew_strength),
        }
        means = group.groupby("method")[list(METRICS)].mean(numeric_only=True)
        for metric in METRICS:
            for method in (GAUSSIAN, STUDENT_T, ADAPTIVE):
                if method in means.index:
                    row[f"{metric}_{method}"] = float(means.loc[method, metric])
            if ADAPTIVE in means.index and GAUSSIAN in means.index:
                row[f"{metric}_adaptive_minus_vade"] = float(means.loc[ADAPTIVE, metric] - means.loc[GAUSSIAN, metric])
            if ADAPTIVE in means.index and STUDENT_T in means.index:
                row[f"{metric}_adaptive_minus_student_t"] = float(means.loc[ADAPTIVE, metric] - means.loc[STUDENT_T, metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["true_family", "design_id"])


def _selection_summary(ok: pd.DataFrame) -> pd.DataFrame:
    selector = ok.loc[ok["method"] == ADAPTIVE].copy()
    if selector.empty:
        return pd.DataFrame()
    selector["selected_is_asymmetric"] = selector["selected_family"].isin(ASYMMETRIC_SELECTED)
    selector["truth_is_asymmetric"] = selector["true_family"].isin(ASYMMETRIC_FAMILIES)
    counts = (
        selector.groupby(["true_family", "selected_family"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["true_family", "selected_family"])
    )
    totals = selector.groupby("true_family", dropna=False).size().rename("truth_total").reset_index()
    counts = counts.merge(totals, on="true_family", how="left")
    counts["within_truth_fraction"] = counts["count"] / counts["truth_total"]
    return counts


def _selection_rates(ok: pd.DataFrame) -> dict[str, object]:
    selector = ok.loc[ok["method"] == ADAPTIVE].copy()
    if selector.empty:
        return {}
    selector["selected_is_asymmetric"] = selector["selected_family"].isin(ASYMMETRIC_SELECTED)
    selector["truth_is_asymmetric"] = selector["true_family"].isin(ASYMMETRIC_FAMILIES)
    out: dict[str, object] = {
        "n_selector_runs": int(len(selector)),
        "selected_counts": {
            str(key): int(value) for key, value in selector["selected_family"].value_counts().sort_index().items()
        },
    }
    masks = {
        "asymmetric_truth": selector["truth_is_asymmetric"],
        "gaussian_control": selector["true_family"].eq("gaussian"),
        "student_t_control": selector["true_family"].eq("student_t"),
    }
    for name, mask in masks.items():
        values = selector.loc[mask, "selected_is_asymmetric"]
        out[f"{name}_selected_asymmetric_rate"] = float(values.mean()) if len(values) else math.nan
        out[f"{name}_n"] = int(len(values))
    return out


def _failure_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", dropna=False)["status"]
        .apply(lambda values: pd.Series({"n_total": len(values), "n_failed": int((values != "ok").sum())}))
        .unstack()
        .reset_index()
        .assign(failure_rate=lambda item: item["n_failed"] / item["n_total"])
        .sort_values("method")
    )


def _summary_json(
    df: pd.DataFrame,
    paired: pd.DataFrame,
    selection_rates: dict[str, object],
) -> dict[str, object]:
    out: dict[str, object] = {
        "n_rows": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_failed": int((df["status"] != "ok").sum()),
        "selection": selection_rates,
    }
    focus: dict[str, object] = {}
    for metric in METRICS:
        for stratum in ("asymmetric_cells", "all_cells", "gaussian_control"):
            row = paired.loc[
                (paired["metric"] == metric)
                & (paired["stratum"] == stratum)
                & (paired["treatment"] == ADAPTIVE)
                & (paired["control"] == STUDENT_T)
            ]
            if not row.empty:
                item = row.iloc[0]
                focus[f"{stratum}_{metric}_adaptive_minus_student_t"] = {
                    "n_pairs": int(item["n_pairs"]),
                    "mean": float(item["delta_mean"]),
                    "ci95_low": float(item["delta_ci95_low"]),
                    "ci95_high": float(item["delta_ci95_high"]),
                    "positive_fraction": float(item["positive_fraction"]),
                }
    out["focus"] = focus
    return out


def _format_float(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    return f"{x:+.{digits}f}"


def _markdown(
    paired: pd.DataFrame,
    method_summary: pd.DataFrame,
    selection_rates: dict[str, object],
    failures: pd.DataFrame,
) -> str:
    lines = [
        "# V9 Shape-Dominant Analysis",
        "",
        "## Method Summary",
        "",
        "| Method | Runs | ACC | ARI | NMI | Failure rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    failure_lookup = {str(row["method"]): float(row["failure_rate"]) for _, row in failures.iterrows()}
    for _, row in method_summary.iterrows():
        method = str(row["method"])
        lines.append(
            f"| `{method}` | {int(row['n_runs'])} | "
            f"{float(row['acc_mean']):.4f} | {float(row['ari_mean']):.4f} | "
            f"{float(row['nmi_mean']):.4f} | {failure_lookup.get(method, math.nan):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "| Stratum | Metric | Comparison | Pairs | Mean delta | 95% CI | Positive | >=0.03 |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    keep = paired.loc[paired["treatment"].isin([ADAPTIVE, STUDENT_T])].copy()
    keep = keep.loc[
        keep["stratum"].isin(["gaussian_control", "student_t_control", "asymmetric_cells", "all_cells"])
    ]
    for _, row in keep.iterrows():
        comparison = f"`{row['treatment']}` - `{row['control']}`"
        ci = f"[{_format_float(row['delta_ci95_low'])}, {_format_float(row['delta_ci95_high'])}]"
        lines.append(
            f"| {row['stratum']} | {row['metric']} | {comparison} | {int(row['n_pairs'])} | "
            f"{_format_float(row['delta_mean'])} | {ci} | "
            f"{float(row['positive_fraction']):.3f} | {float(row['material_ge_003_fraction']):.3f} |"
        )
    lines.extend(["", "## Selection", ""])
    if selection_rates:
        lines.append(f"- Selector runs: `{selection_rates.get('n_selector_runs')}`")
        lines.append(
            "- Asymmetric-truth selected-asymmetric rate: "
            f"`{100 * float(selection_rates.get('asymmetric_truth_selected_asymmetric_rate', math.nan)):.1f}%`"
        )
        lines.append(
            "- Gaussian-control selected-asymmetric rate: "
            f"`{100 * float(selection_rates.get('gaussian_control_selected_asymmetric_rate', math.nan)):.1f}%`"
        )
        lines.append(
            "- Student-t-control selected-asymmetric rate: "
            f"`{100 * float(selection_rates.get('student_t_control_selected_asymmetric_rate', math.nan)):.1f}%`"
        )
        lines.append("")
        lines.append("Selected-family counts:")
        for family, count in selection_rates.get("selected_counts", {}).items():
            lines.append(f"- `{family}`: {count}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze v9 shape-dominant synthetic results.")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_file)
    ok = df.loc[df["status"] == "ok"].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired = _paired_summary(ok)
    method_summary = _method_summary(ok)
    design_means = _design_means(ok)
    selection_summary = _selection_summary(ok)
    selection_rates = _selection_rates(ok)
    failures = _failure_summary(df)
    summary = _summary_json(df, paired, selection_rates)

    paired.to_csv(args.output_dir / "v9_pairwise_strata.csv", index=False)
    method_summary.to_csv(args.output_dir / "v9_method_summary.csv", index=False)
    design_means.to_csv(args.output_dir / "v9_design_means.csv", index=False)
    selection_summary.to_csv(args.output_dir / "v9_selection_summary.csv", index=False)
    failures.to_csv(args.output_dir / "v9_failures.csv", index=False)
    (args.output_dir / "v9_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = _markdown(paired, method_summary, selection_rates, failures)
    (args.output_dir / "v9_shape_dominant_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
