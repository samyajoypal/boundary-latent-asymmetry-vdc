from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("gaussian", "student_t", "skew_t", "nig", "full_gh")
STAGES = ("oracle", "pretrained", "final")
COMPARISON_METRICS = (
    "delta_log_score_vs_student_t",
    "boundary_weighted_delta_log_score_vs_student_t",
    "decision_change_fraction_vs_student_t",
    "boundary_decision_change_fraction_vs_student_t",
    "delta_acc_vs_student_t",
    "delta_ari_vs_student_t",
    "delta_nmi_vs_student_t",
)


def _design(cell_id: str) -> str:
    for suffix in ("_unrestricted", "_affine_gauge"):
        if cell_id.endswith(suffix):
            return cell_id.removesuffix(suffix)
    raise ValueError(f"Unknown encoder suffix in {cell_id!r}.")


def _interval(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    n = int(len(array))
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - critical * se if n > 1 else math.nan,
        "ci_high": mean + critical * se if n > 1 else math.nan,
    }


def _parameter_diagnostics(value: object) -> tuple[int, float]:
    try:
        summary = json.loads(str(value))
        weights = np.asarray(summary.get("weights", []), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0, math.nan
    if weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all():
        return int(weights.size), math.nan
    return int(weights.size), float(np.min(weights))


def _add_fit_validity(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    parsed = data["prior_parameters_json"].map(_parameter_diagnostics)
    data["fitted_component_count"] = parsed.map(lambda value: value[0])
    data["min_component_weight"] = parsed.map(lambda value: value[1])
    data["min_expected_component_count"] = (
        data["min_component_weight"] * pd.to_numeric(data["n_fit"], errors="coerce")
    )
    active = pd.to_numeric(data["n_assigned"], errors="coerce")
    score = pd.to_numeric(data["heldout_log_score_mean"], errors="coerce")
    data["active_predicted_components"] = active
    data["fit_valid"] = (
        data["status"].eq("ok")
        & np.isfinite(score)
        & active.eq(data["fitted_component_count"])
        & data["min_expected_component_count"].ge(1.0)
    )

    reasons: list[str] = []
    for row in data.itertuples(index=False):
        row_reasons: list[str] = []
        if row.status != "ok":
            row_reasons.append("fit_status_not_ok")
        if not np.isfinite(float(row.heldout_log_score_mean)):
            row_reasons.append("nonfinite_heldout_score")
        if int(row.fitted_component_count) <= 0:
            row_reasons.append("missing_component_weights")
        if float(row.active_predicted_components) != float(row.fitted_component_count):
            row_reasons.append("not_all_components_receive_map_assignments")
        if not np.isfinite(float(row.min_expected_component_count)):
            row_reasons.append("invalid_component_weights")
        elif float(row.min_expected_component_count) < 1.0:
            row_reasons.append("component_has_less_than_one_expected_fit_observation")
        reasons.append(";".join(row_reasons))
    data["fit_validity_reason"] = reasons

    keys = ["protocol_cell_id", "seed", "representation_stage"]
    reference = data[data["family"] == "student_t"][keys + ["fit_valid"]].rename(
        columns={"fit_valid": "student_t_reference_fit_valid"}
    )
    if reference.duplicated(keys).any():
        raise RuntimeError("Student-t reference rows are duplicated.")
    data = data.merge(reference, on=keys, how="left", validate="many_to_one")
    data["comparison_valid"] = (
        data["fit_valid"] & data["student_t_reference_fit_valid"].fillna(False)
    )
    for metric in COMPARISON_METRICS:
        data[f"raw_{metric}"] = data[metric]
        data.loc[~data["comparison_valid"], metric] = math.nan
    return data


def _read(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/latent_signal/diagnostics.csv"))
    if len(files) != 40:
        raise RuntimeError(f"Expected 40 diagnostic files, found {len(files)}.")
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    if len(data) != 40 * len(FAMILIES) * len(STAGES):
        raise RuntimeError(f"Expected 600 diagnostic rows, found {len(data)}.")
    data["design"] = data["protocol_cell_id"].map(_design)
    data["encoder_regime"] = np.where(data["affine_gauge"], "affine_gauge", "unrestricted")
    expected = pd.MultiIndex.from_product(
        [sorted(data["design"].unique()), ("unrestricted", "affine_gauge"), range(500, 505), STAGES, FAMILIES],
        names=["design", "encoder_regime", "seed", "representation_stage", "family"],
    )
    counts = data.groupby(list(expected.names)).size().reindex(expected)
    if counts.isna().any() or (counts != 1).any():
        raise RuntimeError("The latent-geometry factorial grid is incomplete or duplicated.")
    return _add_fit_validity(data)


def analyze(root: Path, output_dir: Path) -> dict[str, object]:
    data = _read(root)
    numeric = [
        "heldout_log_score_mean",
        "delta_log_score_vs_student_t",
        "boundary_weighted_delta_log_score_vs_student_t",
        "decision_change_fraction_vs_student_t",
        "boundary_decision_change_fraction_vs_student_t",
        "delta_acc_vs_student_t",
        "delta_ari_vs_student_t",
        "delta_nmi_vs_student_t",
        "affine_residual_from_oracle",
        "affine_residual_from_pretrained",
        "knn_overlap_with_oracle",
        "knn_overlap_with_pretrained",
        "fit_seconds",
    ]
    summary = data[data["fit_valid"]].groupby(
        ["design", "encoder_regime", "representation_stage", "family"]
    )[numeric].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(value for value in column if value).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    # Oracle latents are generated identically for the two encoder regimes.
    oracle = data[
        (data["representation_stage"] == "oracle")
        & (data["encoder_regime"] == "unrestricted")
    ].copy()
    matching_family = {
        "student_t": "student_t",
        "nig_boundary": "nig",
        "full_gh_boundary": "full_gh",
        "skew_t_outward": "skew_t",
    }
    matching = pd.concat(
        [
            oracle[(oracle["design"] == design) & (oracle["family"] == family)]
            for design, family in matching_family.items()
        ],
        ignore_index=True,
    )
    matching = matching[matching["comparison_valid"]]

    matching_rows: list[dict[str, object]] = []
    for design, group in matching.groupby("design"):
        for metric in COMPARISON_METRICS:
            matching_rows.append({"design": design, "metric": metric, **_interval(group[metric])})
    matching_summary = pd.DataFrame(matching_rows)

    control = oracle[oracle["design"] == "student_t"]
    control_enriched = control[
        control["family"].isin(("skew_t", "nig", "full_gh"))
        & control["comparison_valid"]
    ]
    best_control_gain = (
        control_enriched.groupby("seed")["delta_log_score_vs_student_t"].max()
    )
    control_gain = float(best_control_gain.mean())
    control_pass = control_gain <= 0.01

    boundary_details: dict[str, dict[str, float | bool]] = {}
    boundary_passes: list[bool] = []
    for design, family in (("nig_boundary", "nig"), ("full_gh_boundary", "full_gh")):
        rows = oracle[
            (oracle["design"] == design)
            & (oracle["family"] == family)
            & oracle["comparison_valid"]
        ]
        score = float(rows["delta_log_score_vs_student_t"].mean())
        changes = float(rows["boundary_decision_change_fraction_vs_student_t"].mean())
        passed = score > 0.0 and changes > 0.0
        boundary_details[design] = {
            "mean_score_gain": score,
            "mean_boundary_decision_change": changes,
            "passed": passed,
        }
        boundary_passes.append(passed)
    boundary_pass = any(boundary_passes)

    outward = oracle[
        (oracle["design"] == "skew_t_outward")
        & (oracle["family"] == "skew_t")
        & oracle["comparison_valid"]
    ]
    outward_score = float(outward["delta_log_score_vs_student_t"].mean())
    outward_ari = float(outward["delta_ari_vs_student_t"].mean())
    outward_pass = outward_score > 0.0 and outward_ari <= 0.005

    drift_rows = data[
        (data["design"].isin(("nig_boundary", "full_gh_boundary")))
        & (data["representation_stage"] == "final")
        & (
            ((data["design"] == "nig_boundary") & (data["family"] == "nig"))
            | ((data["design"] == "full_gh_boundary") & (data["family"] == "full_gh"))
        )
        & data["comparison_valid"]
    ]
    drift_summary = drift_rows.groupby(["design", "encoder_regime"])[
        [
            "delta_log_score_vs_student_t",
            "delta_ari_vs_student_t",
            "affine_residual_from_pretrained",
            "knn_overlap_with_pretrained",
        ]
    ].mean().reset_index()
    unrestricted_residual = float(
        drift_summary.loc[
            drift_summary["encoder_regime"] == "unrestricted",
            "affine_residual_from_pretrained",
        ].mean()
    )
    gauge_residual = float(
        drift_summary.loc[
            drift_summary["encoder_regime"] == "affine_gauge",
            "affine_residual_from_pretrained",
        ].mean()
    )
    oracle_boundary_score = max(
        value["mean_score_gain"] for value in boundary_details.values()
    )
    final_unrestricted_score = float(
        drift_summary.loc[
            drift_summary["encoder_regime"] == "unrestricted",
            "delta_log_score_vs_student_t",
        ].max()
    )
    drift_pass = bool(
        unrestricted_residual > gauge_residual + 1e-6
        and final_unrestricted_score < oracle_boundary_score
    )

    status_failures = int((data["status"] != "ok").sum())
    invalid_fits = int((~data["fit_valid"]).sum())
    invalid_comparisons = int((~data["comparison_valid"]).sum())
    invalid_fit_rate = float(invalid_fits / len(data))
    numerical_pass = invalid_fit_rate <= 0.05
    gate = {
        "complete": True,
        "fit_failures": status_failures,
        "student_t_control": {
            "mean_best_enrichment_score_gain": control_gain,
            "competitive_margin": 0.01,
            "passed": control_pass,
        },
        "boundary_signal": {
            "designs": boundary_details,
            "passed": boundary_pass,
        },
        "outward_separation": {
            "mean_score_gain": outward_score,
            "mean_ari_gain": outward_ari,
            "family_matched_to_passing_boundary_design": False,
            "passed": outward_pass,
        },
        "representation_drift": {
            "mean_unrestricted_affine_residual_from_pretrained": unrestricted_residual,
            "mean_gauge_affine_residual_from_pretrained": gauge_residual,
            "best_oracle_boundary_score_gain": oracle_boundary_score,
            "best_final_unrestricted_score_gain": final_unrestricted_score,
            "passed": drift_pass,
        },
        "numerical_stability": {
            "status_failures": status_failures,
            "invalid_fits": invalid_fits,
            "invalid_fit_rate": invalid_fit_rate,
            "invalid_comparisons": invalid_comparisons,
            "maximum_invalid_fit_rate": 0.05,
            "passed": numerical_pass,
        },
    }
    gate["passed"] = bool(
        control_pass and boundary_pass and outward_pass and drift_pass and numerical_pass
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "diagnostics_all.csv", index=False)
    summary.to_csv(output_dir / "family_stage_summary.csv", index=False)
    matching_summary.to_csv(output_dir / "oracle_matching_family_summary.csv", index=False)
    drift_summary.to_csv(output_dir / "representation_drift_summary.csv", index=False)
    data[~data["fit_valid"]].to_csv(output_dir / "invalid_fits.csv", index=False)
    (output_dir / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Paper 2 latent-geometry pilot",
        "",
        f"Overall gate: **{'PASS' if gate['passed'] else 'FAIL'}**.",
        "",
        f"- Student-t control: {'pass' if control_pass else 'fail'} "
        f"(best enrichment gain {control_gain:+.5f} nats/observation).",
        f"- Boundary signal: {'pass' if boundary_pass else 'fail'}.",
    ]
    for design, values in boundary_details.items():
        lines.append(
            f"  - {design}: score {values['mean_score_gain']:+.5f}, "
            f"boundary changes {values['mean_boundary_decision_change']:.4f}."
        )
    lines.extend(
        [
            f"- Outward separation: {'pass' if outward_pass else 'fail'} "
            f"(score {outward_score:+.5f}, ARI {outward_ari:+.5f}).",
            "  The outward cell is skew-t whereas the passing boundary cell is NIG; "
            "it is not a family-matched orientation control.",
            f"- Representation drift: {'pass' if drift_pass else 'fail'} "
            f"(unrestricted residual {unrestricted_residual:.6f}, "
            f"gauge residual {gauge_residual:.6f}).",
            f"- Numerical stability: {'pass' if numerical_pass else 'fail'} "
            f"({invalid_fits}/{len(data)} invalid mixture fits; "
            f"{status_failures} exception-level failures).",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Paper 2 stop-go pilot.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/paper2_latent_geometry_pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers/paper2_latent_geometry/results/pilot_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = analyze(args.results_root.resolve(), args.output_dir.resolve())
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
