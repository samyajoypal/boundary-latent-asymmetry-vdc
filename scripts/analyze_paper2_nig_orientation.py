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
SEEDS = tuple(range(700, 720))
DESIGNS = ("student_t", "nig_boundary", "nig_outward")
REGIMES = ("unrestricted", "affine_gauge")
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
    data["fit_valid"] = (
        data["status"].eq("ok")
        & np.isfinite(score)
        & active.eq(data["fitted_component_count"])
        & data["min_expected_component_count"].ge(1.0)
    )

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


def _interval(values: pd.Series, confidence: float = 0.95) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if n > 1 else math.nan
    if n == 1:
        return {"n": n, "mean": mean, "sd": sd, "ci_low": math.nan, "ci_high": math.nan}
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    half_width = critical * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def _read(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/latent_signal/diagnostics.csv"))
    if len(files) != 120:
        raise RuntimeError(f"Expected 120 diagnostic files, found {len(files)}.")
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    expected_rows = 120 * len(FAMILIES) * len(STAGES)
    if len(data) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} diagnostic rows, found {len(data)}.")
    data["design"] = data["protocol_cell_id"].map(_design)
    data["encoder_regime"] = np.where(data["affine_gauge"], "affine_gauge", "unrestricted")
    expected = pd.MultiIndex.from_product(
        [DESIGNS, REGIMES, SEEDS, STAGES, FAMILIES],
        names=["design", "encoder_regime", "seed", "representation_stage", "family"],
    )
    counts = data.groupby(list(expected.names)).size().reindex(expected)
    if counts.isna().any() or (counts != 1).any():
        raise RuntimeError("The NIG-orientation factorial grid is incomplete or duplicated.")
    return _add_fit_validity(data)


def _comparison_rows(oracle: pd.DataFrame) -> pd.DataFrame:
    rows = oracle[
        oracle["design"].isin(("nig_boundary", "nig_outward"))
        & oracle["family"].eq("nig")
    ].copy()
    keep = [
        "design",
        "seed",
        "comparison_valid",
        *COMPARISON_METRICS,
    ]
    return rows[keep].sort_values(["design", "seed"])


def analyze(root: Path, output_dir: Path) -> dict[str, object]:
    data = _read(root)
    oracle = data[
        data["representation_stage"].eq("oracle")
        & data["encoder_regime"].eq("unrestricted")
    ].copy()
    orientation = _comparison_rows(oracle)

    boundary = orientation[orientation["design"].eq("nig_boundary")].set_index("seed")
    outward = orientation[orientation["design"].eq("nig_outward")].set_index("seed")
    primary_valid = boundary["comparison_valid"] & outward["comparison_valid"]
    primary_valid_pairs = int(primary_valid.sum())

    boundary_score = _interval(boundary.loc[boundary["comparison_valid"], "delta_log_score_vs_student_t"])
    outward_score = _interval(outward.loc[outward["comparison_valid"], "delta_log_score_vs_student_t"])
    boundary_ari = _interval(boundary.loc[boundary["comparison_valid"], "delta_ari_vs_student_t"])
    outward_ari_90 = _interval(
        outward.loc[outward["comparison_valid"], "delta_ari_vs_student_t"],
        confidence=0.90,
    )

    paired = pd.DataFrame(index=list(SEEDS))
    for metric in COMPARISON_METRICS:
        paired[f"boundary_minus_outward__{metric}"] = boundary[metric] - outward[metric]
    paired["comparison_valid"] = primary_valid
    ari_orientation = _interval(
        paired.loc[paired["comparison_valid"], "boundary_minus_outward__delta_ari_vs_student_t"]
    )
    boundary_score_orientation = _interval(
        paired.loc[
            paired["comparison_valid"],
            "boundary_minus_outward__boundary_weighted_delta_log_score_vs_student_t",
        ]
    )
    boundary_changes_orientation = _interval(
        paired.loc[
            paired["comparison_valid"],
            "boundary_minus_outward__boundary_decision_change_fraction_vs_student_t",
        ]
    )

    control = oracle[
        oracle["design"].eq("student_t")
        & oracle["family"].isin(("skew_t", "nig", "full_gh"))
    ].copy()
    valid_control_counts = control.groupby("seed")["comparison_valid"].sum()
    complete_control_seeds = valid_control_counts[valid_control_counts.eq(3)].index
    control_best = (
        control[control["seed"].isin(complete_control_seeds)]
        .groupby("seed")["delta_log_score_vs_student_t"]
        .max()
    )
    control_summary = _interval(control_best)

    invalid_fits = int((~data["fit_valid"]).sum())
    invalid_fit_rate = float(invalid_fits / len(data))
    status_failures = int((data["status"] != "ok").sum())
    equivalence_margin = 0.01

    checks = {
        "complete_primary_pairs": primary_valid_pairs == 20,
        "numerical_stability": invalid_fit_rate <= 0.05,
        "student_t_control": int(control_summary["n"]) >= 18
        and float(control_summary["mean"]) <= 0.01,
        "boundary_nig_density": int(boundary_score["n"]) == 20
        and float(boundary_score["ci_low"]) > 0.0,
        "outward_nig_density": int(outward_score["n"]) == 20
        and float(outward_score["ci_low"]) > 0.0,
        "boundary_nig_ari": int(boundary_ari["n"]) == 20
        and float(boundary_ari["ci_low"]) > 0.0,
        "orientation_ari": int(ari_orientation["n"]) == 20
        and float(ari_orientation["ci_low"]) > 0.0,
        "outward_ari_equivalence": int(outward_ari_90["n"]) == 20
        and float(outward_ari_90["ci_low"]) > -equivalence_margin
        and float(outward_ari_90["ci_high"]) < equivalence_margin,
    }

    stage_summary = (
        data[
            data["design"].isin(("nig_boundary", "nig_outward"))
            & data["family"].eq("nig")
            & data["comparison_valid"]
        ]
        .groupby(["design", "encoder_regime", "representation_stage"])[
            list(COMPARISON_METRICS)
            + ["affine_residual_from_pretrained", "knn_overlap_with_pretrained"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stage_summary.columns = [
        "_".join(value for value in column if value).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in stage_summary.columns
    ]

    gate = {
        "complete": True,
        "n_diagnostic_files": 120,
        "n_diagnostic_rows": int(len(data)),
        "primary_valid_pairs": primary_valid_pairs,
        "fit_failures": status_failures,
        "invalid_fits": invalid_fits,
        "invalid_fit_rate": invalid_fit_rate,
        "student_t_control_best_enrichment": control_summary,
        "boundary_nig_score_gain": boundary_score,
        "outward_nig_score_gain": outward_score,
        "boundary_nig_ari_gain": boundary_ari,
        "outward_nig_ari_gain_90_ci": outward_ari_90,
        "boundary_minus_outward_ari_gain": ari_orientation,
        "boundary_minus_outward_boundary_score": boundary_score_orientation,
        "boundary_minus_outward_boundary_decision_changes": boundary_changes_orientation,
        "equivalence_margin": equivalence_margin,
        "checks": checks,
    }
    gate["passed"] = bool(all(checks.values()))

    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "diagnostics_all.csv", index=False)
    orientation.to_csv(output_dir / "oracle_nig_orientation_rows.csv", index=False)
    paired.reset_index(names="seed").to_csv(
        output_dir / "oracle_nig_orientation_paired_differences.csv", index=False
    )
    stage_summary.to_csv(output_dir / "nig_stage_regime_summary.csv", index=False)
    data[~data["fit_valid"]].to_csv(output_dir / "invalid_fits.csv", index=False)
    (output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# Paper 2 NIG orientation confirmation",
        "",
        f"Overall locked gate: **{'PASS' if gate['passed'] else 'FAIL'}**.",
        "",
        f"- Valid paired oracle NIG seeds: {primary_valid_pairs}/20.",
        f"- Invalid family-stage fits: {invalid_fits}/{len(data)} ({invalid_fit_rate:.2%}).",
        f"- Student-t control best enrichment: {control_summary['mean']:+.5f} "
        f"(n={control_summary['n']}).",
        f"- Boundary NIG score gain: {boundary_score['mean']:+.5f}, 95% CI "
        f"[{boundary_score['ci_low']:+.5f}, {boundary_score['ci_high']:+.5f}].",
        f"- Outward NIG score gain: {outward_score['mean']:+.5f}, 95% CI "
        f"[{outward_score['ci_low']:+.5f}, {outward_score['ci_high']:+.5f}].",
        f"- Boundary NIG ARI gain: {boundary_ari['mean']:+.5f}, 95% CI "
        f"[{boundary_ari['ci_low']:+.5f}, {boundary_ari['ci_high']:+.5f}].",
        f"- Boundary-minus-outward ARI contrast: {ari_orientation['mean']:+.5f}, "
        f"95% CI [{ari_orientation['ci_low']:+.5f}, {ari_orientation['ci_high']:+.5f}].",
        f"- Outward NIG ARI gain: {outward_ari_90['mean']:+.5f}, 90% CI "
        f"[{outward_ari_90['ci_low']:+.5f}, {outward_ari_90['ci_high']:+.5f}].",
        "",
        "## Locked checks",
        "",
    ]
    lines.extend(
        f"- {name}: {'pass' if passed else 'fail'}" for name, passed in checks.items()
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the locked Paper 2 NIG orientation study.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/paper2_nig_orientation_confirmation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers/paper2_latent_geometry/results/nig_orientation_confirmation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = analyze(args.results_root.resolve(), args.output_dir.resolve())
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
