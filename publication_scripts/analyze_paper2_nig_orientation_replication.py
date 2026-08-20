from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from scripts.analyze_paper2_nig_orientation import (
        COMPARISON_METRICS,
        FAMILIES,
        STAGES,
        _add_fit_validity,
        _design,
        _interval,
    )
except ModuleNotFoundError:  # Direct execution places ``scripts`` on sys.path.
    from analyze_paper2_nig_orientation import (
        COMPARISON_METRICS,
        FAMILIES,
        STAGES,
        _add_fit_validity,
        _design,
        _interval,
    )


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(8000, 8100))
DESIGNS = ("student_t", "nig_boundary", "nig_outward")
EXPECTED_CONFIG_SHA256 = "4f6ef3afc6e1dc3c4075d39101d15c707808584bbc5421b89c235e10889628cc"
EXPECTED_SOURCE_TREE_SHA256 = "47105f4b54b8808a5022e271b9e757d6a37775b61344a9713acf1f899b93a460"
EXPECTED_FMVMM_COMMIT = "8b0570822a2d774221ca35e01b619d3b6b08ee54"


def _audit_manifests(root: Path) -> dict[str, object]:
    files = sorted(root.glob("*/manifest.json"))
    errors: list[str] = []
    if len(files) != 300:
        errors.append(f"expected 300 run manifests, found {len(files)}")
    expected = {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
        "fmvmm_commit": EXPECTED_FMVMM_COMMIT,
        "status": "completed",
        "exit_code": 0,
    }
    for path in files:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                errors.append(
                    f"{path}: {key}={manifest.get(key)!r}, expected {expected_value!r}"
                )
    audit = {
        "n_manifests": len(files),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256,
        "fmvmm_commit": EXPECTED_FMVMM_COMMIT,
        "errors": errors,
        "passed": not errors,
    }
    if errors:
        raise RuntimeError(f"Replication provenance audit failed: {errors[0]}")
    return audit


def _read(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/latent_signal/diagnostics.csv"))
    if len(files) != 300:
        raise RuntimeError(f"Expected 300 diagnostic files, found {len(files)}.")
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    expected_rows = 300 * len(FAMILIES) * len(STAGES)
    if len(data) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} diagnostic rows, found {len(data)}.")
    data["design"] = data["protocol_cell_id"].map(_design)
    expected = pd.MultiIndex.from_product(
        [DESIGNS, SEEDS, STAGES, FAMILIES],
        names=["design", "seed", "representation_stage", "family"],
    )
    counts = data.groupby(list(expected.names)).size().reindex(expected)
    if counts.isna().any() or (counts != 1).any():
        raise RuntimeError("The independent replication grid is incomplete or duplicated.")
    return _add_fit_validity(data)


def _greater_test(values: pd.Series) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return {"n": int(array.size), "mean": math.nan, "p_one_sided": math.nan}
    test = stats.ttest_1samp(array, 0.0, alternative="greater")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "p_one_sided": float(test.pvalue),
    }


def _holm_adjust(tests: dict[str, dict[str, float | int]]) -> dict[str, float]:
    ordered = sorted(
        ((name, float(result["p_one_sided"])) for name, result in tests.items()),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[name] = running
    return adjusted


def analyze(root: Path, output_dir: Path) -> dict[str, object]:
    provenance = _audit_manifests(root)
    data = _read(root)
    oracle = data[data["representation_stage"].eq("oracle")].copy()

    nig = oracle[
        oracle["design"].isin(("nig_boundary", "nig_outward"))
        & oracle["family"].eq("nig")
    ].copy()
    keep = ["design", "seed", "comparison_valid", *COMPARISON_METRICS]
    nig = nig[keep].sort_values(["design", "seed"])
    boundary = nig[nig["design"].eq("nig_boundary")].set_index("seed")
    outward = nig[nig["design"].eq("nig_outward")].set_index("seed")
    paired_valid = boundary["comparison_valid"] & outward["comparison_valid"]

    paired = pd.DataFrame(index=list(SEEDS))
    for metric in COMPARISON_METRICS:
        paired[f"boundary_minus_outward__{metric}"] = boundary[metric] - outward[metric]
    paired["comparison_valid"] = paired_valid

    boundary_score_values = boundary.loc[
        boundary["comparison_valid"], "delta_log_score_vs_student_t"
    ]
    boundary_ari_values = boundary.loc[
        boundary["comparison_valid"], "delta_ari_vs_student_t"
    ]
    orientation_ari_values = paired.loc[
        paired["comparison_valid"], "boundary_minus_outward__delta_ari_vs_student_t"
    ]
    outward_ari_values = outward.loc[
        outward["comparison_valid"], "delta_ari_vs_student_t"
    ]

    boundary_score = _interval(boundary_score_values)
    boundary_ari = _interval(boundary_ari_values)
    orientation_ari = _interval(orientation_ari_values)
    outward_ari_90 = _interval(outward_ari_values, confidence=0.90)
    orientation_boundary_score = _interval(
        paired.loc[
            paired["comparison_valid"],
            "boundary_minus_outward__boundary_weighted_delta_log_score_vs_student_t",
        ]
    )
    orientation_boundary_changes = _interval(
        paired.loc[
            paired["comparison_valid"],
            "boundary_minus_outward__boundary_decision_change_fraction_vs_student_t",
        ]
    )

    confirmatory_tests = {
        "orientation_ari": _greater_test(orientation_ari_values),
        "boundary_ari": _greater_test(boundary_ari_values),
        "boundary_density": _greater_test(boundary_score_values),
    }
    adjusted = _holm_adjust(confirmatory_tests)
    for name, adjusted_p in adjusted.items():
        confirmatory_tests[name]["p_holm"] = adjusted_p

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
    valid_pairs = int(paired_valid.sum())
    equivalence_margin = 0.01

    checks = {
        "provenance": bool(provenance["passed"]),
        "complete_primary_pairs": valid_pairs >= 95,
        "numerical_stability": invalid_fit_rate <= 0.05,
        "student_t_control": int(control_summary["n"]) >= 90
        and float(control_summary["mean"]) <= 0.01,
        "orientation_ari_ci": float(orientation_ari["ci_low"]) > 0.0,
        "confirmatory_holm_family": all(
            float(result["mean"]) > 0.0 and float(result["p_holm"]) < 0.05
            for result in confirmatory_tests.values()
        ),
        "outward_ari_equivalence": float(outward_ari_90["ci_low"])
        > -equivalence_margin
        and float(outward_ari_90["ci_high"]) < equivalence_margin,
    }

    gate: dict[str, object] = {
        "complete": True,
        "n_diagnostic_files": 300,
        "n_diagnostic_rows": int(len(data)),
        "primary_valid_pairs": valid_pairs,
        "complete_control_seeds": int(len(complete_control_seeds)),
        "fit_failures": status_failures,
        "invalid_fits": invalid_fits,
        "invalid_fit_rate": invalid_fit_rate,
        "student_t_control_best_enrichment": control_summary,
        "boundary_nig_score_gain": boundary_score,
        "boundary_nig_ari_gain": boundary_ari,
        "boundary_minus_outward_ari_gain": orientation_ari,
        "outward_nig_ari_gain_90_ci": outward_ari_90,
        "boundary_minus_outward_boundary_score": orientation_boundary_score,
        "boundary_minus_outward_boundary_decision_changes": orientation_boundary_changes,
        "confirmatory_tests": confirmatory_tests,
        "equivalence_margin": equivalence_margin,
        "checks": checks,
    }
    gate["passed"] = bool(all(checks.values()))

    stage_summary = (
        data[
            data["design"].isin(("nig_boundary", "nig_outward"))
            & data["family"].eq("nig")
            & data["comparison_valid"]
        ]
        .groupby(["design", "representation_stage"])[list(COMPARISON_METRICS)]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stage_summary.columns = [
        "_".join(value for value in column if value).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in stage_summary.columns
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance_audit.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    data.to_csv(output_dir / "diagnostics_all.csv", index=False)
    nig.to_csv(output_dir / "oracle_nig_orientation_rows.csv", index=False)
    paired.reset_index(names="seed").to_csv(
        output_dir / "oracle_nig_orientation_paired_differences.csv", index=False
    )
    stage_summary.to_csv(output_dir / "nig_stage_summary.csv", index=False)
    data[~data["fit_valid"]].to_csv(output_dir / "invalid_fits.csv", index=False)
    (output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# Paper 2 independent NIG-orientation replication",
        "",
        f"Overall locked gate: **{'PASS' if gate['passed'] else 'FAIL'}**.",
        "",
        f"- Valid paired seeds: {valid_pairs}/100.",
        f"- Invalid family-stage fits: {invalid_fits}/{len(data)} ({invalid_fit_rate:.2%}).",
        f"- Student-t control best enrichment: {control_summary['mean']:+.5f} "
        f"(n={control_summary['n']}).",
        f"- Boundary NIG score gain: {boundary_score['mean']:+.5f}, 95% CI "
        f"[{boundary_score['ci_low']:+.5f}, {boundary_score['ci_high']:+.5f}].",
        f"- Boundary NIG ARI gain: {boundary_ari['mean']:+.5f}, 95% CI "
        f"[{boundary_ari['ci_low']:+.5f}, {boundary_ari['ci_high']:+.5f}].",
        f"- Boundary-minus-outward ARI: {orientation_ari['mean']:+.5f}, 95% CI "
        f"[{orientation_ari['ci_low']:+.5f}, {orientation_ari['ci_high']:+.5f}].",
        f"- Outward NIG ARI: {outward_ari_90['mean']:+.5f}, 90% CI "
        f"[{outward_ari_90['ci_low']:+.5f}, {outward_ari_90['ci_high']:+.5f}].",
        "",
        "## Holm-adjusted confirmatory tests",
        "",
    ]
    lines.extend(
        f"- {name}: mean={result['mean']:+.5f}, one-sided p={result['p_one_sided']:.6g}, "
        f"Holm p={result['p_holm']:.6g}"
        for name, result in confirmatory_tests.items()
    )
    lines.extend(["", "## Locked checks", ""])
    lines.extend(
        f"- {name}: {'pass' if passed else 'fail'}" for name, passed in checks.items()
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the independent Paper 2 NIG-orientation replication."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/paper2_nig_orientation_replication",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "papers/paper2_latent_geometry/results/nig_orientation_replication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = analyze(args.results_root.resolve(), args.output_dir.resolve())
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
