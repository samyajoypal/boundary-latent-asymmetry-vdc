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
EXPECTED_SEEDS = tuple(range(4200, 4220))
EXPECTED_CONFIG_SHA256 = (
    "761bba0c4c7937061f9003870417fe9fa20451a26e6f56fe90806234b02cf80a"
)


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
        manifest_path = path.parent / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        slurm = manifest.get("slurm") if isinstance(manifest.get("slurm"), dict) else {}
        manifest_fields = {
            "manifest_status": manifest.get("status"),
            "manifest_exit_code": manifest.get("exit_code"),
            "manifest_config_sha256": manifest.get("config_sha256"),
            "manifest_resolved_config_sha256": manifest.get("resolved_config_sha256"),
            "manifest_source_tree_sha256": manifest.get("source_tree_sha256"),
            "manifest_repository_commit": manifest.get("repository_commit"),
            "manifest_fmvmm_commit": manifest.get("fmvmm_commit"),
            "manifest_slurm_account": slurm.get("SLURM_JOB_ACCOUNT"),
            "manifest_slurm_partition": slurm.get("SLURM_JOB_PARTITION"),
            "manifest_slurm_job_id": slurm.get("SLURM_JOB_ID"),
        }
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict):
                rows.append({"run_dir": path.parent.name, **item, **manifest_fields})
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
        for field in (
            "cpu_process_seconds",
            "peak_rss_mb",
            "candidate_fit_and_score_seconds",
        ):
            field_mean, field_sd = _mean_sd(
                [_number(row.get(field)) for row in method_rows]
            )
            summary[f"{field}_mean"] = field_mean
            summary[f"{field}_sd"] = field_sd
        summary["failed_candidate_fits_total"] = sum(
            int(_number(row.get("n_failed_candidate_fits")))
            for row in method_rows
            if math.isfinite(_number(row.get("n_failed_candidate_fits")))
        )
        summary["rejected_prior_updates_total"] = sum(
            int(_number(row.get("n_rejected_prior_updates")))
            for row in method_rows
            if math.isfinite(_number(row.get("n_rejected_prior_updates")))
        )
        summaries.append(summary)
    return summaries


def _paired_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("method")), int(row.get("seed"))): row
        for row in rows
        if row.get("status") == "ok" and row.get("method") in METHOD_ORDER
    }
    contrasts: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for comparator in ("ghvade_mvt", "vade_diag"):
        for metric in METRICS:
            differences: list[float] = []
            seeds: list[int] = []
            for seed in EXPECTED_SEEDS:
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
            p_values[key] = p_value
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
    adjusted = _holm_adjust(p_values)
    for row in contrasts:
        key = f"{row['comparator']}:{row['metric']}"
        row["holm_p"] = adjusted[key]
    return contrasts


def _candidate_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("method") != "ghvade_adaptive" or row.get("status") != "ok":
            continue
        history = row.get("prior_update_history")
        if not isinstance(history, list):
            continue
        for event in history:
            if not isinstance(event, dict):
                continue
            selected = str(event.get("selected_candidate", ""))
            active = str(event.get("active_prior_family", ""))
            accepted = bool(event.get("prior_update_accepted"))
            candidates = event.get("candidates")
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                family = str(candidate.get("candidate", "unknown"))
                record = records.setdefault(
                    family,
                    {
                        "candidate": family,
                        "attempts": 0,
                        "successful_fits": 0,
                        "failed_fits": 0,
                        "primary_initialization_successes": 0,
                        "alternate_initialization_successes": 0,
                        "event_selections": 0,
                        "accepted_active_updates": 0,
                        "fit_seconds": [],
                    },
                )
                record["attempts"] += 1
                fit_seconds = _number(candidate.get("fit_and_score_seconds"))
                if math.isfinite(fit_seconds):
                    record["fit_seconds"].append(fit_seconds)
                if candidate.get("status") == "ok":
                    record["successful_fits"] += 1
                    diagnostics = candidate.get("fit_diagnostics")
                    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
                    attempt = str(diagnostics.get("fit_attempt", ""))
                    if attempt == "primary":
                        record["primary_initialization_successes"] += 1
                    elif attempt:
                        record["alternate_initialization_successes"] += 1
                else:
                    record["failed_fits"] += 1
                if family == selected:
                    record["event_selections"] += 1
                if accepted and family == active:
                    record["accepted_active_updates"] += 1

    output: list[dict[str, Any]] = []
    for _, record in sorted(records.items()):
        fit_mean, fit_sd = _mean_sd(record.pop("fit_seconds"))
        record["fit_seconds_mean"] = fit_mean
        record["fit_seconds_sd"] = fit_sd
        record["failure_rate"] = record["failed_fits"] / max(record["attempts"], 1)
        output.append(record)
    return output


def _selector_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("method") != "ghvade_adaptive" or row.get("status") != "ok":
            continue
        history = row.get("prior_update_history")
        history = history if isinstance(history, list) else []
        initial = history[0] if history and isinstance(history[0], dict) else {}
        final = row.get("fit_diagnostics")
        final = final if isinstance(final, dict) else {}
        boundary = final.get("shape_gate_student_t_boundary_diagnostics")
        boundary = boundary if isinstance(boundary, dict) else {}
        records.append(
            {
                "seed": row.get("seed"),
                "initial_active_family": initial.get("active_prior_family"),
                "final_active_family": row.get("selected_family"),
                "final_best_asymmetric_family": final.get(
                    "shape_gate_best_asymmetric_candidate"
                ),
                "final_asymmetric_minus_student_t_score": final.get(
                    "shape_gate_asymmetric_margin"
                ),
                "final_abs_marginal_skewness": final.get("shape_gate_abs_skew"),
                "final_shape_gate_triggered": final.get("shape_gate_triggered"),
                "final_update_accepted": final.get("prior_update_accepted"),
                "final_decision_change_fraction": boundary.get(
                    "decision_change_fraction"
                ),
                "final_boundary_disagreement_share": boundary.get(
                    "boundary_disagreement_share"
                ),
                "final_boundary_disagreement_concentration": boundary.get(
                    "boundary_disagreement_concentration"
                ),
            }
        )
    return records


def _pilot_selector_stage_rows(pilot_gate_path: Path) -> list[dict[str, Any]]:
    pilot_root = pilot_gate_path.parent.parent
    records: list[dict[str, Any]] = []
    for path in sorted(
        pilot_root.glob("pilot__dry_bean__ghvade_adaptive__seed*/results.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("external_metrics_blinded") is not True:
                raise RuntimeError(f"Pilot selector history is not blinded: {path}")
            history = item.get("prior_update_history")
            history = history if isinstance(history, list) else []
            initial = history[0] if history and isinstance(history[0], dict) else {}
            records.append(
                {
                    "seed": item.get("seed"),
                    "initial_active_family": initial.get("active_prior_family"),
                    "final_active_family": item.get("selected_family"),
                    "external_metrics_blinded": True,
                }
            )
    if len(records) != 5:
        raise RuntimeError(f"Expected five blinded pilot histories, found {len(records)}.")
    return records


def _selector_stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_rows = _selector_stage_rows(rows)
    margins = [_number(row["final_asymmetric_minus_student_t_score"]) for row in stage_rows]
    changes = [_number(row["final_decision_change_fraction"]) for row in stage_rows]
    concentrations = [
        _number(row["final_boundary_disagreement_concentration"]) for row in stage_rows
    ]
    margin_mean, margin_sd = _mean_sd(margins)
    change_mean, change_sd = _mean_sd(changes)
    concentration_mean, concentration_sd = _mean_sd(concentrations)
    finite_margins = [value for value in margins if math.isfinite(value)]
    return {
        "n_runs": len(stage_rows),
        "initial_family_counts": dict(
            sorted(Counter(str(row["initial_active_family"]) for row in stage_rows).items())
        ),
        "final_family_counts": dict(
            sorted(Counter(str(row["final_active_family"]) for row in stage_rows).items())
        ),
        "final_shape_gate_triggered": sum(
            bool(row["final_shape_gate_triggered"]) for row in stage_rows
        ),
        "final_update_accepted": sum(
            bool(row["final_update_accepted"]) for row in stage_rows
        ),
        "final_asymmetric_margin_mean": margin_mean,
        "final_asymmetric_margin_sd": margin_sd,
        "final_asymmetric_margin_min": min(finite_margins, default=math.nan),
        "final_asymmetric_margin_max": max(finite_margins, default=math.nan),
        "final_decision_change_fraction_mean": change_mean,
        "final_decision_change_fraction_sd": change_sd,
        "final_boundary_concentration_mean": concentration_mean,
        "final_boundary_concentration_sd": concentration_sd,
    }


def _numerical_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adaptive = [
        row
        for row in rows
        if row.get("method") == "ghvade_adaptive" and row.get("status") == "ok"
    ]
    runtime_values = [_number(row.get("runtime_seconds")) for row in adaptive]
    candidate_values = [
        _number(row.get("candidate_fit_and_score_seconds")) for row in adaptive
    ]
    cpu_values = [_number(row.get("cpu_process_seconds")) for row in adaptive]
    peak_values = [_number(row.get("peak_rss_mb")) for row in adaptive]
    runtime = sum(value for value in runtime_values if math.isfinite(value))
    candidate = sum(value for value in candidate_values if math.isfinite(value))
    cpu = sum(value for value in cpu_values if math.isfinite(value))
    runtime_mean, runtime_sd = _mean_sd(runtime_values)
    candidate_mean, candidate_sd = _mean_sd(candidate_values)
    peak_mean, peak_sd = _mean_sd(peak_values)
    return [
        {
            "method": "ghvade_adaptive",
            "n_runs": len(adaptive),
            "runtime_seconds_mean": runtime_mean,
            "runtime_seconds_sd": runtime_sd,
            "runtime_hours_total": runtime / 3600.0,
            "candidate_fit_seconds_mean": candidate_mean,
            "candidate_fit_seconds_sd": candidate_sd,
            "candidate_fit_share_of_wall_time": candidate / runtime if runtime else math.nan,
            "cpu_to_wall_ratio": cpu / runtime if runtime else math.nan,
            "peak_rss_mb_mean": peak_mean,
            "peak_rss_mb_sd": peak_sd,
            "failed_candidate_fits_total": sum(
                int(row.get("n_failed_candidate_fits", 0)) for row in adaptive
            ),
            "prior_updates_total": sum(int(row.get("n_prior_updates", 0)) for row in adaptive),
            "rejected_prior_updates_total": sum(
                int(row.get("n_rejected_prior_updates", 0)) for row in adaptive
            ),
            "regular_final_priors": sum(
                bool((row.get("final_prior_regularity") or {}).get("regular"))
                for row in adaptive
                if isinstance(row.get("final_prior_regularity"), dict)
            ),
        }
    ]


def _integrity_audit(
    rows: list[dict[str, Any]], expected: set[tuple[str, int]]
) -> dict[str, Any]:
    keys = [(str(row.get("method")), int(row.get("seed"))) for row in rows]
    observed = set(keys)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicate_count = len(keys) - len(observed)
    non_ok = [key for key, row in zip(keys, rows) if row.get("status") != "ok"]
    blinded = [key for key, row in zip(keys, rows) if bool(row.get("external_metrics_blinded"))]

    identity_fields = (
        "dataset_sha256",
        "dataset_hash_scope",
        "n_samples",
        "n_classes",
        "input_dim",
    )
    identity_values = {
        field: sorted({str(row.get(field)) for row in rows}) for field in identity_fields
    }
    inconsistent_identity = {
        field: values for field, values in identity_values.items() if len(values) != 1
    }
    manifest_bad = [
        key
        for key, row in zip(keys, rows)
        if row.get("manifest_status") != "completed"
        or int(row.get("manifest_exit_code", -1)) != 0
        or row.get("manifest_config_sha256") != EXPECTED_CONFIG_SHA256
    ]
    source_hashes = sorted({str(row.get("manifest_source_tree_sha256")) for row in rows})
    fmvmm_commits = sorted({str(row.get("manifest_fmvmm_commit")) for row in rows})

    if (
        missing
        or unexpected
        or duplicate_count
        or non_ok
        or blinded
        or inconsistent_identity
        or manifest_bad
        or len(source_hashes) != 1
        or len(fmvmm_commits) != 1
    ):
        raise RuntimeError(
            "The frozen result matrix failed its integrity audit: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicates={duplicate_count}, non_ok={len(non_ok)}, "
            f"blinded={len(blinded)}, identity={len(inconsistent_identity)}, "
            f"manifest={len(manifest_bad)}, source_hashes={len(source_hashes)}, "
            f"fmvmm_commits={len(fmvmm_commits)}."
        )
    return {
        "n_expected": len(expected),
        "n_rows": len(rows),
        "missing_tasks": [],
        "unexpected_tasks": [],
        "duplicate_rows": 0,
        "non_ok_rows": 0,
        "blinded_rows": 0,
        "dataset_identity": {field: values[0] for field, values in identity_values.items()},
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "source_tree_sha256": source_hashes[0],
        "fmvmm_commit": fmvmm_commits[0],
        "slurm_accounts": sorted({str(row.get("manifest_slurm_account")) for row in rows}),
        "slurm_partitions": sorted(
            {str(row.get("manifest_slurm_partition")) for row in rows}
        ),
    }


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
        default=Path("protocols/DRY_BEAN_PROTOCOL_AMENDMENT.md"),
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
    expected = {(method, seed) for method in METHOD_ORDER for seed in EXPECTED_SEEDS}
    integrity = _integrity_audit(rows, expected)

    method_summary = _method_summary(rows)
    paired = _paired_contrasts(rows)
    selection = _selection_summary(rows)
    boundary = _boundary_rows(rows)
    candidate_audit = _candidate_audit(rows)
    selector_stages = _selector_stage_rows(rows)
    pilot_selector_stages = _pilot_selector_stage_rows(args.pilot_gate)
    selector_summary = _selector_stage_summary(rows)
    numerical = _numerical_audit(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "dry_bean_results_long.csv", rows)
    _write_csv(args.output_dir / "dry_bean_method_summary.csv", method_summary)
    _write_csv(args.output_dir / "dry_bean_paired_contrasts.csv", paired)
    _write_csv(args.output_dir / "dry_bean_selection_counts.csv", selection)
    _write_csv(args.output_dir / "dry_bean_boundary_deciles.csv", boundary)
    _write_csv(args.output_dir / "dry_bean_candidate_audit.csv", candidate_audit)
    _write_csv(args.output_dir / "dry_bean_selector_stages.csv", selector_stages)
    _write_csv(
        args.output_dir / "dry_bean_pilot_selector_stages.csv",
        pilot_selector_stages,
    )
    _write_csv(args.output_dir / "dry_bean_numerical_audit.csv", numerical)
    _plot_boundary(args.output_dir / "dry_bean_boundary_margin.png", boundary)
    _plot_boundary(args.output_dir / "dry_bean_boundary_margin.pdf", boundary)

    summary = {
        "pilot_gate_passed": gate.get("gate_passed"),
        "analysis_mode": args.analysis_mode,
        **integrity,
        "method_summary": method_summary,
        "paired_contrasts": paired,
        "selection_counts": selection,
        "selector_stage_summary": selector_summary,
        "candidate_audit": candidate_audit,
        "numerical_audit": numerical,
    }
    clean = _json_clean(summary)
    (args.output_dir / "dry_bean_integrity_audit.json").write_text(
        json.dumps(_json_clean(integrity), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dry_bean_confirmatory_summary.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(clean, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
