from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional reporting path
    stats = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.realdata_protocol import DEFAULT_PROTOCOL, RealdataTask, load_tasks


DEFAULT_CONTROLS = (
    "raw_kmeans",
    "raw_gmm",
    "dec",
    "idec",
    "vae_gmm",
    "vade_diag",
    "conv_vade_diag",
    "conv_vade_diag_mi",
    "ghvade_trust_mvt",
    "ghvade_trust_mvt_mi",
    "conv_ghvade_trust_mvt",
    "conv_ghvade_trust_mvt_mi",
)
DEFAULT_TREATMENTS = (
    "ghvade_mvt",
    "ghvade_tail_select",
    "ghvade_conservative",
    "ghvade_gh_family",
    "ghvade_adaptive",
    "ghvade_guarded_adaptive",
    "conv_ghvade_mvt",
    "conv_ghvade_mvt_mi",
    "conv_ghvade_mvsn",
    "conv_ghvade_mvvg",
    "conv_ghvade_mghp",
    "conv_ghvade_tail_select",
    "conv_ghvade_tail_select_mi",
    "conv_ghvade_conservative",
    "conv_ghvade_conservative_mi",
    "conv_ghvade_adaptive",
    "conv_ghvade_adaptive_mi",
    "conv_ghvade_guarded_adaptive",
    "conv_ghvade_guarded_adaptive_mi",
    "ghvade_trust_component_adaptive",
    "ghvade_trust_component_adaptive_mi",
    "conv_ghvade_trust_component_adaptive",
    "conv_ghvade_trust_component_adaptive_mi",
)
METRICS = ("acc", "nmi", "ari", "purity")


def _read_manifest(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect(tasks: list[RealdataTask], output_root: Path) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    rows: list[dict[str, str]] = []
    incomplete: list[dict[str, object]] = []
    for task in tasks:
        run_dir = output_root / task.run_id
        manifest = _read_manifest(run_dir)
        result_path = run_dir / "results.csv"
        if manifest is None or manifest.get("status") != "completed" or not result_path.exists():
            incomplete.append(
                {
                    "task_id": task.task_id,
                    "run_id": task.run_id,
                    "manifest_status": None if manifest is None else manifest.get("status"),
                    "result_exists": result_path.exists(),
                }
            )
            continue
        with result_path.open(newline="", encoding="utf-8") as handle:
            task_rows = list(csv.DictReader(handle))
        if len(task_rows) != 1:
            incomplete.append(
                {
                    "task_id": task.task_id,
                    "run_id": task.run_id,
                    "manifest_status": "invalid_result_count",
                    "n_rows": len(task_rows),
                }
            )
            continue
        row = task_rows[0]
        if (
            row.get("dataset") != task.dataset
            or row.get("method") != task.method
            or int(row.get("seed", -1)) != task.seed
        ):
            incomplete.append(
                {
                    "task_id": task.task_id,
                    "run_id": task.run_id,
                    "manifest_status": "task_identity_mismatch",
                }
            )
            continue
        row["protocol_run_id"] = task.run_id
        row["protocol_stage"] = task.stage
        row["protocol_cell_id"] = task.cell_id
        rows.append(row)
    return rows, incomplete


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _method_group(method: str) -> str:
    if method in {
        "raw_kmeans",
        "raw_gmm",
        "dec",
        "idec",
        "vae_kmeans",
        "vae_gmm",
        "vade_diag",
        "conv_vade_diag",
        "conv_vade_diag_mi",
    }:
        return "baseline"
    if method.startswith("conv_ghvade_") or method.startswith("ghvade_"):
        return "proposed"
    return "other"


def _cell_id(row: dict[str, str]) -> str:
    return row.get("protocol_cell_id") or "default"


def _summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault((_cell_id(row), row["dataset"], row["method"]), []).append(row)

    out: list[dict[str, object]] = []
    for (cell_id, dataset, method), group in sorted(grouped.items()):
        item: dict[str, object] = {
            "protocol_cell_id": cell_id,
            "dataset": dataset,
            "method": method,
            "group": _method_group(method),
            "n_runs": len(group),
        }
        for metric in METRICS:
            values = [value for row in group if (value := _float(row, metric)) is not None]
            if values:
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def _paired_delta(
    rows: list[dict[str, str]],
    dataset: str,
    treatment: str,
    control: str,
    metric: str,
    protocol_cell_id: str | None = None,
) -> dict[str, object]:
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        cell_id = _cell_id(row)
        if (
            row.get("status") == "ok"
            and row.get("dataset") == dataset
            and row.get("method") in {treatment, control}
            and (protocol_cell_id is None or cell_id == protocol_cell_id)
        ):
            by_key[(cell_id, row["method"], row["seed"])] = row

    deltas: list[float] = []
    pair_keys = sorted({(cell_id, seed) for cell_id, _, seed in by_key}, key=lambda item: (item[0], int(item[1])))
    for cell_id, seed in pair_keys:
        treatment_row = by_key.get((cell_id, treatment, seed))
        control_row = by_key.get((cell_id, control, seed))
        if treatment_row is None or control_row is None:
            continue
        treatment_value = _float(treatment_row, metric)
        control_value = _float(control_row, metric)
        if treatment_value is not None and control_value is not None:
            deltas.append(treatment_value - control_value)

    observed_cells = sorted({cell_id for cell_id, _, _ in by_key})
    output_cell_id = protocol_cell_id or (observed_cells[0] if len(observed_cells) == 1 else "all")
    out: dict[str, object] = {
        "protocol_cell_id": output_cell_id,
        "dataset": dataset,
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "n_pairs": len(deltas),
    }
    if deltas:
        out[f"delta_{metric}_mean"] = mean(deltas)
        out[f"delta_{metric}_std"] = stdev(deltas) if len(deltas) > 1 else 0.0
        out["wins"] = sum(1 for value in deltas if value > 0)
        out["ties"] = sum(1 for value in deltas if value == 0)
        out["losses"] = sum(1 for value in deltas if value < 0)
        if len(deltas) > 1 and max(deltas) - min(deltas) <= 1e-12:
            out[f"delta_{metric}_ci95_low"] = mean(deltas)
            out[f"delta_{metric}_ci95_high"] = mean(deltas)
            out["ttest_p"] = 1.0 if deltas[0] == 0 else 0.0
            out["wilcoxon_p"] = (
                1.0
                if deltas[0] == 0
                else float(stats.wilcoxon(deltas).pvalue) if stats is not None else math.nan
            )
        elif len(deltas) > 1 and stats is not None:
            sem = stats.sem(deltas)
            if sem > 0:
                ci = stats.t.interval(0.95, len(deltas) - 1, loc=mean(deltas), scale=sem)
                out[f"delta_{metric}_ci95_low"] = float(ci[0])
                out[f"delta_{metric}_ci95_high"] = float(ci[1])
            else:
                out[f"delta_{metric}_ci95_low"] = mean(deltas)
                out[f"delta_{metric}_ci95_high"] = mean(deltas)
            out["ttest_p"] = float(stats.ttest_rel(deltas, [0.0] * len(deltas)).pvalue)
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(deltas).pvalue)
            except ValueError:
                out["wilcoxon_p"] = math.nan
    return out


def _paired_comparisons(
    rows: list[dict[str, str]],
    treatments: tuple[str, ...],
    controls: tuple[str, ...],
) -> list[dict[str, object]]:
    cells = sorted(
        {
            (_cell_id(row), row.get("dataset", ""))
            for row in rows
            if row.get("status") == "ok"
        }
    )
    out: list[dict[str, object]] = []
    for cell_id, dataset in cells:
        methods = {
            row.get("method", "")
            for row in rows
            if row.get("status") == "ok"
            and _cell_id(row) == cell_id
            and row.get("dataset", "") == dataset
        }
        out.extend(
            _paired_delta(rows, dataset, treatment, control, metric, cell_id)
            for treatment in treatments
            for control in controls
            for metric in ("acc", "ari", "nmi")
            if treatment in methods and control in methods
        )
    return out


def _failure_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (_cell_id(row), row.get("dataset", ""), row.get("method", "")), []
        ).append(row)
    out: list[dict[str, object]] = []
    for (cell_id, dataset, method), group in sorted(grouped.items()):
        n_failed = sum(1 for row in group if row.get("status") != "ok")
        if n_failed:
            out.append(
                {
                    "protocol_cell_id": cell_id,
                    "dataset": dataset,
                    "method": method,
                    "n_failed": n_failed,
                    "n_total": len(group),
                    "failure_rate": n_failed / max(len(group), 1),
                }
            )
    return out


def _best_by_dataset(summary: list[dict[str, object]], min_runs: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    cells = sorted(
        {
            (str(item.get("protocol_cell_id", "default")), str(item["dataset"]))
            for item in summary
        }
    )
    for cell_id, dataset in cells:
        candidates = [
            item
            for item in summary
            if item["dataset"] == dataset
            and str(item.get("protocol_cell_id", "default")) == cell_id
            and int(item.get("n_runs", 0)) >= min_runs
            and "acc_mean" in item
        ]
        baselines = [item for item in candidates if item.get("group") == "baseline"]
        proposed = [item for item in candidates if item.get("group") == "proposed"]
        best_overall = max(candidates, key=lambda item: float(item["acc_mean"])) if candidates else None
        best_baseline = max(baselines, key=lambda item: float(item["acc_mean"])) if baselines else None
        best_proposed = max(proposed, key=lambda item: float(item["acc_mean"])) if proposed else None
        item: dict[str, object] = {"protocol_cell_id": cell_id, "dataset": dataset}
        if best_overall:
            item["best_overall_method"] = best_overall["method"]
            item["best_overall_acc_mean"] = best_overall["acc_mean"]
        if best_baseline:
            item["best_baseline_method"] = best_baseline["method"]
            item["best_baseline_acc_mean"] = best_baseline["acc_mean"]
        if best_proposed:
            item["best_proposed_method"] = best_proposed["method"]
            item["best_proposed_acc_mean"] = best_proposed["acc_mean"]
        if best_baseline and best_proposed:
            item["proposed_minus_baseline_acc_mean"] = float(best_proposed["acc_mean"]) - float(best_baseline["acc_mean"])
        out.append(item)
    return out


def _write_csv(
    path: Path,
    rows: list[dict[str, object] | dict[str, str]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate frozen real-data image benchmark tasks.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "realdata_image_pilot_v1")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--min-runs", type=int, default=None)
    parser.add_argument("--controls", nargs="+", default=list(DEFAULT_CONTROLS))
    parser.add_argument("--treatments", nargs="+", default=list(DEFAULT_TREATMENTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    tasks = load_tasks(protocol_path, args.stage)
    aggregate_dir = args.output_root / f"aggregate_{args.stage}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    rows, incomplete = _collect(tasks, args.output_root)

    _write_csv(aggregate_dir / "raw_results.csv", rows)
    (aggregate_dir / "incomplete_tasks.json").write_text(
        json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
    )

    min_runs = args.min_runs if args.min_runs is not None else (3 if args.stage == "pilot" else 10)
    summary = _summary(rows)
    comparisons = _paired_comparisons(rows, tuple(args.treatments), tuple(args.controls))
    failures = _failure_summary(rows)
    best = _best_by_dataset(summary, min_runs=min_runs)
    _write_csv(aggregate_dir / "realdata_method_summary.csv", summary)
    _write_csv(
        aggregate_dir / "realdata_paired_comparisons.csv",
        comparisons,
        fieldnames=[
            "control",
            "dataset",
            "delta_acc_ci95_high",
            "delta_acc_ci95_low",
            "delta_acc_mean",
            "delta_acc_std",
            "delta_ari_ci95_high",
            "delta_ari_ci95_low",
            "delta_ari_mean",
            "delta_ari_std",
            "delta_nmi_ci95_high",
            "delta_nmi_ci95_low",
            "delta_nmi_mean",
            "delta_nmi_std",
            "losses",
            "metric",
            "n_pairs",
            "protocol_cell_id",
            "ties",
            "treatment",
            "ttest_p",
            "wilcoxon_p",
            "wins",
        ],
    )
    _write_csv(
        aggregate_dir / "realdata_failures.csv",
        failures,
        fieldnames=["dataset", "failure_rate", "method", "n_failed", "n_total", "protocol_cell_id"],
    )
    _write_csv(aggregate_dir / "realdata_best_by_dataset.csv", best)

    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    aggregate_manifest = {
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "stage": args.stage,
        "n_expected": len(tasks),
        "n_collected": len(rows),
        "n_incomplete": len(incomplete),
        "min_runs": min_runs,
        "summary_files": {
            "raw_results": str(aggregate_dir / "raw_results.csv"),
            "method_summary": str(aggregate_dir / "realdata_method_summary.csv"),
            "paired_comparisons": str(aggregate_dir / "realdata_paired_comparisons.csv"),
            "failures": str(aggregate_dir / "realdata_failures.csv"),
            "best_by_dataset": str(aggregate_dir / "realdata_best_by_dataset.csv"),
        },
    }
    (aggregate_dir / "aggregation_manifest.json").write_text(
        json.dumps(aggregate_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(aggregate_manifest, indent=2, sort_keys=True))
    if incomplete and not args.allow_incomplete:
        raise SystemExit("Protocol is incomplete; inspect incomplete_tasks.json.")


if __name__ == "__main__":
    main()
