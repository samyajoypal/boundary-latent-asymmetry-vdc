from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_realdata_protocol import _collect, _paired_delta, _write_csv
from scripts.realdata_protocol import load_tasks


PRIMARY = "conv_ghvade_adaptive"
METRICS = ("acc", "ari", "nmi")
DEFAULT_PROTOCOLS = tuple(
    ROOT / "configs" / name
    for name in (
        "realdata_final_v9_clean.toml",
        "realdata_final_v9_clean_auxiliary.toml",
        "realdata_final_v9_uniform05.toml",
        "realdata_final_v9_uniform10.toml",
        "realdata_final_v9_uniform20.toml",
        "realdata_final_v9_saltpepper10.toml",
        "realdata_final_v9_semantic10.toml",
    )
)
PRIMARY_CONTROLS = (
    "raw_kmeans",
    "raw_gmm",
    "vae_kmeans",
    "dec",
    "idec",
    "vae_gmm",
    "conv_vade_diag",
    "conv_ghvade_mvt",
    "conv_ghvade_mvsn",
    "conv_ghvade_mgst",
    "conv_cc",
)
TIMING_FIELDS = (
    "runtime_seconds",
    "pretrain_seconds",
    "neural_training_seconds",
    "prior_update_pipeline_seconds",
    "candidate_fit_and_score_seconds",
    "cpu_process_seconds",
    "peak_rss_mb",
    "gpu_peak_memory_mb",
)


def _float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _condition(protocol: Path) -> str:
    with protocol.open("rb") as handle:
        config = tomllib.load(handle)
    arguments = config["arguments"]
    contamination = float(arguments.get("contamination", 0.0))
    mode = str(arguments.get("contamination_mode", "uniform"))
    if contamination <= 0.0:
        return "clean"
    return f"{mode}_{int(round(100 * contamination)):02d}"


def collect_protocols(
    protocols: Iterable[Path],
    results_root: Path,
    stage: str,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    rows: list[dict[str, str]] = []
    incomplete: list[dict[str, object]] = []
    for protocol in protocols:
        condition = _condition(protocol)
        with protocol.open("rb") as handle:
            config = tomllib.load(handle)
        output_root = Path(config["experiment"]["output_dir"])
        if not output_root.is_absolute():
            output_root = results_root / output_root.name
        aggregate_dir = output_root / f"aggregate_{stage}"
        aggregate_rows = aggregate_dir / "raw_results.csv"
        aggregate_missing = aggregate_dir / "incomplete_tasks.json"
        if aggregate_rows.is_file() and aggregate_missing.is_file():
            with aggregate_rows.open(newline="", encoding="utf-8") as handle:
                protocol_rows = list(csv.DictReader(handle))
            missing = json.loads(aggregate_missing.read_text(encoding="utf-8"))
        else:
            protocol_rows, missing = _collect(load_tasks(protocol, stage), output_root)
        for row in protocol_rows:
            row["condition"] = condition
            row["protocol"] = protocol.name
        for item in missing:
            item["condition"] = condition
            item["protocol"] = protocol.name
        rows.extend(protocol_rows)
        incomplete.extend(missing)
    return rows, incomplete


def absolute_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["condition"], row["dataset"], row["method"]), []).append(row)
    output: list[dict[str, object]] = []
    for (condition, dataset, method), group in sorted(groups.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        item: dict[str, object] = {
            "condition": condition,
            "dataset": dataset,
            "method": method,
            "n_ok": len(ok),
            "n_total": len(group),
            "n_failed": len(group) - len(ok),
        }
        for field in (*METRICS, *TIMING_FIELDS):
            values = [value for row in ok if (value := _float(row, field)) is not None]
            if values:
                item[f"{field}_mean"] = mean(values)
                item[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        for field in ("n_prior_updates", "n_rejected_prior_updates", "n_failed_candidate_fits"):
            values = [value for row in ok if (value := _float(row, field)) is not None]
            if values:
                item[f"{field}_total"] = int(round(sum(values)))
                item[f"{field}_mean"] = mean(values)
        output.append(item)
    return output


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    valid = [(index, value) for index, value in enumerate(p_values) if value is not None and math.isfinite(value)]
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(sorted(valid, key=lambda item: item[1])):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def primary_comparisons(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    cells = sorted({(row["condition"], row["dataset"]) for row in rows})
    for condition, dataset in cells:
        cell_rows = [
            row for row in rows if row["condition"] == condition and row["dataset"] == dataset
        ]
        methods = {row["method"] for row in cell_rows if row.get("status") == "ok"}
        if PRIMARY not in methods:
            continue
        for control in PRIMARY_CONTROLS:
            if control not in methods:
                continue
            for metric in METRICS:
                item = _paired_delta(cell_rows, dataset, PRIMARY, control, metric)
                item["condition"] = condition
                output.append(item)
    for p_name in ("ttest_p", "wilcoxon_p"):
        adjusted = holm_adjust(
            [float(item[p_name]) if p_name in item and math.isfinite(float(item[p_name])) else None for item in output]
        )
        for item, value in zip(output, adjusted):
            item[f"holm_{p_name}"] = value
    return output


def family_frequencies(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    primary_rows = [
        row for row in rows if row.get("status") == "ok" and row.get("method") == PRIMARY
    ]
    groups: dict[tuple[str, str], Counter[str]] = {}
    for row in primary_rows:
        groups.setdefault((row["condition"], row["dataset"]), Counter())[row.get("selected_family", "unknown")] += 1
    return [
        {
            "condition": condition,
            "dataset": dataset,
            "selected_family": family,
            "count": count,
            "frequency": count / max(sum(counter.values()), 1),
        }
        for (condition, dataset), counter in sorted(groups.items())
        for family, count in sorted(counter.items())
    ]


def _latex_clean_table(summary: list[dict[str, object]], path: Path) -> None:
    clean = [item for item in summary if item["condition"] == "clean"]
    datasets = sorted({str(item["dataset"]) for item in clean})
    methods = sorted({str(item["method"]) for item in clean})
    lookup = {(str(item["dataset"]), str(item["method"])): item for item in clean}
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Method & ACC & ARI & NMI \\",
        r"\midrule",
    ]
    for dataset in datasets:
        first = True
        for method in methods:
            item = lookup.get((dataset, method))
            if item is None:
                continue
            values = []
            for metric in METRICS:
                metric_mean = item.get(f"{metric}_mean")
                metric_std = item.get(f"{metric}_std")
                values.append(
                    "--" if metric_mean is None else f"{float(metric_mean):.3f} $\\pm$ {float(metric_std):.3f}"
                )
            label = dataset.replace("_", r"\_") if first else ""
            display_method = method.replace("_", r"\_")
            lines.append(f"{label} & {display_method} & " + " & ".join(values) + r" \\")
            first = False
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the frozen v9 real-data protocols.")
    parser.add_argument("--protocols", nargs="+", type=Path, default=list(DEFAULT_PROTOCOLS))
    parser.add_argument("--stage", choices=("pilot", "final"), default="final")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_analysis",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocols = [path.resolve() for path in args.protocols]
    rows, incomplete = collect_protocols(protocols, args.results_root, args.stage)
    if incomplete and not args.allow_incomplete:
        raise SystemExit(
            f"Refusing incomplete analysis: {len(incomplete)} tasks are absent or invalid. "
            "Use --allow-incomplete only for a pilot audit."
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = absolute_summary(rows)
    comparisons = primary_comparisons(rows)
    frequencies = family_frequencies(rows)
    _write_csv(output_dir / "raw_results.csv", rows)
    _write_csv(output_dir / "absolute_results.csv", summary)
    _write_csv(output_dir / "primary_paired_comparisons.csv", comparisons)
    _write_csv(output_dir / "selected_family_frequencies.csv", frequencies)
    _latex_clean_table(summary, output_dir / "clean_absolute_results.tex")
    (output_dir / "incomplete_tasks.json").write_text(
        json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = {
        "primary_method": PRIMARY,
        "stage": args.stage,
        "n_rows": len(rows),
        "n_incomplete": len(incomplete),
        "protocols": [path.name for path in protocols],
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
