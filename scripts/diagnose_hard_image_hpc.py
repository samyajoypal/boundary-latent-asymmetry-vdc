from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.realdata_protocol import load_tasks


@dataclass(frozen=True)
class JobSpec:
    label: str
    job_id: str
    protocol: Path
    aggregate_dir: Path


def _parse_spec(raw: str) -> JobSpec:
    parts = raw.split(",", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "spec must be label,job_id,protocol_path,aggregate_dir"
        )
    label, job_id, protocol, aggregate_dir = parts
    return JobSpec(label, job_id, Path(protocol), Path(aggregate_dir))


def _sacct(job_id: str) -> dict[int, dict[str, str]]:
    command = [
        "sacct",
        "-X",
        "-j",
        job_id,
        "--format=JobID,State,ExitCode,Elapsed,NodeList%80",
        "-P",
        "--noheader",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    rows: dict[int, dict[str, str]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 5 or "_" not in fields[0] or "." in fields[0]:
            continue
        _, task_text = fields[0].split("_", 1)
        if not task_text.isdigit():
            continue
        rows[int(task_text)] = {
            "state": fields[1],
            "exit_code": fields[2],
            "elapsed": fields[3],
            "node": fields[4],
        }
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _error_kind(row: dict[str, str]) -> str:
    message = " ".join(
        str(row.get(key, ""))
        for key in ("error_message", "error", "exception", "status_message")
    )
    lowered = message.lower()
    if "no kernel image" in lowered:
        return "cuda_no_kernel_image"
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "cuda_oom"
    if row.get("status") and row["status"] != "ok":
        return row["status"]
    return "unknown_failure"


def _manifest_status(run_dir: Path) -> str:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return "missing_manifest"
    try:
        return str(json.loads(manifest.read_text(encoding="utf-8")).get("status"))
    except json.JSONDecodeError:
        return "bad_manifest"


def diagnose(spec: JobSpec) -> None:
    tasks = load_tasks(spec.protocol, "pilot")
    by_run = {task.run_id: task for task in tasks}
    by_task = {task.task_id: task for task in tasks}
    slurm = _sacct(spec.job_id)
    aggregate = spec.aggregate_dir
    raw_rows = _read_csv(aggregate / "raw_results.csv")
    failures = [row for row in raw_rows if row.get("status") != "ok"]
    incomplete = json.loads((aggregate / "incomplete_tasks.json").read_text(encoding="utf-8"))

    print(f"\n## {spec.label} ({spec.job_id})")
    manifest = json.loads((aggregate / "aggregation_manifest.json").read_text(encoding="utf-8"))
    print(
        "aggregate:",
        f"{manifest.get('n_collected')}/{manifest.get('n_expected')} collected,",
        f"{manifest.get('n_incomplete')} incomplete",
    )

    node_error_counts: Counter[tuple[str, str]] = Counter()
    method_error_counts: Counter[tuple[str, str]] = Counter()
    for row in failures:
        task = by_run.get(row.get("protocol_run_id", ""))
        node = slurm.get(task.task_id, {}).get("node", "unknown") if task else "unknown"
        kind = _error_kind(row)
        node_error_counts[(node, kind)] += 1
        method_error_counts[(row.get("method", "unknown"), kind)] += 1

    if failures:
        print("failures by node/error:")
        for (node, kind), count in sorted(node_error_counts.items()):
            print(f"  {node}: {kind} x{count}")
        print("failures by method/error:")
        for (method, kind), count in sorted(method_error_counts.items()):
            print(f"  {method}: {kind} x{count}")
    else:
        print("failures by node/error: none")

    if incomplete:
        print("incomplete tasks:")
        for item in incomplete:
            task_id = int(item["task_id"])
            task = by_task[task_id]
            state = slurm.get(task_id, {}).get("state", "unknown")
            elapsed = slurm.get(task_id, {}).get("elapsed", "unknown")
            node = slurm.get(task_id, {}).get("node", "unknown")
            run_dir = spec.aggregate_dir.parent / task.run_id
            print(
                f"  task {task_id:02d} {task.dataset}/{task.method}/seed{task.seed}: "
                f"{state} on {node} elapsed={elapsed} manifest={_manifest_status(run_dir)}"
            )
    else:
        print("incomplete tasks: none")

    no_kernel_nodes = sorted(
        node for (node, kind), _ in node_error_counts.items() if kind == "cuda_no_kernel_image"
    )
    if no_kernel_nodes:
        print("nodes with CUDA architecture failures:", ", ".join(no_kernel_nodes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose hard-image Slurm pilot failures.")
    parser.add_argument("--spec", action="append", type=_parse_spec, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for spec in args.spec:
        diagnose(spec)


if __name__ == "__main__":
    main()
