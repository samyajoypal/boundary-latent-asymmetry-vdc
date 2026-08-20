from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "realdata_image_pilot_v1.toml"


@dataclass(frozen=True)
class RealdataTask:
    task_id: int
    stage: str
    dataset: str
    method: str
    seed: int
    arguments: dict[str, Any]
    cell_id: str = ""

    @property
    def run_id(self) -> str:
        cell = f"__{self.cell_id}" if self.cell_id else ""
        return f"{self.stage}{cell}__{self.dataset}__{self.method}__seed{self.seed:02d}"

    def overrides(self) -> list[str]:
        values = {
            **self.arguments,
            "datasets": [self.dataset],
            "methods": [self.method],
            "seeds": [self.seed],
        }
        return [f"{key}={json.dumps(value, separators=(',', ':'))}" for key, value in sorted(values.items())]


def load_tasks(protocol_path: Path, stage: str) -> list[RealdataTask]:
    with protocol_path.open("rb") as handle:
        config = tomllib.load(handle)
    protocol = config["protocol"]
    seeds = protocol[f"{stage}_seeds"]
    default_datasets = protocol["datasets"]
    default_methods = protocol["methods"]
    cells = protocol.get("cells") or [
        {"id": "", "datasets": default_datasets, "methods": default_methods, "arguments": {}}
    ]
    tasks: list[RealdataTask] = []
    for cell in cells:
        cell_id = str(cell.get("id", ""))
        datasets = cell.get("datasets", default_datasets)
        methods = cell.get("methods", default_methods)
        cell_arguments = dict(cell.get("arguments", {}))
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    tasks.append(
                        RealdataTask(
                            task_id=len(tasks),
                            stage=stage,
                            dataset=str(dataset),
                            method=str(method),
                            seed=int(seed),
                            arguments=cell_arguments,
                            cell_id=cell_id,
                        )
                    )
    run_ids = [task.run_id for task in tasks]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Protocol generated duplicate run identifiers.")
    return tasks


def _task_payload(task: RealdataTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "run_id": task.run_id,
        "stage": task.stage,
        "dataset": task.dataset,
        "method": task.method,
        "seed": task.seed,
        "cell_id": task.cell_id,
        "overrides": task.overrides(),
    }


def completed_run_dir(protocol_path: Path, task: RealdataTask) -> Path | None:
    with protocol_path.open("rb") as handle:
        config = tomllib.load(handle)
    output_root = Path(config["experiment"]["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = output_root / task.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = run_dir / "results.csv"
    if manifest.get("status") != "completed" or not result_path.exists():
        return None
    with result_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        return None
    row = rows[0]
    if (
        row.get("status") == "ok"
        and row.get("dataset") == task.dataset
        and row.get("method") == task.method
        and int(row.get("seed", -1)) == task.seed
    ):
        return run_dir
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize or run frozen real-data protocol tasks.")
    parser.add_argument("command", choices=("count", "show", "run"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    tasks = load_tasks(protocol_path, args.stage)
    if args.command == "count":
        print(len(tasks))
        return
    if args.task_id is None or args.task_id < 0 or args.task_id >= len(tasks):
        raise SystemExit(f"--task-id must be in [0, {len(tasks) - 1}].")
    task = tasks[args.task_id]
    if args.command == "show":
        print(json.dumps(_task_payload(task), indent=2, sort_keys=True))
        return

    completed_dir = completed_run_dir(protocol_path, task)
    if completed_dir is not None:
        print(json.dumps({"status": "already_completed", "run_dir": str(completed_dir)}, sort_keys=True))
        return

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config",
        str(protocol_path),
        "--run-id",
        task.run_id,
    ]
    for override in task.overrides():
        command.extend(["--set", override])
    if os.environ.get("FMVMM_PATH"):
        command.extend(["--set", f"fmvmm_path={json.dumps(os.environ['FMVMM_PATH'])}"])
    if args.dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
