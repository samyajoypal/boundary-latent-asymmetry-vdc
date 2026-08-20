from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation_protocol import DEFAULT_PROTOCOL, ProtocolTask, load_tasks


def _read_manifest(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect(tasks: list[ProtocolTask], output_root: Path) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
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
        if row.get("method") != task.method or int(row.get("seed", -1)) != task.seed:
            incomplete.append(
                {
                    "task_id": task.task_id,
                    "run_id": task.run_id,
                    "manifest_status": "task_identity_mismatch",
                }
            )
            continue
        row["protocol_run_id"] = task.run_id
        row["protocol_tier"] = task.tier
        rows.append(row)
    return rows, incomplete


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministically aggregate frozen simulation tasks.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--tier", choices=("all", "core", "stress"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "simulation_protocol_v1")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    tasks = load_tasks(protocol_path, args.stage, args.tier)
    aggregate_dir = args.output_root / f"aggregate_{args.stage}_{args.tier}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    rows, incomplete = _collect(tasks, args.output_root)
    combined_path = aggregate_dir / "raw_results.csv"
    _write_csv(combined_path, rows)
    (aggregate_dir / "incomplete_tasks.json").write_text(
        json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
    )
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    summary = {
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "stage": args.stage,
        "tier": args.tier,
        "n_expected": len(tasks),
        "n_collected": len(rows),
        "n_incomplete": len(incomplete),
    }
    (aggregate_dir / "aggregation_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if incomplete and not args.allow_incomplete:
        raise SystemExit("Protocol is incomplete; inspect incomplete_tasks.json.")
    if rows:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_simulation_results.py"),
                str(combined_path),
                "--output-dir",
                str(aggregate_dir / "analysis"),
                "--min-runs",
                "3" if args.stage == "pilot" else "20",
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        protocol_analysis = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_protocol_results.py"),
                str(combined_path),
                "--output-dir",
                str(aggregate_dir / "protocol_analysis"),
                "--stage",
                args.stage,
            ],
            cwd=ROOT,
            check=False,
        )
        raise SystemExit(protocol_analysis.returncode)


if __name__ == "__main__":
    main()
