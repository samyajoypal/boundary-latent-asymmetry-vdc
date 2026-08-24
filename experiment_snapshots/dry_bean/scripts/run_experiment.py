from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {
    "simulation": ROOT / "scripts" / "run_simulation_benchmark.py",
    "realdata": ROOT / "scripts" / "run_realdata_benchmark.py",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    experiment = config.get("experiment")
    arguments = config.get("arguments")
    if not isinstance(experiment, dict) or not isinstance(arguments, dict):
        raise ValueError("Config requires [experiment] and [arguments] tables.")
    kind = experiment.get("kind")
    if kind not in RUNNERS:
        raise ValueError(f"experiment.kind must be one of {sorted(RUNNERS)}.")
    if not experiment.get("name"):
        raise ValueError("experiment.name is required.")
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    arguments = resolved["arguments"]
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in arguments:
            raise KeyError(f"Unknown argument override {key!r}.")
        current = arguments[key]
        if isinstance(current, list):
            value = json.loads(raw_value)
            if not isinstance(value, list):
                raise ValueError(f"List override {key!r} requires a JSON array.")
        elif isinstance(current, bool):
            normalized = raw_value.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Boolean override {key!r} requires true or false.")
            value = normalized == "true"
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw_value)
        elif isinstance(current, float):
            value = float(raw_value)
        elif current is None:
            value = json.loads(raw_value)
        else:
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError:
                decoded = raw_value
            value = decoded if isinstance(decoded, str) else raw_value
        arguments[key] = value
    resolved["experiment"]["overrides"] = list(overrides)
    return resolved


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_command(config: dict[str, Any], run_id: str) -> tuple[list[str], Path]:
    experiment = config["experiment"]
    arguments = dict(config["arguments"])
    output_base = Path(experiment.get("output_dir", f"results/{experiment['name']}"))
    if not output_base.is_absolute():
        output_base = ROOT / output_base
    command = [
        sys.executable,
        str(RUNNERS[str(experiment["kind"])]),
        "--output-dir",
        str(output_base),
        "--run-id",
        run_id,
    ]
    for key in sorted(arguments):
        value = arguments[key]
        if value is None:
            continue
        if isinstance(value, bool):
            command.append(_flag(key) if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            command.append(_flag(key))
            command.extend(str(item) for item in value)
        else:
            command.extend([_flag(key), str(value)])
    return command, output_base / run_id


def prepare_run_dir(run_dir: Path) -> dict[str, Any] | None:
    """Create or reuse a run directory after an interrupted scheduler attempt."""

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return None
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {"previous_status": "missing_manifest"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"previous_status": "invalid_manifest"}
    if manifest.get("status") == "completed":
        result_path = run_dir / "results.csv"
        if result_path.exists():
            with result_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows and all(row.get("status") == "ok" for row in rows):
                raise FileExistsError(f"Completed run directory already exists: {run_dir}")
            return {
                "previous_status": "completed_with_failed_results",
                "previous_created_at_utc": manifest.get("created_at_utc"),
            }
        return {
            "previous_status": "completed_without_results",
            "previous_created_at_utc": manifest.get("created_at_utc"),
        }
    return {
        "previous_status": manifest.get("status", "unknown"),
        "previous_created_at_utc": manifest.get("created_at_utc"),
    }


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        revision_file = path / "REVISION"
        if revision_file.exists():
            return revision_file.read_text(encoding="utf-8").strip() or None
        return None
    return result.stdout.strip() or None


def _source_tree_sha256(root: Path) -> str:
    files = [
        *root.glob("src/**/*.py"),
        *root.glob("scripts/**/*.py"),
        *(path for path in (root / "pyproject.toml",) if path.is_file()),
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def environment_manifest(config: dict[str, Any], config_path: Path, command: list[str]) -> dict[str, Any]:
    arguments = config["arguments"]
    fmvmm_path = arguments.get("fmvmm_path")
    distributions = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        torch_info = {"available": False}
    config_bytes = config_path.read_bytes()
    resolved_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
        "command": command,
        "command_shell": shlex.join(command),
        "repository_commit": _git_revision(ROOT),
        "source_tree_sha256": _source_tree_sha256(ROOT),
        "fmvmm_path": str(Path(fmvmm_path).expanduser()) if fmvmm_path else None,
        "fmvmm_commit": _git_revision(Path(fmvmm_path).expanduser()) if fmvmm_path else None,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": dict(sorted(distributions.items(), key=lambda item: item[0].lower())),
        "torch": torch_info,
        "slurm": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a versioned GH-VaDE experiment config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one [arguments] value; list values use JSON arrays.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = apply_overrides(load_config(config_path), args.overrides)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    command, run_dir = build_command(config, run_id)
    previous_attempt = prepare_run_dir(run_dir)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = environment_manifest(config, config_path, command)
    if previous_attempt is not None:
        manifest["reused_incomplete_run_dir"] = True
        manifest["previous_attempt"] = previous_attempt
    manifest["status"] = "dry_run" if args.dry_run else "running"
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "command": command}, indent=2))
    if args.dry_run:
        return

    started = time.monotonic()
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest["exit_code"] = int(completed.returncode)
    manifest["status"] = "completed" if completed.returncode == 0 else "failed"
    manifest["stdout_log"] = str(stdout_path)
    manifest["stderr_log"] = str(stderr_path)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
