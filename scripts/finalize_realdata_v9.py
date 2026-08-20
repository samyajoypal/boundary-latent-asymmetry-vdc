from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = (
    ROOT / "configs" / "realdata_final_v9_clean.toml",
    ROOT / "configs" / "realdata_final_v9_clean_auxiliary.toml",
    ROOT / "configs" / "realdata_final_v9_uniform05.toml",
    ROOT / "configs" / "realdata_final_v9_uniform10.toml",
    ROOT / "configs" / "realdata_final_v9_uniform20.toml",
    ROOT / "configs" / "realdata_final_v9_saltpepper10.toml",
    ROOT / "configs" / "realdata_final_v9_semantic10.toml",
    ROOT / "configs" / "realdata_final_v9_ablations.toml",
)
MAIN_PROTOCOLS = PROTOCOLS[:-1]
FMVMM_COMMIT = "90aa86dfebc9a86f352f5fe1de96a38bbba2aa35"


def _output_root(protocol: Path) -> Path:
    with protocol.open("rb") as handle:
        config = tomllib.load(handle)
    return ROOT / config["experiment"]["output_dir"]


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def audit_provenance(
    protocols: tuple[Path, ...],
    stage: str,
    output_path: Path,
) -> dict[str, object]:
    from scripts.realdata_protocol import load_tasks

    errors: list[dict[str, object]] = []
    source_hashes: set[str] = set()
    config_hashes: dict[str, str] = {}
    n_manifests = 0
    for protocol in protocols:
        expected_config_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
        config_hashes[protocol.name] = expected_config_hash
        output_root = _output_root(protocol)
        for task in load_tasks(protocol, stage):
            manifest_path = output_root / task.run_id / "manifest.json"
            if not manifest_path.is_file():
                errors.append({"run_id": task.run_id, "error": "missing_manifest"})
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    {"run_id": task.run_id, "error": "invalid_manifest", "detail": str(exc)}
                )
                continue
            n_manifests += 1
            if manifest.get("status") != "completed":
                errors.append(
                    {
                        "run_id": task.run_id,
                        "error": "manifest_not_completed",
                        "value": manifest.get("status"),
                    }
                )
            if manifest.get("config_sha256") != expected_config_hash:
                errors.append(
                    {
                        "run_id": task.run_id,
                        "error": "config_hash_mismatch",
                        "value": manifest.get("config_sha256"),
                    }
                )
            source_hash = manifest.get("source_tree_sha256")
            if not isinstance(source_hash, str) or not source_hash:
                errors.append({"run_id": task.run_id, "error": "missing_source_tree_hash"})
            else:
                source_hashes.add(source_hash)
            if manifest.get("fmvmm_commit") != FMVMM_COMMIT:
                errors.append(
                    {
                        "run_id": task.run_id,
                        "error": "fmvmm_commit_mismatch",
                        "value": manifest.get("fmvmm_commit"),
                    }
                )
    if len(source_hashes) != 1:
        errors.append(
            {
                "error": "mixed_source_tree_hashes",
                "values": sorted(source_hashes),
            }
        )
    report: dict[str, object] = {
        "passed": not errors,
        "stage": stage,
        "n_expected": sum(len(load_tasks(protocol, stage)) for protocol in protocols),
        "n_manifests": n_manifests,
        "source_tree_sha256": next(iter(source_hashes)) if len(source_hashes) == 1 else None,
        "fmvmm_commit": FMVMM_COMMIT,
        "config_sha256": config_hashes,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise RuntimeError(
            f"Confirmatory provenance audit failed with {len(errors)} error(s); "
            f"see {output_path}."
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and generate all frozen v9 real-data artifacts."
    )
    parser.add_argument("--stage", choices=("pilot", "final"), default="final")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit missing tasks for diagnostic pilot analyses only.",
    )
    parser.add_argument("--skip-interpretability", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "final" and args.allow_incomplete:
        raise SystemExit("Confirmatory finalization cannot use --allow-incomplete.")

    for protocol in PROTOCOLS:
        command = [
            sys.executable,
            "scripts/aggregate_realdata_protocol.py",
            "--protocol",
            str(protocol),
            "--stage",
            args.stage,
            "--output-root",
            str(_output_root(protocol)),
        ]
        if args.allow_incomplete:
            command.append("--allow-incomplete")
        _run(command)

    analysis_dir = ROOT / "results" / "realdata_final_v9_analysis"
    if args.stage == "final":
        audit_provenance(PROTOCOLS, args.stage, analysis_dir / "provenance_audit.json")
    analysis_command = [
        sys.executable,
        "scripts/analyze_final_realdata.py",
        "--stage",
        args.stage,
        "--output-dir",
        str(analysis_dir),
        "--protocols",
        *(str(path) for path in MAIN_PROTOCOLS),
    ]
    if args.allow_incomplete:
        analysis_command.append("--allow-incomplete")
    _run(analysis_command)

    if args.stage == "final":
        _run([sys.executable, "scripts/analyze_realdata_ablations.py"])
        _run([sys.executable, "scripts/make_final_realdata_artifacts.py"])
        if not args.skip_interpretability:
            _run([sys.executable, "scripts/make_interpretability_figures.py"])


if __name__ == "__main__":
    main()
