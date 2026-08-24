from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _normalized_scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.get("arguments", {}).pop("seeds", None)
    normalized.get("protocol", {}).pop("final_seeds", None)
    normalized.get("experiment", {}).pop("overrides", None)
    return normalized


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit(root: Path, expected_seeds: int = 20) -> dict[str, Any]:
    manifests = sorted(root.glob("final__*/manifest.json"))
    config_hashes: Counter[str | None] = Counter()
    fmvmm_commits: Counter[str | None] = Counter()
    cell_signatures: dict[tuple[str, str], set[str]] = defaultdict(set)
    cell_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    statuses: Counter[str | None] = Counter()
    missing_resolved: list[str] = []

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_hashes[manifest.get("config_sha256")] += 1
        fmvmm_commits[manifest.get("fmvmm_commit")] += 1
        statuses[manifest.get("status")] += 1
        resolved_path = manifest_path.with_name("resolved_config.json")
        if not resolved_path.is_file():
            missing_resolved.append(str(resolved_path))
            continue
        config = json.loads(resolved_path.read_text(encoding="utf-8"))
        arguments = config["arguments"]
        design = str(arguments["design_id"])
        method = str(arguments["methods"][0])
        seed = int(arguments["seeds"][0])
        key = (design, method)
        cell_signatures[key].add(_digest(_normalized_scientific_config(config)))
        cell_seeds[key].add(seed)

    cells = [
        {
            "design_id": design,
            "method": method,
            "n_scientific_signatures": len(cell_signatures[(design, method)]),
            "n_seeds": len(cell_seeds[(design, method)]),
            "seeds": sorted(cell_seeds[(design, method)]),
        }
        for design, method in sorted(cell_signatures)
    ]
    return {
        "n_manifests": len(manifests),
        "expected_cells": 30,
        "expected_seeds_per_cell": int(expected_seeds),
        "config_sha256_counts": {str(key): value for key, value in config_hashes.items()},
        "fmvmm_commit_counts": {str(key): value for key, value in fmvmm_commits.items()},
        "status_counts": {str(key): value for key, value in statuses.items()},
        "missing_resolved_configs": missing_resolved,
        "cells": cells,
        "checks": {
            "all_completed": statuses == Counter({"completed": len(manifests)}),
            "all_resolved_configs_present": not missing_resolved,
            "one_scientific_signature_per_cell": all(
                item["n_scientific_signatures"] == 1 for item in cells
            ),
            "complete_seed_coverage": len(cells) == 30
            and all(item["n_seeds"] == expected_seeds for item in cells),
            "single_fmvmm_revision": len(fmvmm_commits) == 1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen v9 simulation provenance.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results" / "simulation_protocol_v9_shape_dominant",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results"
        / "simulation_protocol_v9_shape_dominant"
        / "aggregate_final_core"
        / "provenance_audit.json",
    )
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["checks"], sort_keys=True))
    if not all(report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
