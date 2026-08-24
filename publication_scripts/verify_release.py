from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "47105f4b54b8808a5022e271b9e757d6a37775b61344a9713acf1f899b93a460"
EXPECTED_REPLICATION_CONFIG = "4f6ef3afc6e1dc3c4075d39101d15c707808584bbc5421b89c235e10889628cc"
EXPECTED_SHAPE_CONFIG = "fc8c721a1e173df58b04489c8fb7978ca13a89d6dee3d982397706415a7d4d90"
EXPECTED_DRY_BEAN_SOURCE = "fff93aed975b4ff74b1588b9250ea20e98d1317f6e370fb7334a6f2b5ec6ba69"
EXPECTED_DRY_BEAN_CONFIG = "761bba0c4c7937061f9003870417fe9fa20451a26e6f56fe90806234b02cf80a"
EXPECTED_DRY_BEAN_DATA = "d534cb4e12c1f320638877c040e6c931b3459b947ee21e0219614fdaf4663f9b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree_sha256(root: Path = ROOT) -> str:
    files = [
        *root.glob("src/**/*.py"),
        *root.glob("scripts/**/*.py"),
        root / "pyproject.toml",
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, byteorder="big"))
        digest.update(content)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    _require(_source_tree_sha256() == EXPECTED_SOURCE, "experiment source hash mismatch")
    _require(
        _sha256(ROOT / "configs/paper2_nig_orientation_replication.toml")
        == EXPECTED_REPLICATION_CONFIG,
        "replication protocol hash mismatch",
    )
    _require(
        _sha256(ROOT / "configs/simulation_protocol_v9_shape_dominant.toml")
        == EXPECTED_SHAPE_CONFIG,
        "shape-study protocol hash mismatch",
    )
    _require(
        _sha256(ROOT / "configs/paper2_dry_bean_confirmatory.toml")
        == EXPECTED_DRY_BEAN_CONFIG,
        "Dry Bean protocol hash mismatch",
    )
    dry_source = ROOT / "experiment_snapshots/dry_bean"
    _require(
        _source_tree_sha256(dry_source) == EXPECTED_DRY_BEAN_SOURCE,
        "Dry Bean experiment-source snapshot mismatch",
    )

    shape = pd.read_csv(ROOT / "results/shape_dominant/raw_results.csv")
    _require(len(shape) == 600, "shape study does not have 600 rows")
    _require(shape["status"].eq("ok").all(), "shape study contains a failed row")
    counts = shape.groupby(["design_id", "seed", "method"]).size()
    _require(len(counts) == 600 and counts.eq(1).all(), "shape-study grid is not unique")

    confirmation = pd.read_csv(
        ROOT / "results/nig_orientation_confirmation/diagnostics_all.csv"
    )
    replication = pd.read_csv(
        ROOT / "results/nig_orientation_replication/diagnostics_all.csv"
    )
    _require(len(confirmation) == 1800, "confirmation row count mismatch")
    _require(len(replication) == 4500, "replication row count mismatch")

    gate = json.loads(
        (ROOT / "results/nig_orientation_replication/gate.json").read_text(
            encoding="utf-8"
        )
    )
    _require(gate["n_diagnostic_files"] == 300, "replication task count mismatch")
    _require(gate["primary_valid_pairs"] == 98, "valid-pair count mismatch")
    _require(gate["invalid_fits"] == 46, "invalid-fit count mismatch")
    _require(not gate["passed"], "composite gate must remain failed")
    _require(not gate["checks"]["outward_ari_equivalence"], "failed equivalence was lost")

    expected = {
        "boundary_minus_outward_ari_gain": 0.028961600561695125,
        "boundary_nig_ari_gain": 0.03375217569461508,
        "boundary_nig_score_gain": 0.036613948032100614,
    }
    for key, value in expected.items():
        observed = float(gate[key]["mean"])
        _require(np.isclose(observed, value, rtol=0.0, atol=1e-12), f"{key} mismatch")

    elapsed = pd.read_csv(
        ROOT / "results/nig_orientation_replication/task_elapsed_seconds.csv"
    )
    _require(len(elapsed) == 300, "task timing count mismatch")

    dry_root = ROOT / "results/paper2_dry_bean_confirmatory/analysis"
    dry_summary = json.loads(
        (dry_root / "dry_bean_confirmatory_summary.json").read_text(encoding="utf-8")
    )
    _require(
        dry_summary["analysis_mode"] == "descriptive_fallback",
        "Dry Bean analysis must remain descriptive fallback",
    )
    _require(dry_summary["pilot_gate_passed"] is False, "failed pilot gate was lost")
    _require(dry_summary["n_rows"] == 140, "Dry Bean row count mismatch")
    _require(dry_summary["config_sha256"] == EXPECTED_DRY_BEAN_CONFIG, "Dry Bean config provenance mismatch")
    _require(dry_summary["source_tree_sha256"] == EXPECTED_DRY_BEAN_SOURCE, "Dry Bean source provenance mismatch")
    _require(
        dry_summary["dataset_identity"]["dataset_sha256"] == EXPECTED_DRY_BEAN_DATA,
        "Dry Bean dataset identity mismatch",
    )
    dry_long = pd.read_csv(dry_root / "dry_bean_results_long.csv")
    _require(len(dry_long) == 140, "Dry Bean long table does not have 140 rows")
    dry_counts = dry_long.groupby(["method", "seed"]).size()
    _require(len(dry_counts) == 140 and dry_counts.eq(1).all(), "Dry Bean grid is not unique")
    _require(dry_long["status"].eq("ok").all(), "Dry Bean contains a failed method row")
    selector = dry_summary["selector_stage_summary"]
    _require(selector["final_family_counts"] == {"gaussian": 18, "skew_normal": 2}, "Dry Bean final-family counts mismatch")
    _require(
        dry_summary["numerical_audit"][0]["failed_candidate_fits_total"] == 74,
        "Dry Bean candidate-failure count mismatch",
    )
    dry_contrasts = pd.read_csv(dry_root / "dry_bean_paired_contrasts.csv")
    t_ari = dry_contrasts[
        dry_contrasts["comparator"].eq("ghvade_mvt")
        & dry_contrasts["metric"].eq("ari")
    ].iloc[0]
    _require(
        np.isclose(float(t_ari["mean_difference"]), -0.002645368627871317, rtol=0.0, atol=1e-12),
        "Dry Bean adaptive-minus-Student-t ARI mismatch",
    )
    _require(float(t_ari["holm_p"]) == 1.0, "Dry Bean Holm correction mismatch")
    print(
        json.dumps(
            {
                "status": "ok",
                "shape_rows": len(shape),
                "confirmation_rows": len(confirmation),
                "replication_rows": len(replication),
                "replication_valid_pairs": gate["primary_valid_pairs"],
                "replication_invalid_fits": gate["invalid_fits"],
                "source_tree_sha256": EXPECTED_SOURCE,
                "dry_bean_rows": dry_summary["n_rows"],
                "dry_bean_analysis_mode": dry_summary["analysis_mode"],
                "dry_bean_source_tree_sha256": EXPECTED_DRY_BEAN_SOURCE,
                "dry_bean_dataset_sha256": EXPECTED_DRY_BEAN_DATA,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
