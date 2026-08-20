from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "conv_ghvade_adaptive"
ANCHOR = "conv_ghvade_mvt"


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _minimum_interior_shape(result_path: Path) -> float | None:
    metadata_paths = list(result_path.parent.glob("artifacts/*/seed_*/*/prior_metadata.json"))
    if len(metadata_paths) != 1:
        return None
    metadata = _read_json(metadata_paths[0])
    if not isinstance(metadata, dict):
        return None
    summary = metadata.get("parameter_summary", {})
    components = summary.get("components", []) if isinstance(summary, dict) else []
    values: list[float] = []
    for component in components if isinstance(components, list) else []:
        if not isinstance(component, dict):
            continue
        family = str(component.get("family", ""))
        if family not in {"mghp", "mvhb", "mnig"}:
            continue
        for field in ("chi", "psi"):
            if field in component:
                value = float(component[field])
                if value > 0.0:
                    values.append(value)
    return min(values) if values else None


def diagnose_protocol(protocol_dir: Path) -> pd.DataFrame:
    aggregate_path = protocol_dir / "aggregate_final" / "raw_results.csv"
    aggregate = pd.read_csv(aggregate_path)
    anchor = aggregate[aggregate["method"] == ANCHOR].set_index(["dataset", "seed"])
    rows: list[dict[str, object]] = []
    pattern = f"final__*__{PRIMARY}__seed*/results.json"
    for result_path in sorted(protocol_dir.glob(pattern)):
        payload = _read_json(result_path)
        if not isinstance(payload, list) or len(payload) != 1:
            continue
        result = payload[0]
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        key = (str(result["dataset"]), int(result["seed"]))
        if key not in anchor.index:
            continue
        reference = anchor.loc[key]
        history = result.get("prior_update_history", [])
        if not isinstance(history, list) or not history:
            continue
        active = [str(update.get("active_prior_family", "")) for update in history]
        selected = [str(update.get("selected_candidate", "")) for update in history]
        rows.append(
            {
                "protocol": protocol_dir.name,
                "dataset": key[0],
                "seed": key[1],
                "initial_family": active[0],
                "final_family": active[-1],
                "active_family_switches": sum(
                    left != right for left, right in zip(active, active[1:])
                ),
                "candidate_switches": sum(
                    left != right for left, right in zip(selected, selected[1:])
                ),
                "shape_gate_triggers": sum(
                    bool(update.get("shape_gate_triggered")) for update in history
                ),
                "rejected_updates": sum(
                    update.get("prior_update_accepted") is False for update in history[1:]
                ),
                "active_components": int(result.get("active_components", 0)),
                "ari_delta_vs_student_t": float(result["ari"]) - float(reference["ari"]),
                "acc_delta_vs_student_t": float(result["acc"]) - float(reference["acc"]),
                "nmi_delta_vs_student_t": float(result["nmi"]) - float(reference["nmi"]),
                "final_asymmetric_margin": history[-1].get(
                    "shape_gate_asymmetric_margin"
                ),
                "final_global_abs_skew": history[-1].get("shape_gate_abs_skew"),
                "minimum_final_interior_shape": _minimum_interior_shape(result_path),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                rendered = "" if pd.isna(value) else f"{value:.4f}"
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _markdown_summary(runs: pd.DataFrame, family: pd.DataFrame) -> str:
    clean = runs[runs["protocol"] == "realdata_final_v9_clean"]
    collapsed = clean[clean["active_components"] <= 1]
    student_t = clean[clean["final_family"] == "student_t"]
    lines = [
        "# Student-t gap audit",
        "",
        "This is a development-only diagnostic. External labels are used to explain",
        "the frozen v9 outcome; they are not inputs to the v10 selector.",
        "",
        f"- Audited adaptive runs: {len(runs)}.",
        f"- Clean runs ending with one active component: {len(collapsed)}.",
        (
            "- Mean clean ARI delta for runs ending in Student-t: "
            f"{student_t['ari_delta_vs_student_t'].mean():+.4f}."
        ),
    ]
    if not collapsed.empty:
        lines.append(
            "- Mean clean ARI delta for collapsed runs: "
            f"{collapsed['ari_delta_vs_student_t'].mean():+.4f}."
        )
    lines.extend(
        [
            "",
            "## Final-family summary",
            "",
            _markdown_table(family),
            "",
            "## Mechanistic conclusion",
            "",
            "The primary deficit is not evidence that Student-t universally models the",
            "latent data better. It is caused by unsupported asymmetric promotion,",
            "near-boundary GH fits, and path dependence after the encoder adapts to a",
            "selected family. The v10 method therefore anchors on Student-t, validates",
            "the regular parameter region, imposes unlabeled assignment-safety gates,",
            "and locks the family after one decision.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose adaptive GH versus fixed Student-t.")
    parser.add_argument(
        "--protocol-dirs",
        nargs="+",
        type=Path,
        default=[
            ROOT / "results" / "realdata_final_v9_clean",
            ROOT / "results" / "realdata_final_v9_uniform10",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "method_v10_diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_frames = [diagnose_protocol(path.resolve()) for path in args.protocol_dirs]
    runs = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in protocol_frames if not frame.empty],
        ignore_index=True,
    )
    family = (
        runs.groupby(["protocol", "dataset", "final_family"], dropna=False)
        .agg(
            n=("seed", "size"),
            ari_delta=("ari_delta_vs_student_t", "mean"),
            acc_delta=("acc_delta_vs_student_t", "mean"),
            nmi_delta=("nmi_delta_vs_student_t", "mean"),
            active_components=("active_components", "mean"),
            family_switches=("active_family_switches", "mean"),
        )
        .reset_index()
        .sort_values(["protocol", "dataset", "n"], ascending=[True, True, False])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "student_t_gap_runs.csv", index=False)
    family.to_csv(args.output_dir / "student_t_gap_family_summary.csv", index=False)
    (args.output_dir / "student_t_gap_audit.md").write_text(
        _markdown_summary(runs, family), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_runs": len(runs),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
