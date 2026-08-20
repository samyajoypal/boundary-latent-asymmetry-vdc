from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional reporting path
    stats = None


BASELINE_METHODS = {
    "raw_kmeans",
    "raw_gmm",
    "ae_kmeans",
    "dec",
    "idec",
    "vae_kmeans",
    "vae_gmm",
    "vae_diag_gmm",
    "diag_gmm_prior",
}


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _method_group(method: str) -> str:
    if method in {"latent_kmeans", "latent_gmm", "latent_diag_gmm"}:
        return "oracle_control"
    if method.startswith("latent_"):
        return "oracle_flexible"
    if method.startswith("ghvade_") or method.startswith("conv_ghvade_"):
        return "mixture_aware"
    if method.startswith("vade_diag") or method.startswith("conv_vade_diag"):
        return "baseline"
    if method.startswith("vae_fmvmm_"):
        return "vae_fmvmm"
    if method.startswith("fmvmm_"):
        return "prior_fmvmm"
    if method in BASELINE_METHODS:
        return "baseline"
    return "other"


def _summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(row["dataset"], row["method"])].append(row)

    out: list[dict[str, object]] = []
    for (dataset, method), group in sorted(grouped.items()):
        item: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "group": _method_group(method),
            "n_runs": len(group),
        }
        for metric in ("acc", "nmi", "ari", "purity"):
            values = [v for row in group if (v := _float(row, metric)) is not None]
            if values:
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(item)
    return out


def _best_by_group(summary: list[dict[str, object]], min_runs: int) -> dict[str, dict[str, dict[str, object]]]:
    best: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for item in summary:
        if int(item.get("n_runs", 0)) < int(min_runs):
            continue
        dataset = str(item["dataset"])
        group = str(item["group"])
        if "acc_mean" not in item:
            continue
        current = best[dataset].get(group)
        if current is None or float(item["acc_mean"]) > float(current["acc_mean"]):
            best[dataset][group] = item
    return best


def _paired_delta(
    ok_rows: list[dict[str, str]],
    dataset: str,
    treatment: str,
    control: str,
) -> dict[str, object]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in ok_rows:
        if row.get("dataset") == dataset and row.get("method") in {treatment, control}:
            by_key[(row["method"], row["seed"])] = row

    deltas: list[float] = []
    for seed in sorted({seed for method, seed in by_key if method in {treatment, control}}, key=int):
        t = by_key.get((treatment, seed))
        c = by_key.get((control, seed))
        if t is None or c is None:
            continue
        t_acc = _float(t, "acc")
        c_acc = _float(c, "acc")
        if t_acc is not None and c_acc is not None:
            deltas.append(t_acc - c_acc)

    out: dict[str, object] = {
        "dataset": dataset,
        "treatment": treatment,
        "control": control,
        "n_pairs": len(deltas),
    }
    if deltas:
        out["delta_acc_mean"] = mean(deltas)
        out["delta_acc_std"] = pstdev(deltas) if len(deltas) > 1 else 0.0
        out["wins"] = sum(1 for value in deltas if value > 0)
        out["ties"] = sum(1 for value in deltas if value == 0)
        out["losses"] = sum(1 for value in deltas if value < 0)
        if len(deltas) > 1 and stats is not None:
            sem = stats.sem(deltas)
            ci = stats.t.interval(0.95, len(deltas) - 1, loc=mean(deltas), scale=sem)
            out["delta_acc_ci95_low"] = float(ci[0])
            out["delta_acc_ci95_high"] = float(ci[1])
            out["ttest_p"] = float(stats.ttest_rel(deltas, [0.0] * len(deltas)).pvalue)
            try:
                out["wilcoxon_p"] = float(stats.wilcoxon(deltas).pvalue)
            except ValueError:
                out["wilcoxon_p"] = math.nan
    return out


def _failure_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    total = Counter((row.get("dataset", ""), row.get("method", "")) for row in rows)
    failed = Counter(
        (row.get("dataset", ""), row.get("method", ""))
        for row in rows
        if row.get("status") != "ok"
    )
    out = []
    for key, n_failed in sorted(failed.items()):
        dataset, method = key
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "n_failed": n_failed,
                "n_total": total[key],
                "failure_rate": n_failed / max(total[key], 1),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_mean_std(item: dict[str, object], metric: str = "acc") -> str:
    mean_value = float(item.get(f"{metric}_mean", math.nan))
    std_value = float(item.get(f"{metric}_std", math.nan))
    return f"{mean_value:.3f} +/- {std_value:.3f}"


def _fmt_mean_std_tex(item: dict[str, object], metric: str = "acc") -> str:
    mean_value = float(item.get(f"{metric}_mean", math.nan))
    std_value = float(item.get(f"{metric}_std", math.nan))
    return f"${mean_value:.3f}\\pm {std_value:.3f}$"


def _tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


METHOD_LABELS = {
    "raw_kmeans": "raw k-means",
    "raw_gmm": "raw GMM",
    "ae_kmeans": "AE+k-means",
    "dec": "DEC",
    "idec": "IDEC",
    "vae_kmeans": "VAE+k-means",
    "vae_gmm": "VAE+GMM",
    "vae_diag_gmm": "VAE+diag-GMM",
    "diag_gmm_prior": "diag-GMM prior VAE",
    "vae_fmvmm_mvsn": "VAE+skew-normal",
    "vae_fmvmm_mvt": "VAE+Student-t",
    "vae_fmvmm_mgst": "VAE+skew-t",
    "vae_fmvmm_gh_family": "VAE+GH catalogue",
    "fmvmm_mvsn": "prior skew-normal",
    "fmvmm_mvt": "prior Student-t",
    "fmvmm_mgst": "prior skew-t",
    "fmvmm_gh_family": "prior GH catalogue",
    "latent_kmeans": "oracle k-means",
    "latent_gmm": "oracle GMM",
    "latent_diag_gmm": "oracle diag-GMM",
    "latent_fmvmm_mvsn": "oracle skew-normal",
    "latent_fmvmm_mvt": "oracle Student-t",
    "latent_fmvmm_mgst": "oracle skew-t",
    "latent_fmvmm_gh_family": "oracle GH catalogue",
    "vade_diag": "VaDE",
    "vade_diag_mi": "VaDE+MI",
    "ghvade_mvsn": "GH-VaDE skew-normal",
    "ghvade_mvsn_mi": "GH-VaDE skew-normal+MI",
    "ghvade_mvt": "GH-VaDE Student-t",
    "ghvade_mgst": "GH-VaDE skew-t",
    "ghvade_mvvg": "GH-VaDE variance-gamma",
    "ghvade_mvhb": "GH-VaDE hyperbolic",
    "ghvade_mnig": "GH-VaDE NIG",
    "ghvade_mghp": "GH-VaDE full GH",
    "ghvade_adaptive": "Adaptive GH-VaDE",
    "ghvade_mvt_mi": "GH-VaDE Student-t+MI",
    "ghvade_gh_family": "GH-VaDE catalogue",
    "ghvade_gh_family_mi": "GH-VaDE catalogue+MI",
    "ghvade_tail_select": "Tail-selected GH-VaDE",
    "ghvade_tail_select_mi": "Tail-selected GH-VaDE+MI",
    "ghvade_conservative": "Conservative GH-VaDE",
    "ghvade_conservative_mi": "Conservative GH-VaDE+MI",
    "conv_vade_diag": "ConvVaDE",
    "conv_vade_diag_mi": "ConvVaDE+MI",
    "conv_ghvade_mvt": "Conv GH-VaDE Student-t",
    "conv_ghvade_mvt_mi": "Conv GH-VaDE Student-t+MI",
    "conv_ghvade_gh_family": "Conv GH-VaDE catalogue",
    "conv_ghvade_tail_select": "Conv tail-selected GH-VaDE",
    "conv_ghvade_tail_select_mi": "Conv tail-selected GH-VaDE+MI",
    "conv_ghvade_conservative": "Conv conservative GH-VaDE",
    "conv_ghvade_conservative_mi": "Conv conservative GH-VaDE+MI",
}


def _display_method(method: object) -> str:
    return METHOD_LABELS.get(str(method), str(method))


def _best_oracle(groups: dict[str, dict[str, object]]) -> dict[str, object] | None:
    candidates = [
        item
        for item in (groups.get("oracle_control"), groups.get("oracle_flexible"))
        if item is not None
    ]
    return max(candidates, key=lambda item: float(item["acc_mean"])) if candidates else None


def _main_table(
    best: dict[str, dict[str, dict[str, object]]],
    paired: list[dict[str, object]],
) -> str:
    paired_by_dataset = {str(row["dataset"]): row for row in paired if row.get("treatment") != ""}
    lines = [
        "# Simulation Screen Summary",
        "",
        "| Scenario | Best baseline | Best proposed VAE/prior | Oracle latent ceiling | Proposed - baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in sorted(best):
        groups = best[dataset]
        baseline = groups.get("baseline")
        vae_fmvmm = groups.get("vae_fmvmm")
        prior_fmvmm = groups.get("prior_fmvmm")
        mixture_aware = groups.get("mixture_aware")
        oracle = _best_oracle(groups)
        proposed_candidates = [item for item in (vae_fmvmm, prior_fmvmm, mixture_aware) if item is not None]
        proposed = max(proposed_candidates, key=lambda item: float(item["acc_mean"])) if proposed_candidates else None
        if baseline is None or proposed is None:
            continue
        paired_row = paired_by_dataset.get(dataset, {})
        delta = paired_row.get("delta_acc_mean")
        delta_text = "" if delta is None else f"{float(delta):+.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset.replace("simulation_", ""),
                    f"{_display_method(baseline['method'])} ({_fmt_mean_std(baseline)})",
                    f"{_display_method(proposed['method'])} ({_fmt_mean_std(proposed)})",
                    "" if oracle is None else f"{_display_method(oracle['method'])} ({_fmt_mean_std(oracle)})",
                    delta_text,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _latex_table(best: dict[str, dict[str, dict[str, object]]], n_seeds: int) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{Hard synthetic-screen results. Entries are mean $\\pm$ standard deviation over {n_seeds} seeds.}}",
        "\\label{tab:hard-synthetic-screen}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Scenario & Best baseline & Best proposed & Oracle latent ceiling\\\\",
        "\\midrule",
    ]
    for dataset in sorted(best):
        groups = best[dataset]
        baseline = groups.get("baseline")
        vae_fmvmm = groups.get("vae_fmvmm")
        prior_fmvmm = groups.get("prior_fmvmm")
        mixture_aware = groups.get("mixture_aware")
        oracle = _best_oracle(groups)
        proposed_candidates = [item for item in (vae_fmvmm, prior_fmvmm, mixture_aware) if item is not None]
        if baseline is None or not proposed_candidates:
            continue
        proposed = max(proposed_candidates, key=lambda item: float(item["acc_mean"]))
        oracle_text = "--" if oracle is None else f"{_display_method(oracle['method'])} ({_fmt_mean_std_tex(oracle)})"
        scenario = _tex_escape(dataset.replace("simulation_", ""))
        baseline_text = _tex_escape(f"{_display_method(baseline['method'])} ({_fmt_mean_std_tex(baseline)})")
        proposed_text = _tex_escape(f"{_display_method(proposed['method'])} ({_fmt_mean_std_tex(proposed)})")
        oracle_text = _tex_escape(oracle_text)
        lines.append(
            f"{scenario} & "
            f"{baseline_text} & "
            f"{proposed_text} & "
            f"{oracle_text}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-facing summaries for hard simulation benchmarks.")
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("tables"))
    parser.add_argument("--min-runs", type=int, default=4)
    args = parser.parse_args()

    rows = _read_rows(args.csv_files)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = _summary(rows)
    best = _best_by_group(summary, args.min_runs)

    paired: list[dict[str, object]] = []
    oracle_paired: list[dict[str, object]] = []
    vade_paired: list[dict[str, object]] = []
    for dataset, groups in sorted(best.items()):
        baseline = groups.get("baseline")
        vae_fmvmm = groups.get("vae_fmvmm")
        prior_fmvmm = groups.get("prior_fmvmm")
        mixture_aware = groups.get("mixture_aware")
        proposed_candidates = [item for item in (vae_fmvmm, prior_fmvmm, mixture_aware) if item is not None]
        if baseline is not None and proposed_candidates:
            proposed = max(proposed_candidates, key=lambda item: float(item["acc_mean"]))
            paired.append(_paired_delta(ok_rows, dataset, str(proposed["method"]), str(baseline["method"])))
        oracle = groups.get("oracle_flexible")
        latent_controls = [
            item
            for item in summary
            if item["dataset"] == dataset
            and item["method"] in {"latent_gmm", "latent_kmeans"}
            and int(item.get("n_runs", 0)) >= int(args.min_runs)
        ]
        if oracle is not None and latent_controls:
            latent_control = max(latent_controls, key=lambda item: float(item.get("acc_mean", -math.inf)))
            oracle_paired.append(_paired_delta(ok_rows, dataset, str(oracle["method"]), str(latent_control["method"])))

        mixture_methods = [
            item
            for item in summary
            if item["dataset"] == dataset
            and item["group"] == "mixture_aware"
            and int(item.get("n_runs", 0)) >= int(args.min_runs)
        ]
        available_methods = {
            str(item["method"])
            for item in summary
            if item["dataset"] == dataset and int(item.get("n_runs", 0)) >= int(args.min_runs)
        }
        for treatment in mixture_methods:
            treatment_name = str(treatment["method"])
            control_name = "vade_diag_mi" if treatment_name.endswith("_mi") else "vade_diag"
            if control_name in available_methods:
                vade_paired.append(_paired_delta(ok_rows, dataset, treatment_name, control_name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "simulation_screen_method_summary.csv", summary)
    _write_csv(args.output_dir / "simulation_screen_pairwise.csv", paired)
    _write_csv(args.output_dir / "simulation_screen_oracle_pairwise.csv", oracle_paired)
    _write_csv(args.output_dir / "simulation_screen_vs_vade.csv", vade_paired)
    _write_csv(args.output_dir / "simulation_screen_failures.csv", _failure_summary(rows))
    (args.output_dir / "simulation_screen_summary.md").write_text(_main_table(best, paired), encoding="utf-8")
    n_seeds = len({int(row["seed"]) for row in ok_rows if row.get("seed") not in (None, "")})
    (args.output_dir / "simulation_screen_table.tex").write_text(
        _latex_table(best, n_seeds=n_seeds), encoding="utf-8"
    )
    (args.output_dir / "simulation_screen_analysis.json").write_text(
        json.dumps(
            {
                "n_rows": len(rows),
                "n_ok": len(ok_rows),
                "n_failed": len(rows) - len(ok_rows),
                "paired": paired,
                "oracle_paired": oracle_paired,
                "vade_paired": vade_paired,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print((args.output_dir / "simulation_screen_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
