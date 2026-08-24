from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "pattern_recognition_submission"
FIGURES = SUBMISSION / "figures"
PRIMARY = "conv_ghvade_adaptive"
DATASETS = ("mnist", "fashion_mnist", "usps")
CONDITIONS = ("clean", "uniform_05", "uniform_10", "uniform_20", "salt_pepper_10", "semantic_10")
MAIN_METHODS = (
    "dec",
    "idec",
    "vae_gmm",
    "conv_vade_diag",
    "conv_ghvade_mvt",
    PRIMARY,
    "conv_cc",
)
ROBUST_METHODS = ("conv_vade_diag", "conv_ghvade_mvt", PRIMARY, "conv_cc")
METHOD_LABELS = {
    "raw_kmeans": "$k$-means",
    "raw_gmm": "Raw GMM",
    "vae_kmeans": "VAE+$k$-means",
    "dec": "DEC",
    "idec": "IDEC",
    "vae_gmm": "VAE+GMM",
    "conv_vade_diag": "Gaussian VaDE",
    "conv_ghvade_mvt": "Student-$t$ VaDE",
    "conv_ghvade_mvsn": "Skew-normal VaDE",
    "conv_ghvade_mgst": "Skew-$t$ VaDE",
    PRIMARY: "Adaptive GH-VaDE",
    "conv_ghvade_adaptive_mi": "Adaptive GH-VaDE+MI",
    "ghvade_adaptive": "Adaptive GH-VaDE (MLP)",
    "conv_cc": "Contrastive Clustering",
}


def _supp_ablation(comparisons: pd.DataFrame) -> None:
    if comparisons.empty:
        return
    lines = [
        r"\begin{longtable}{llllrrrr}",
        r"\caption{Paired ablations. Deltas are frozen-primary minus variant.}\label{tab:supp-real-ablation-paired}\\",
        r"\toprule",
        r"Condition & Dataset & Ablation & Metric & $\Delta$ & 95\% low & 95\% high & Holm $p$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Condition & Dataset & Ablation & Metric & $\Delta$ & 95\% low & 95\% high & Holm $p$ \\",
        r"\midrule",
        r"\endhead",
    ]
    order = comparisons.sort_values(["condition", "dataset", "ablation", "metric"])
    for _, row in order.iterrows():
        p_value = row.get("holm_wilcoxon_p", np.nan)
        ablation = str(row["ablation"]).replace("_", r"\_")
        lines.append(
            f"{CONDITION_LABELS.get(row['condition'], row['condition'])} & "
            f"{DATASET_LABELS.get(row['dataset'], row['dataset'])} & {ablation} & "
            f"{str(row['metric']).upper()} & {float(row['delta_mean']):+.3f} & "
            f"{float(row['ci95_low']):+.3f} & {float(row['ci95_high']):+.3f} & "
            f"{float(p_value):.3g} \\\\" 
        )
    lines.extend((r"\bottomrule", r"\end{longtable}"))
    _write(SUBMISSION / "supp_realdata_ablation_table.tex", lines)
DATASET_LABELS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "usps": "USPS"}
CONDITION_LABELS = {
    "clean": "Clean",
    "uniform_05": r"Random 5\%",
    "uniform_10": r"Random 10\%",
    "uniform_20": r"Random 20\%",
    "salt_pepper_10": r"Salt-pepper 10\%",
    "semantic_10": r"Semantic/near 10\%",
}


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_sd(row: pd.Series, metric: str, bold: bool = False) -> str:
    value = f"{float(row[f'{metric}_mean']):.3f} $\\pm$ {float(row[f'{metric}_std']):.3f}"
    return f"\\textbf{{{value}}}" if bold else value


def _clean_table(summary: pd.DataFrame) -> None:
    clean = summary[(summary["condition"] == "clean") & summary["method"].isin(MAIN_METHODS)]
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Method & ACC & ARI & NMI \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        group = clean[clean["dataset"] == dataset]
        maxima = {metric: group[f"{metric}_mean"].max() for metric in ("acc", "ari", "nmi")}
        for method_index, method in enumerate(MAIN_METHODS):
            row = group[group["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            label = DATASET_LABELS[dataset] if method_index == 0 else ""
            values = [
                _mean_sd(row, metric, math.isclose(float(row[f"{metric}_mean"]), float(maxima[metric])))
                for metric in ("acc", "ari", "nmi")
            ]
            lines.append(f"{label} & {METHOD_LABELS[method]} & " + " & ".join(values) + r" \\")
        if dataset_index != len(DATASETS) - 1:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write(SUBMISSION / "final_realdata_clean_absolute.tex", lines)


def _robust_ari_table(summary: pd.DataFrame) -> None:
    robust = summary[(summary["condition"] != "clean") & summary["method"].isin(ROBUST_METHODS)]
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Condition & Gaussian & Student-$t$ & Adaptive GH & CC \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for condition_index, condition in enumerate(CONDITIONS[1:]):
            group = robust[(robust["dataset"] == dataset) & (robust["condition"] == condition)]
            maximum = group["ari_mean"].max()
            values = []
            for method in ROBUST_METHODS:
                row = group[group["method"] == method]
                if row.empty:
                    values.append("--")
                else:
                    row = row.iloc[0]
                    values.append(_mean_sd(row, "ari", math.isclose(float(row["ari_mean"]), float(maximum))))
            dataset_label = DATASET_LABELS[dataset] if condition_index == 0 else ""
            lines.append(
                f"{dataset_label} & {CONDITION_LABELS[condition]} & " + " & ".join(values) + r" \\")
        if dataset_index != len(DATASETS) - 1:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write(SUBMISSION / "final_realdata_robust_ari.tex", lines)


def _supp_absolute(summary: pd.DataFrame) -> None:
    ordered = summary.copy()
    ordered["condition_order"] = ordered["condition"].map({value: i for i, value in enumerate(CONDITIONS)})
    ordered["dataset_order"] = ordered["dataset"].map({value: i for i, value in enumerate(DATASETS)})
    ordered = ordered.sort_values(["condition_order", "dataset_order", "method"])
    lines = [
        r"\begin{longtable}{lllcccc}",
        r"\caption{Complete absolute real-data results.}\label{tab:supp-real-absolute}\\",
        r"\toprule",
        r"Condition & Dataset & Method & $n$ & ACC & ARI & NMI \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Condition & Dataset & Method & $n$ & ACC & ARI & NMI \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in ordered.iterrows():
        method_name = str(row["method"])
        method_display = METHOD_LABELS.get(method_name, method_name.replace("_", r"\_"))
        lines.append(
            f"{CONDITION_LABELS.get(row['condition'], row['condition'])} & "
            f"{DATASET_LABELS[row['dataset']]} & {method_display} & "
            f"{int(row['n_ok'])} & {_mean_sd(row, 'acc')} & {_mean_sd(row, 'ari')} & {_mean_sd(row, 'nmi')} \\\\" 
        )
    lines.extend((r"\bottomrule", r"\end{longtable}"))
    _write(SUBMISSION / "supp_realdata_absolute_table.tex", lines)


def _supp_paired(comparisons: pd.DataFrame) -> None:
    lines = [
        r"\begin{longtable}{llllrrrr}",
        r"\caption{Paired primary-method comparisons.}\label{tab:supp-real-paired-full}\\",
        r"\toprule",
        r"Condition & Dataset & Control & Metric & $\Delta$ & 95\% low & 95\% high & Holm $p$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Condition & Dataset & Control & Metric & $\Delta$ & 95\% low & 95\% high & Holm $p$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in comparisons.sort_values(["condition", "dataset", "control", "metric"]).iterrows():
        metric = str(row["metric"])
        p_value = row.get("holm_wilcoxon_p", np.nan)
        lines.append(
            f"{CONDITION_LABELS.get(row['condition'], row['condition'])} & {DATASET_LABELS[row['dataset']]} & "
            f"{METHOD_LABELS.get(row['control'], row['control'])} & {metric.upper()} & "
            f"{float(row[f'delta_{metric}_mean']):+.3f} & {float(row[f'delta_{metric}_ci95_low']):+.3f} & "
            f"{float(row[f'delta_{metric}_ci95_high']):+.3f} & "
            f"{float(p_value):.3g} \\\\" 
        )
    lines.extend((r"\bottomrule", r"\end{longtable}"))
    _write(SUBMISSION / "supp_realdata_paired_table.tex", lines)


def _runtime_table(summary: pd.DataFrame) -> None:
    clean = summary[(summary["condition"] == "clean") & summary["method"].isin(MAIN_METHODS)]
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Method & Wall s & CPU s & Peak MB \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        for method in MAIN_METHODS:
            row = clean[(clean["dataset"] == dataset) & (clean["method"] == method)]
            if row.empty:
                continue
            row = row.iloc[0]
            lines.append(
                f"{DATASET_LABELS[dataset]} & {METHOD_LABELS[method]} & "
                f"{float(row['runtime_seconds_mean']):.1f} $\\pm$ {float(row['runtime_seconds_std']):.1f} & "
                f"{float(row['cpu_process_seconds_mean']):.1f} & "
                f"{float(row['peak_rss_mb_mean']):.0f} \\\\" 
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write(SUBMISSION / "supp_realdata_runtime_table.tex", lines)


def _numerical_table(summary: pd.DataFrame) -> None:
    primary = summary[summary["method"] == PRIMARY].copy()
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Condition & Dataset & Runs & Prior updates & Rejected & Failed candidates \\",
        r"\midrule",
    ]
    def count(row: pd.Series, key: str) -> int:
        value = row.get(key, 0)
        return 0 if pd.isna(value) else int(value)

    for _, row in primary.sort_values(["condition", "dataset"]).iterrows():
        lines.append(
            f"{CONDITION_LABELS.get(row['condition'], row['condition'])} & "
            f"{DATASET_LABELS[row['dataset']]} & {int(row['n_ok'])} & "
            f"{count(row, 'n_prior_updates_total')} & "
            f"{count(row, 'n_rejected_prior_updates_total')} & "
            f"{count(row, 'n_failed_candidate_fits_total')} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write(SUBMISSION / "supp_realdata_numerical_table.tex", lines)


def _run_failure_table(summary: pd.DataFrame) -> None:
    failed = summary[summary["n_failed"] > 0].sort_values(
        ["condition", "dataset", "method"]
    )
    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Condition & Dataset & Method & Successful & Planned & Failed \\",
        r"\midrule",
    ]
    for _, row in failed.iterrows():
        method_name = str(row["method"])
        method_display = METHOD_LABELS.get(method_name, method_name.replace("_", r"\_"))
        lines.append(
            f"{CONDITION_LABELS.get(row['condition'], row['condition'])} & "
            f"{DATASET_LABELS[row['dataset']]} & {method_display} & "
            f"{int(row['n_ok'])} & {int(row['n_total'])} & {int(row['n_failed'])} \\\\"
        )
    if failed.empty:
        lines.append(r"All conditions & -- & -- & -- & -- & 0 \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    _write(SUBMISSION / "supp_realdata_run_failures_table.tex", lines)


def _robustness_figures(summary: pd.DataFrame) -> None:
    uniform_conditions = CONDITIONS[:4]
    x = np.asarray([0, 5, 10, 20])
    colors = {
        "conv_vade_diag": "#4C78A8",
        "conv_ghvade_mvt": "#59A14F",
        PRIMARY: "#E15759",
        "conv_cc": "#B279A2",
    }
    markers = {"conv_vade_diag": "o", "conv_ghvade_mvt": "s", PRIMARY: "^", "conv_cc": "D"}
    for metric in ("acc", "ari", "nmi"):
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False, constrained_layout=True)
        for ax, dataset in zip(axes, DATASETS):
            for method in ROBUST_METHODS:
                values = []
                errors = []
                for condition in uniform_conditions:
                    row = summary[
                        (summary["condition"] == condition)
                        & (summary["dataset"] == dataset)
                        & (summary["method"] == method)
                    ]
                    values.append(float(row[f"{metric}_mean"].iloc[0]))
                    errors.append(float(row[f"{metric}_std"].iloc[0]))
                ax.errorbar(
                    x,
                    values,
                    yerr=errors,
                    color=colors[method],
                    marker=markers[method],
                    linewidth=1.4,
                    markersize=4,
                    capsize=2,
                    label=METHOD_LABELS[method],
                )
            ax.set_title(DATASET_LABELS[dataset])
            ax.set_xlabel("Random outliers (%)")
            ax.set_xticks(x)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(metric.upper())
        handles, labels = axes[-1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
        figure.savefig(FIGURES / f"Figure_realdata_{metric}_absolute.pdf", bbox_inches="tight")
        plt.close(figure)


def _family_figure(frequencies: pd.DataFrame) -> None:
    if frequencies.empty:
        return
    pivot = frequencies.pivot_table(
        index=["condition", "dataset"], columns="selected_family", values="frequency", fill_value=0.0
    )
    pivot = pivot.reindex(
        pd.MultiIndex.from_product([CONDITIONS, DATASETS], names=["condition", "dataset"])
    ).fillna(0.0)
    figure, ax = plt.subplots(figsize=(8.0, 4.2))
    bottom = np.zeros(len(pivot))
    for family in pivot.columns:
        values = pivot[family].to_numpy()
        ax.bar(range(len(pivot)), values, bottom=bottom, label=str(family).replace("_", " "))
        bottom += values
    labels = [f"{CONDITION_LABELS[c]}\n{DATASET_LABELS[d]}" for c, d in pivot.index]
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Selection frequency")
    ax.set_ylim(0.0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    figure.tight_layout()
    figure.savefig(FIGURES / "Figure_selected_family_frequencies.pdf", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final real-data manuscript artifacts.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_analysis",
    )
    parser.add_argument(
        "--ablation-comparisons",
        type=Path,
        default=ROOT / "results" / "realdata_final_v9_ablations" / "ablation_paired_comparisons.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.analysis_dir / "absolute_results.csv")
    comparisons = pd.read_csv(args.analysis_dir / "primary_paired_comparisons.csv")
    frequencies = pd.read_csv(args.analysis_dir / "selected_family_frequencies.csv")
    ablations = (
        pd.read_csv(args.ablation_comparisons)
        if args.ablation_comparisons.is_file()
        else pd.DataFrame()
    )
    _clean_table(summary)
    _robust_ari_table(summary)
    _supp_absolute(summary)
    _supp_paired(comparisons)
    _runtime_table(summary)
    _numerical_table(summary)
    _run_failure_table(summary)
    _robustness_figures(summary)
    _family_figure(frequencies)
    _supp_ablation(ablations)


if __name__ == "__main__":
    main()
