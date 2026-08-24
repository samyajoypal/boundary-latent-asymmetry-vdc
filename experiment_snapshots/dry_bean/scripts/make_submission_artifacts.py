from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "pattern_recognition_submission"
FIG = SUB / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SIM = ROOT / "results" / "simulation_protocol_v9_shape_dominant" / "aggregate_final_core"
SIM_V9 = SIM / "v9_shape_analysis"
REAL_ROOTS = {
    "0\\%": ROOT / "results" / "realdata_image_clean20_final_v1" / "aggregate_final",
    "5\\%": ROOT / "results" / "realdata_image_robust05_final_v1" / "aggregate_final",
    "10\\%": ROOT / "results" / "realdata_image_robust10_final_v1" / "aggregate_final",
    "20\\%": ROOT / "results" / "realdata_image_robust20_final_v1" / "aggregate_final",
}

DESIGN_LABELS = {
    "g_overlap": "Gaussian control",
    "t_overlap_df05": "Student-$t$ control",
    "sn_shape_s300": "Skew-normal 3",
    "sn_shape_s500": "Skew-normal 5",
    "st_shape_df05_s300": "Skew-$t$ 3",
    "st_shape_df05_s500": "Skew-$t$ 5",
    "nig_shape_a03_s300": "NIG 3",
    "nig_shape_a03_s500": "NIG 5",
    "gh_shape_l07_a03_s300": "Full GH 3",
    "gh_shape_l07_a03_s500": "Full GH 5",
}
DESIGN_ORDER = [
    "g_overlap",
    "t_overlap_df05",
    "sn_shape_s300",
    "sn_shape_s500",
    "st_shape_df05_s300",
    "st_shape_df05_s500",
    "nig_shape_a03_s300",
    "nig_shape_a03_s500",
    "gh_shape_l07_a03_s300",
    "gh_shape_l07_a03_s500",
]
FAMILY_LABELS = {
    "gaussian": "Gaussian",
    "student_t": "Student-$t$",
    "skew_normal": "skew-normal",
    "skew_t": "skew-$t$",
    "nig": "NIG",
    "full_gh": "full GH",
}

METHOD_LABELS = {
    "vade_diag": "VaDE",
    "ghvade_adaptive": "Adaptive GH",
    "ghvade_gh_family": "GH catalogue",
    "ghvade_mvsn": "Skew-normal",
    "ghvade_mvt": "Student-$t$",
    "ghvade_tail_select": "Tail-selected",
    "conv_ghvade_mvt": "Conv Student-$t$",
    "conv_ghvade_mvt_mi": "Conv Student-$t$+MI",
    "conv_ghvade_tail_select": "Conv tail-selected",
    "conv_ghvade_tail_select_mi": "Conv tail-selected+MI",
    "conv_ghvade_conservative": "Conv conservative",
    "conv_ghvade_conservative_mi": "Conv conservative+MI",
    "dec": "DEC",
    "idec": "IDEC",
    "vae_gmm": "VAE+GMM",
    "conv_vade_diag": "ConvVaDE",
    "conv_vade_diag_mi": "ConvVaDE+MI",
    "raw_gmm": "raw GMM",
    "raw_kmeans": "raw $k$-means",
}

DATASET_LABELS = {
    "fashion_mnist": "Fashion-MNIST",
    "mnist": "MNIST",
    "usps": "USPS",
}
DATASET_ORDER = ["fashion_mnist", "mnist", "usps"]
CONTAM_ORDER = ["0\\%", "5\\%", "10\\%", "20\\%"]
CONTAM_X = {"0\\%": 0, "5\\%": 5, "10\\%": 10, "20\\%": 20}


def esc(text: object) -> str:
    s = str(text)
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, esc(method))


def fmt(x: object, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_delta(x: object, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{float(x):+.{digits}f}"


def fmt_p(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    x = float(x)
    if x < 1e-4:
        return "$<10^{-4}$"
    return f"{x:.3f}"


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_synthetic_table() -> pd.DataFrame:
    design_means = pd.read_csv(SIM_V9 / "v9_design_means.csv")
    rows = []
    for design in DESIGN_ORDER:
        row = design_means[design_means["design_id"] == design].iloc[0]
        rows.append(
            {
                "design_id": design,
                "design": DESIGN_LABELS[design],
                "true_family": row["true_family"],
                "family": FAMILY_LABELS.get(row["true_family"], esc(row["true_family"])),
                "vade_acc": row["acc_vade_diag"],
                "student_t_acc": row["acc_ghvade_mvt"],
                "adaptive_acc": row["acc_ghvade_adaptive"],
                "delta_acc": row["acc_adaptive_minus_vade"],
                "adaptive_minus_student_t_ari": row["ari_adaptive_minus_student_t"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SUB / "final_synthetic_summary.csv", index=False)
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Design & Family & VaDE ACC & Student-$t$ ACC & Adaptive ACC & $\\Delta$ ACC & $\\Delta$ ARI vs $t$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['design']} & {row['family']} & "
            f"{fmt(row['vade_acc'])} & {fmt(row['student_t_acc'])} & "
            f"{fmt(row['adaptive_acc'])} & {fmt_delta(row['delta_acc'])} & "
            f"{fmt_delta(row['adaptive_minus_student_t_ari'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    write_lines(SUB / "final_synthetic_table.tex", lines)
    return out


def load_real_metric_standing() -> pd.DataFrame:
    frames = []
    for contam, root in REAL_ROOTS.items():
        summary = pd.read_csv(root / "realdata_method_summary.csv")
        for dataset in DATASET_ORDER:
            ds = summary[summary["dataset"] == dataset]
            for metric in ("acc", "ari", "nmi"):
                mean_col = f"{metric}_mean"
                baseline = ds[ds["group"] == "baseline"].sort_values(mean_col, ascending=False).iloc[0]
                proposed = ds[ds["group"] == "proposed"].sort_values(mean_col, ascending=False).iloc[0]
                frames.append(
                    {
                        "contamination": contam,
                        "dataset": dataset,
                        "metric": metric,
                        "best_baseline": baseline["method"],
                        "baseline_mean": baseline[mean_col],
                        "best_proposed": proposed["method"],
                        "proposed_mean": proposed[mean_col],
                        "delta": proposed[mean_col] - baseline[mean_col],
                    }
                )
    out = pd.DataFrame(frames)
    out.to_csv(SUB / "final_realdata_metric_standing.csv", index=False)
    return out


def make_realdata_table(real: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Outliers & Proposed method & $\\Delta$ ACC & $\\Delta$ ARI & $\\Delta$ NMI \\\\",
        "\\midrule",
    ]
    for contam in CONTAM_ORDER:
        for dataset in DATASET_ORDER:
            cell = real[(real["contamination"] == contam) & (real["dataset"] == dataset)]
            acc = cell[cell["metric"] == "acc"].iloc[0]
            ari = cell[cell["metric"] == "ari"].iloc[0]
            nmi = cell[cell["metric"] == "nmi"].iloc[0]
            lines.append(
                f"{DATASET_LABELS[dataset]} & {contam} & {method_label(acc['best_proposed'])} & "
                f"{fmt_delta(acc['delta'])} & {fmt_delta(ari['delta'])} & {fmt_delta(nmi['delta'])} \\\\"
            )
        if contam != CONTAM_ORDER[-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_lines(SUB / "final_realdata_table.tex", lines)


def make_supplement_tables(real: pd.DataFrame) -> None:
    paired = pd.read_csv(ROOT / "tables" / "final_realdata_paired_best_baseline.csv")
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Outliers & Dataset & Metric & $\\Delta$ & 95\\% CI low & 95\\% CI high & Wilcoxon $p$ \\\\",
        "\\midrule",
    ]
    for contam in CONTAM_ORDER:
        condition = {"0\\%": "clean", "5\\%": "outlier05", "10\\%": "outlier10", "20\\%": "outlier20"}[contam]
        for dataset in DATASET_ORDER:
            for metric in ("acc", "ari", "nmi"):
                row = paired[
                    (paired["condition"] == condition)
                    & (paired["dataset"] == dataset)
                    & (paired["metric"] == metric)
                ].iloc[0]
                lines.append(
                    f"{contam} & {DATASET_LABELS[dataset]} & {metric.upper()} & "
                    f"{fmt_delta(row['delta'])} & {fmt_delta(row['ci95_low'])} & "
                    f"{fmt_delta(row['ci95_high'])} & {fmt_p(row['wilcoxon_p'])} \\\\"
                )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_lines(SUB / "supp_realdata_paired_table.tex", lines)

    paired_syn = pd.read_csv(SIM_V9 / "v9_pairwise_strata.csv")
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Stratum & Metric & Comparison & Pairs & $\\Delta$ & 95\\% CI & Wilcoxon $p$ \\\\",
        "\\midrule",
    ]
    paired_syn = paired_syn[
        (paired_syn["treatment"] == "ghvade_adaptive")
        & (paired_syn["control"].isin(["vade_diag", "ghvade_mvt"]))
    ].copy()
    paired_syn["comparison_order"] = paired_syn["control"].map({"vade_diag": 0, "ghvade_mvt": 1})
    paired_syn["stratum_order"] = paired_syn["stratum"].map(
        {"gaussian_control": 0, "student_t_control": 1, "asymmetric_cells": 2, "all_cells": 3}
    )
    paired_syn = paired_syn.sort_values(["stratum_order", "comparison_order", "metric"])
    for _, row in paired_syn.iterrows():
        comparison = "adaptive vs VaDE" if row["control"] == "vade_diag" else "adaptive vs $t$"
        ci = f"[{fmt_delta(row['delta_ci95_low'])}, {fmt_delta(row['delta_ci95_high'])}]"
        lines.append(
            f"{esc(row['stratum'])} & {str(row['metric']).upper()} & {comparison} & "
            f"{int(row['n_pairs'])} & {fmt_delta(row['delta_mean'])} & {ci} & "
            f"{fmt_p(row['wilcoxon_p'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_lines(SUB / "supp_synthetic_paired_table.tex", lines)

    failure = pd.read_csv(SIM_V9 / "v9_failures.csv")
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Method & Failed & Total & Rate \\\\", "\\midrule"]
    for _, row in failure.iterrows():
        lines.append(
            f"{method_label(row['method'])} & {int(row['n_failed'])} & {int(row['n_total'])} & {fmt(row['failure_rate'], 3)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_lines(SUB / "supp_failure_table.tex", lines)

    fam = pd.read_csv(SIM / "protocol_analysis" / "family_selection.csv")
    lines = [
        "\\begin{tabular}{lllr}",
        "\\toprule",
        "Design & True family & Selected family & Count \\\\",
        "\\midrule",
    ]
    for _, row in fam.iterrows():
        lines.append(
            f"{DESIGN_LABELS.get(row['design_id'], esc(row['design_id']))} & "
            f"{esc(row['true_family'])} & {esc(row['selected_family'])} & {int(row['count'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_lines(SUB / "supp_family_selection_table.tex", lines)


def make_figures(synthetic: pd.DataFrame, real: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = ["#4C78A8" if x >= 0 else "#B55555" for x in synthetic["delta_acc"]]
    ax.bar(range(len(synthetic)), synthetic["delta_acc"], color=colors, width=0.72)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Adaptive ACC gain over Gaussian VaDE")
    ax.set_xticks(range(len(synthetic)))
    ax.set_xticklabels([DESIGN_LABELS[x] for x in synthetic["design_id"]], rotation=35, ha="right")
    ax.set_ylim(min(-0.015, synthetic["delta_acc"].min() - 0.02), synthetic["delta_acc"].max() + 0.03)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_1_synthetic_acc_delta.pdf")
    plt.close(fig)

    for metric, filename, ylabel in [
        ("ari", "Figure_2_robustness_ari_delta.pdf", "ARI gain over strongest baseline"),
        ("acc", "Figure_3_robustness_acc_delta.pdf", "ACC gain over strongest baseline"),
    ]:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        for dataset, color, marker in [
            ("fashion_mnist", "#4C78A8", "o"),
            ("mnist", "#59A14F", "s"),
            ("usps", "#E15759", "^"),
        ]:
            vals = []
            xs = []
            for contam in CONTAM_ORDER:
                row = real[
                    (real["contamination"] == contam)
                    & (real["dataset"] == dataset)
                    & (real["metric"] == metric)
                ].iloc[0]
                xs.append(CONTAM_X[contam])
                vals.append(row["delta"])
            ax.plot(xs, vals, marker=marker, linewidth=2, color=color, label=DATASET_LABELS[dataset])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Unlabeled outlier contamination (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 5, 10, 20])
        ax.legend(frameon=False, ncol=1)
        fig.tight_layout()
        fig.savefig(FIG / filename)
        plt.close(fig)


def main() -> None:
    synthetic = make_synthetic_table()
    real = load_real_metric_standing()
    make_realdata_table(real)
    make_supplement_tables(real)
    make_figures(synthetic, real)
    print("Wrote submission tables and figures to", SUB)


if __name__ == "__main__":
    main()
