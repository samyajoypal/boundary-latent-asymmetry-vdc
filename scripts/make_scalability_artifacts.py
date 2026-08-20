from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "pattern_recognition_submission"
METHOD_LABELS = {
    "gaussian": "Diagonal Gaussian",
    "student_t": "Student-$t$",
    "adaptive": "Adaptive GH",
}
CELL_ORDER = ("n0500", "n1000", "n3000", "d02", "d10", "k03", "k10")
CELL_LABELS = {
    "n0500": "$n=500$",
    "n1000": "$n=1{,}000$",
    "n3000": "$n=3{,}000$",
    "d02": "$d=2$",
    "d10": "$d=10$",
    "k03": "$K=3$",
    "k10": "$K=10$",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper artifacts for the scalability study.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "scalability_v9_summary.csv",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Setting & Fitter & Time (s) & SD (s) & Peak MB & Failures \\",
        r"\midrule",
    ]
    for cell in CELL_ORDER:
        subset = frame[frame["cell_id"] == cell]
        for index, method in enumerate(("gaussian", "student_t", "adaptive")):
            row = subset[subset["method"] == method].iloc[0]
            setting = CELL_LABELS[cell] if index == 0 else ""
            lines.append(
                f"{setting} & {METHOD_LABELS[method]} & "
                f"{row['elapsed_seconds_mean']:.2f} & {row['elapsed_seconds_std']:.2f} & "
                f"{row['peak_rss_mb_mean']:.1f} & {int(row['n_failed'])} \\\\" 
            )
        if cell != CELL_ORDER[-1]:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (SUBMISSION / "supp_scalability_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    figure, ax = plt.subplots(figsize=(7.0, 3.6))
    x = range(len(CELL_ORDER))
    for method, color, marker in (
        ("gaussian", "#4C78A8", "o"),
        ("student_t", "#59A14F", "s"),
        ("adaptive", "#E15759", "^"),
    ):
        values = [
            float(frame[(frame["cell_id"] == cell) & (frame["method"] == method)]["elapsed_seconds_mean"].iloc[0])
            for cell in CELL_ORDER
        ]
        ax.plot(x, values, color=color, marker=marker, linewidth=1.8, label=METHOD_LABELS[method])
    ax.set_yscale("log")
    ax.set_xticks(list(x), [CELL_LABELS[cell].replace("$", "") for cell in CELL_ORDER])
    ax.set_ylabel("Prior-fitting time (seconds, log scale)")
    ax.set_xlabel("One-factor setting; reference is n=1,000, d=5, K=5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(SUBMISSION / "figures" / "Figure_S1_scalability_runtime.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()

