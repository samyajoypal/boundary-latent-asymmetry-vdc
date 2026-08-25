from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MANUSCRIPT = PAPER / "manuscript"
TABLES = MANUSCRIPT / "tables"
FIGURES = MANUSCRIPT / "figures"

V9 = ROOT / "results" / "shape_dominant"
REPLICATION = ROOT / "results" / "nig_orientation_replication"
CONFIRMATION = ROOT / "results" / "nig_orientation_confirmation"
REPLICATION_RUNS = ROOT / "results" / "paper2_nig_orientation_replication"
DRY_BEAN = ROOT / "results" / "paper2_dry_bean_confirmatory" / "analysis"

METHODS = ("vade_diag", "ghvade_mvt", "ghvade_adaptive")
METHOD_LABELS = {
    "vade_diag": "Gaussian VaDE",
    "ghvade_mvt": "Student-$t$ VaDE",
    "ghvade_adaptive": "Adaptive catalogue",
}
DESIGN_LABELS = {
    "g_overlap": "Gaussian control",
    "t_overlap_df05": "Student-$t$ control",
    "sn_shape_s300": "Skew-normal, $s=3$",
    "sn_shape_s500": "Skew-normal, $s=5$",
    "st_shape_df05_s300": "Skew-$t$, $s=3$",
    "st_shape_df05_s500": "Skew-$t$, $s=5$",
    "nig_shape_a03_s300": "NIG, $s=3$",
    "nig_shape_a03_s500": "NIG, $s=5$",
    "gh_shape_l07_a03_s300": "Full GH, $s=3$",
    "gh_shape_l07_a03_s500": "Full GH, $s=5$",
}
DESIGN_ORDER = tuple(DESIGN_LABELS)
ASYMMETRIC = {"skew_normal", "skew_t", "nig", "full_gh"}
DRY_BEAN_METHOD_ORDER = (
    "raw_kmeans",
    "raw_gmm",
    "dec",
    "idec",
    "vade_diag",
    "ghvade_mvt",
    "ghvade_adaptive",
)
DRY_BEAN_METHOD_LABELS = {
    "raw_kmeans": "K-means",
    "raw_gmm": "Gaussian mixture",
    "dec": "DEC",
    "idec": "IDEC",
    "vade_diag": "Gaussian VaDE",
    "ghvade_mvt": "Student-$t$ GH-VaDE",
    "ghvade_adaptive": "Adaptive GH-VaDE",
}
DRY_BEAN_FAMILY_LABELS = {
    "gaussian": "Gaussian",
    "student_t": "Student-$t$",
    "skew_normal": "Skew-normal",
    "variance_gamma": "Variance-gamma",
    "hyperbolic": "Hyperbolic",
    "nig": "NIG",
    "skew_t": "Skew-$t$",
    "full_gh": "Full GH",
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _interval(values: pd.Series | np.ndarray, confidence: float = 0.95) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    mean = float(array.mean()) if n else math.nan
    sd = float(array.std(ddof=1)) if n > 1 else math.nan
    if n > 1:
        radius = float(stats.t.ppf((1.0 + confidence) / 2.0, n - 1) * sd / math.sqrt(n))
    else:
        radius = math.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - radius,
        "ci_high": mean + radius,
    }


def _mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.3f} $\\pm$ {values.std(ddof=1):.3f}"


def _load_v9() -> pd.DataFrame:
    data = pd.read_csv(V9 / "raw_results.csv")
    data = data[data["status"].eq("ok")].copy()
    expected = pd.MultiIndex.from_product(
        [DESIGN_ORDER, range(20), METHODS], names=["design_id", "seed", "method"]
    )
    counts = data.groupby(list(expected.names)).size().reindex(expected)
    if counts.isna().any() or (counts != 1).any() or len(data) != 600:
        raise RuntimeError("The frozen v9 grid is incomplete or duplicated.")
    return data


def _seed_block_delta(data: pd.DataFrame, metric: str, designs: set[str]) -> tuple[pd.Series, dict[str, float | int]]:
    subset = data[data["design_id"].isin(designs)]
    wide = subset.pivot(index=["design_id", "seed"], columns="method", values=metric)
    delta = wide["ghvade_adaptive"] - wide["ghvade_mvt"]
    blocks = delta.groupby("seed").mean()
    return blocks, _interval(blocks)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def _overall_table(data: pd.DataFrame) -> None:
    strata = {
        "All 10 designs": set(DESIGN_ORDER),
        "Eight asymmetric designs": {
            design for design in DESIGN_ORDER if design not in {"g_overlap", "t_overlap_df05"}
        },
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Absolute clustering performance in the shape-dominant study. Values are mean $\pm$ standard deviation over the design--seed rows in each declared stratum. Each design has 20 paired seeds.}",
        r"\label{tab:v9-overall}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Stratum & Method & ACC & ARI & NMI \\",
        r"\midrule",
    ]
    for stratum_index, (stratum, designs) in enumerate(strata.items()):
        subset = data[data["design_id"].isin(designs)]
        means = subset.groupby("method")[["acc", "ari", "nmi"]].mean()
        maxima = means.max(axis=0)
        for method_index, method in enumerate(METHODS):
            row = subset[subset["method"].eq(method)]
            values = []
            for metric in ("acc", "ari", "nmi"):
                cell = _mean_sd(row[metric])
                if np.isclose(means.loc[method, metric], maxima[metric]):
                    cell = r"\textbf{" + cell + "}"
                values.append(cell)
            prefix = stratum if method_index == 0 else ""
            lines.append(
                f"{prefix} & {METHOD_LABELS[method]} & " + " & ".join(values) + r" \\"
            )
        if stratum_index == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write(TABLES / "v9_overall.tex", "\n".join(lines))


def _design_table(data: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Design-level ARI in the 20-seed shape-dominant study. The final column is the paired adaptive-minus-Student-$t$ mean. Positive entries in every asymmetric row are descriptive; confirmatory inference is performed on seed-block averages in Table~\ref{tab:v9-effects}.}",
        r"\label{tab:v9-designs}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Generating design & Gaussian VaDE & Student-$t$ VaDE & Adaptive catalogue & $\Delta$ adaptive--$t$ \\",
        r"\midrule",
    ]
    for design in DESIGN_ORDER:
        subset = data[data["design_id"].eq(design)]
        wide = subset.pivot(index="seed", columns="method", values="ari")
        cells = [_mean_sd(wide[method]) for method in METHODS]
        delta = (wide["ghvade_adaptive"] - wide["ghvade_mvt"]).mean()
        lines.append(
            f"{DESIGN_LABELS[design]} & " + " & ".join(cells) + f" & {delta:+.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    _write(TABLES / "v9_designs.tex", "\n".join(lines))


def _effect_table(data: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    asym_designs = {d for d in DESIGN_ORDER if d not in {"g_overlap", "t_overlap_df05"}}
    summaries: dict[str, dict[str, float | int]] = {}
    p_values: dict[str, float] = {}
    blocks_by_metric: dict[str, pd.Series] = {}
    for metric in ("acc", "ari", "nmi"):
        blocks, summary = _seed_block_delta(data, metric, asym_designs)
        blocks_by_metric[metric] = blocks
        summaries[metric] = summary
        p_values[metric] = float(stats.ttest_1samp(blocks, 0.0).pvalue)
    adjusted = _holm(p_values)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adaptive-catalogue minus fixed Student-$t$ effects over the eight asymmetric designs. Inference uses one average per seed across designs ($n=20$), avoiding pseudoreplication. Holm correction covers ACC, ARI, and NMI.}",
        r"\label{tab:v9-effects}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Metric & Mean $\Delta$ & 95\% CI & Holm $p$ & Positive seeds \\",
        r"\midrule",
    ]
    for metric in ("acc", "ari", "nmi"):
        summary = summaries[metric]
        blocks = blocks_by_metric[metric]
        lines.append(
            f"{metric.upper()} & {summary['mean']:+.4f} & "
            f"[{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}] & "
            f"{adjusted[metric]:.4g} & {int((blocks > 0).sum())}/20 \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write(TABLES / "v9_effects.tex", "\n".join(lines))
    return summaries


def _selection_table(data: pd.DataFrame) -> None:
    selector = data[data["method"].eq("ghvade_adaptive")].copy()
    families = ("gaussian", "student_t", "skew_normal", "hyperbolic", "nig", "skew_t", "full_gh")
    truth_order = ("gaussian", "student_t", "skew_normal", "skew_t", "nig", "full_gh")
    counts = pd.crosstab(selector["true_family"], selector["selected_family"])
    lines = [
        r"\begin{longtable}{lrrrrrrr}",
        r"\caption{Final selected-family counts in the shape-dominant study. Each control truth has 20 runs and each asymmetric truth has 40 runs (two strengths). Variance-gamma was never selected.}\label{tab:selection-counts}\\",
        r"\toprule",
        r"Truth & Gaussian & Student-$t$ & Skew-normal & Hyperbolic & NIG & Skew-$t$ & Full GH \\",
        r"\midrule",
    ]
    for truth in truth_order:
        values = [int(counts.get(family, pd.Series(dtype=int)).get(truth, 0)) for family in families]
        label = {"student_t": "Student-$t$", "skew_normal": "Skew-normal", "skew_t": "Skew-$t$", "full_gh": "Full GH", "nig": "NIG", "gaussian": "Gaussian"}[truth]
        lines.append(label + " & " + " & ".join(str(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    _write(TABLES / "selection_counts.tex", "\n".join(lines))


def _orientation_tables() -> dict[str, object]:
    confirmation = json.loads((CONFIRMATION / "gate.json").read_text(encoding="utf-8"))
    replication = json.loads((REPLICATION / "gate.json").read_text(encoding="utf-8"))
    rows = [
        ("Initial confirmation", confirmation),
        ("Independent replication", replication),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Prospective NIG-orientation studies at oracle latent variables. Superiority endpoints use 95\% CIs; the outward equivalence endpoint uses a 90\% CI and margin $[-0.01,0.01]$. The replication's three superiority tests were Holm-adjusted.}",
        r"\label{tab:orientation}",
        r"\begin{tabular}{llrcl}",
        r"\toprule",
        r"Study & Endpoint & $n$ & Mean $\Delta$ (CI) & Prespecified conclusion \\",
        r"\midrule",
    ]
    for label, gate in rows:
        score = gate["boundary_nig_score_gain"]
        boundary = gate["boundary_nig_ari_gain"]
        orientation = gate["boundary_minus_outward_ari_gain"]
        outward = gate["outward_nig_ari_gain_90_ci"]
        is_replication = label == "Independent replication"
        endpoint_rows = (
            ("Boundary log score", score, "Supports superiority" if is_replication else "Inconclusive"),
            ("Boundary ARI", boundary, "Supports superiority" if is_replication else "Inconclusive"),
            ("Boundary--outward ARI", orientation, "Supports orientation effect" if is_replication else "Inconclusive"),
            (r"Outward ARI (90\% CI)", outward, "Equivalence not established" if is_replication else "Supports equivalence"),
        )
        for endpoint_index, (endpoint, result, conclusion) in enumerate(endpoint_rows):
            study = label if endpoint_index == 0 else ""
            lines.append(
                f"{study} & {endpoint} & {result['n']} & "
                f"{result['mean']:+.3f} [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}] & "
                f"{conclusion} \\\\"
            )
        if not is_replication:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    _write(TABLES / "orientation_studies.tex", "\n".join(lines))

    data = pd.read_csv(REPLICATION / "diagnostics_all.csv")
    oracle = data[data["representation_stage"].eq("oracle")]
    abs_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Absolute oracle-stage clustering results in the independent replication, restricted to valid NIG-versus-Student-$t$ comparisons.}",
        r"\label{tab:orientation-absolute}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Orientation & Fitted family & ACC & ARI & NMI \\",
        r"\midrule",
    ]
    for design, design_label in (("nig_boundary", "Boundary-facing"), ("nig_outward", "Outward")):
        valid_seeds = set(
            oracle[
                oracle["design"].eq(design)
                & oracle["family"].eq("nig")
                & oracle["comparison_valid"]
            ]["seed"]
        )
        subset = oracle[
            oracle["design"].eq(design)
            & oracle["seed"].isin(valid_seeds)
            & oracle["family"].isin(("student_t", "nig"))
        ]
        for index, family in enumerate(("student_t", "nig")):
            row = subset[subset["family"].eq(family)]
            prefix = design_label if index == 0 else ""
            family_label = "Student-$t$" if family == "student_t" else "NIG"
            abs_lines.append(
                f"{prefix} & {family_label} & {_mean_sd(row['acc'])} & "
                f"{_mean_sd(row['ari'])} & {_mean_sd(row['nmi'])} \\\\"
            )
        if design == "nig_boundary":
            abs_lines.append(r"\midrule")
    abs_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write(TABLES / "orientation_absolute.tex", "\n".join(abs_lines))

    return replication


def _numerical_table() -> None:
    data = pd.read_csv(REPLICATION / "diagnostics_all.csv")
    elapsed_path = REPLICATION / "task_elapsed_seconds.csv"
    if elapsed_path.exists():
        elapsed_frame = pd.read_csv(elapsed_path)
        elapsed = pd.to_numeric(
            elapsed_frame["elapsed_seconds"], errors="coerce"
        ).dropna().tolist()
    else:
        elapsed_rows: list[dict[str, float | str]] = []
        for path in sorted(REPLICATION_RUNS.glob("*/manifest.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            value = record.get("elapsed_seconds")
            if value is not None and math.isfinite(float(value)):
                elapsed_rows.append(
                    {"task_id": path.parent.name, "elapsed_seconds": float(value)}
                )
        pd.DataFrame(elapsed_rows).to_csv(elapsed_path, index=False)
        elapsed = [float(row["elapsed_seconds"]) for row in elapsed_rows]
    if len(elapsed) != 300:
        raise RuntimeError(f"Expected 300 elapsed-time records, found {len(elapsed)}.")
    grouped = data.groupby("family")["fit_seconds"].agg(["mean", "median"])
    invalid = data.assign(invalid=~data["fit_valid"]).groupby("family")["invalid"].agg(["sum", "count"])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Numerical and computational audit for the 100-seed replication. Fit times are seconds per family--representation fit; task wall time includes neural training and all 15 diagnostic fits.}",
        r"\label{tab:numerical}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Family & Mean fit time & Median fit time & Invalid fits \\",
        r"\midrule",
    ]
    family_labels = {
        "gaussian": "Gaussian",
        "student_t": "Student-$t$",
        "nig": "NIG",
        "skew_t": "Skew-$t$",
        "full_gh": "Full GH",
    }
    for family in ("gaussian", "student_t", "nig", "skew_t", "full_gh"):
        lines.append(
            f"{family_labels[family]} & {grouped.loc[family, 'mean']:.2f} & "
            f"{grouped.loc[family, 'median']:.2f} & "
            f"{int(invalid.loc[family, 'sum'])}/{int(invalid.loc[family, 'count'])} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"Complete task wall time & {np.mean(elapsed):.2f} & {np.median(elapsed):.2f} & 0/300 task failures \\\\ ",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    _write(TABLES / "numerical_audit.tex", "\n".join(lines))


def _stage_and_validity_tables() -> None:
    data = pd.read_csv(REPLICATION / "diagnostics_all.csv")
    valid = data[
        data["family"].eq("nig")
        & data["design"].isin(("nig_boundary", "nig_outward"))
        & data["comparison_valid"]
    ]
    lines = [
        r"\begin{supplementtable}{tab:stage-effects}{NIG-minus-Student-$t$ effects by representation stage in the independent replication. Intervals are 95\% Student-$t$ intervals over valid seed-level comparisons.}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Orientation & Stage & Held-out score $\Delta$ & ARI $\Delta$ & Decision-change fraction \\",
        r"\midrule",
    ]
    for design, label in (("nig_boundary", "Boundary-facing"), ("nig_outward", "Outward")):
        for stage_index, stage in enumerate(("oracle", "pretrained", "final")):
            row = valid[valid["design"].eq(design) & valid["representation_stage"].eq(stage)]
            cells = []
            for metric in (
                "delta_log_score_vs_student_t",
                "delta_ari_vs_student_t",
                "decision_change_fraction_vs_student_t",
            ):
                result = _interval(row[metric])
                cells.append(
                    f"{result['mean']:+.3f} [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"
                )
            prefix = label if stage_index == 0 else ""
            stage_label = {"oracle": "Oracle", "pretrained": "Pretrained", "final": "Final"}[stage]
            lines.append(f"{prefix} & {stage_label} & " + " & ".join(cells) + r" \\")
        if design == "nig_boundary":
            lines.append(r"\midrule")
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", r"\end{supplementtable}"]
    )
    _write(TABLES / "stage_effects.tex", "\n".join(lines))

    audit = data.assign(invalid=~data["fit_valid"]).groupby(
        ["family", "representation_stage"]
    )["invalid"].agg(["sum", "count"])
    family_labels = {
        "gaussian": "Gaussian",
        "student_t": "Student-$t$",
        "nig": "NIG",
        "skew_t": "Skew-$t$",
        "full_gh": "Full GH",
    }
    validity = [
        r"\begin{longtable}{lrrr}",
        r"\caption{Invalid fits by candidate family and representation stage in the independent replication. Each cell has 300 fits.}\label{tab:validity-stage}\\",
        r"\toprule",
        r"Family & Oracle & Pretrained & Final \\",
        r"\midrule",
    ]
    for family in ("gaussian", "student_t", "nig", "skew_t", "full_gh"):
        values = [int(audit.loc[(family, stage), "sum"]) for stage in ("oracle", "pretrained", "final")]
        validity.append(f"{family_labels[family]} & " + " & ".join(map(str, values)) + r" \\")
    validity.extend([r"\bottomrule", r"\end{longtable}"])
    _write(TABLES / "validity_by_stage.tex", "\n".join(validity))


def _v9_effect_figure(data: pd.DataFrame) -> None:
    records = []
    for design in DESIGN_ORDER:
        subset = data[data["design_id"].eq(design)]
        wide = subset.pivot(index="seed", columns="method", values="ari")
        summary = _interval(wide["ghvade_adaptive"] - wide["ghvade_mvt"])
        records.append((design, summary))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(records))
    means = np.array([item[1]["mean"] for item in records], dtype=float)
    lows = np.array([item[1]["ci_low"] for item in records], dtype=float)
    highs = np.array([item[1]["ci_high"] for item in records], dtype=float)
    colors = ["#6f7782" if item[0] in {"g_overlap", "t_overlap_df05"} else "#167c80" for item in records]
    ax.axvline(0.0, color="#1e1e1e", lw=0.9)
    for index, color in enumerate(colors):
        ax.errorbar(
            means[index],
            y[index],
            xerr=np.array(
                [[means[index] - lows[index]], [highs[index] - means[index]]]
            ),
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
        )
        ax.scatter(means[index], y[index], color=color, s=35, zorder=3)
    ax.set_yticks(y, [DESIGN_LABELS[item[0]].replace("$", "") for item in records])
    ax.invert_yaxis()
    ax.set_xlabel("Adaptive catalogue minus Student-t ARI")
    ax.grid(axis="x", color="#d9dde2", linewidth=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "v9_design_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def _orientation_figure() -> None:
    data = pd.read_csv(REPLICATION / "diagnostics_all.csv")
    subset = data[
        data["family"].eq("nig")
        & data["design"].isin(("nig_boundary", "nig_outward"))
        & data["comparison_valid"]
    ]
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    stages = ("oracle", "pretrained", "final")
    x = np.arange(len(stages))
    styles = {
        "nig_boundary": ("Boundary-facing NIG", "#167c80", "o"),
        "nig_outward": ("Outward NIG", "#b64b43", "s"),
    }
    for design, (label, color, marker) in styles.items():
        means, lower, upper = [], [], []
        for stage in stages:
            summary = _interval(
                subset[
                    subset["design"].eq(design)
                    & subset["representation_stage"].eq(stage)
                ]["delta_ari_vs_student_t"]
            )
            means.append(float(summary["mean"]))
            lower.append(float(summary["mean"]) - float(summary["ci_low"]))
            upper.append(float(summary["ci_high"]) - float(summary["mean"]))
        ax.errorbar(x, means, yerr=np.vstack([lower, upper]), color=color, marker=marker, lw=1.7, capsize=4, label=label)
    ax.axhline(0.0, color="#1e1e1e", lw=0.9)
    ax.set_xticks(x, ["Oracle latent", "Pretrained latent", "Final latent"])
    ax.set_ylabel("NIG minus Student-t ARI")
    ax.grid(axis="y", color="#d9dde2", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "orientation_by_stage.pdf", bbox_inches="tight")
    plt.close(fig)


def _latent_geometry_figure() -> None:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from ghmvae.data.synthetic import make_parametric_mixture

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True, sharey=True)
    settings = (
        ("clockwise_neighbor", "Boundary-facing NIG"),
        ("radial", "Outward NIG"),
    )
    palette = ("#277da1", "#e07a5f", "#3d9970")
    for ax, (orientation, title) in zip(axes, settings):
        _, labels, latent, truth = make_parametric_mixture(
            n_samples=1800,
            input_dim=16,
            latent_dim=2,
            n_components=3,
            family="nig",
            noise=0.12,
            center_scale=0.65,
            cluster_scale=0.75,
            skew_strength=5.0,
            tail_df=5.0,
            alpha_bar=0.3,
            gh_lambda=0.7,
            covariance_ratio=1.5,
            class_probs=(1.0, 1.0, 1.0),
            hidden_width=96,
            seed=8000,
            component_families=("student_t", "student_t", "nig"),
            skew_orientation=orientation,
        )
        z = latent.numpy()
        y = labels.numpy()
        for component, color in enumerate(palette):
            points = z[y == component]
            ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.23, color=color, rasterized=True)
        target = truth["components"][2]
        center = np.asarray(target["mu"], dtype=float) + np.asarray(target["beta"], dtype=float)
        beta = np.asarray(target["beta"], dtype=float)
        ax.annotate("", xy=center + 0.55 * beta, xytext=center, arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#111111"})
        ax.scatter([center[0]], [center[1]], marker="x", s=55, color="#111111", linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("True latent coordinate 1")
        ax.set_xlim(-7.0, 10.0)
        ax.set_ylim(-9.0, 9.0)
        ax.grid(color="#e2e5e9", linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("True latent coordinate 2")
    fig.tight_layout()
    fig.savefig(FIGURES / "nig_orientation_geometry.pdf", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _wall_time_cell(mean_seconds: float, sd_seconds: float) -> str:
    if mean_seconds >= 3600.0:
        return f"{mean_seconds / 3600.0:.2f} $\\pm$ {sd_seconds / 3600.0:.2f} h"
    if mean_seconds >= 120.0:
        return f"{mean_seconds / 60.0:.2f} $\\pm$ {sd_seconds / 60.0:.2f} min"
    return f"{mean_seconds:.2f} $\\pm$ {sd_seconds:.2f} s"


def _p_value_cell(value: float) -> str:
    return "$<10^{-4}$" if value < 1e-4 else f"{value:.4f}"


def _dry_bean_outputs() -> dict[str, object]:
    summary_path = DRY_BEAN / "dry_bean_confirmatory_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Run scripts/analyze_paper2_dry_bean_confirmatory.py before generating Paper 2."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("analysis_mode") != "descriptive_fallback":
        raise RuntimeError("Dry Bean outputs must come from descriptive-fallback mode.")
    if summary.get("pilot_gate_passed") is not False or summary.get("n_rows") != 140:
        raise RuntimeError("The failed pilot gate or complete 140-task matrix is not recorded.")

    methods = pd.read_csv(DRY_BEAN / "dry_bean_method_summary.csv")
    if set(methods["method"]) != set(DRY_BEAN_METHOD_ORDER):
        raise RuntimeError("Dry Bean method table does not contain the frozen seven methods.")
    methods = methods.set_index("method").loc[list(DRY_BEAN_METHOD_ORDER)]
    maxima = methods[["acc_mean", "ari_mean", "nmi_mean"]].max(axis=0)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dry Bean real-data study over 20 unseen paired seeds. Values are mean $\pm$ standard deviation. The original label-blinded promotion gate failed; these external metrics are therefore descriptive and are retained regardless of direction.}",
        r"\label{tab:dry-bean-performance}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & ACC & ARI & NMI & Wall time per run \\",
        r"\midrule",
    ]
    for method, row in methods.iterrows():
        cells = []
        for metric in ("acc", "ari", "nmi"):
            cell = f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_sd']:.3f}"
            if np.isclose(row[f"{metric}_mean"], maxima[f"{metric}_mean"]):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        runtime = _wall_time_cell(
            float(row["runtime_seconds_mean"]), float(row["runtime_seconds_sd"])
        )
        lines.append(
            f"{DRY_BEAN_METHOD_LABELS[method]} & "
            + " & ".join(cells)
            + f" & {runtime} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write(TABLES / "dry_bean_performance.tex", "\n".join(lines))

    contrasts = pd.read_csv(DRY_BEAN / "dry_bean_paired_contrasts.csv")
    if len(contrasts) != 6 or contrasts["holm_p"].isna().any():
        raise RuntimeError("Dry Bean paired contrasts are incomplete or unadjusted.")
    comparator_labels = {
        "ghvade_mvt": "Student-$t$ GH-VaDE",
        "vade_diag": "Gaussian VaDE",
    }
    contrast_lines = [
        r"\begin{longtable}{llrcrr}",
        r"\caption{Paired adaptive-GH-VaDE differences in the Dry Bean real-data study. Confidence intervals are two-sided 95\% Student-$t$ intervals over 20 seed pairs. Holm adjustment covers all six exploratory tests.}\label{tab:dry-bean-contrasts}\\",
        r"\toprule",
        r"Comparator & Metric & Mean $\Delta$ & 95\% CI & Holm $p$ & Adaptive wins \\",
        r"\midrule",
    ]
    for comparator in ("ghvade_mvt", "vade_diag"):
        subset = contrasts[contrasts["comparator"].eq(comparator)].set_index("metric")
        for metric_index, metric in enumerate(("acc", "ari", "nmi")):
            row = subset.loc[metric]
            label = comparator_labels[comparator] if metric_index == 0 else ""
            contrast_lines.append(
                f"{label} & {metric.upper()} & {row['mean_difference']:+.4f} & "
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] & "
                f"{_p_value_cell(float(row['holm_p']))} & "
                f"{int(row['adaptive_wins'])}/20 \\\\"
            )
        if comparator == "ghvade_mvt":
            contrast_lines.append(r"\midrule")
    contrast_lines.extend([r"\bottomrule", r"\end{longtable}"])
    _write(TABLES / "dry_bean_contrasts.tex", "\n".join(contrast_lines))

    stage = pd.read_csv(DRY_BEAN / "dry_bean_selector_stages.csv")
    if len(stage) != 20:
        raise RuntimeError("Expected 20 Dry Bean selector-stage rows.")
    pilot = pd.read_csv(DRY_BEAN / "dry_bean_pilot_selector_stages.csv")
    if len(pilot) != 5 or not pilot["external_metrics_blinded"].all():
        raise RuntimeError("Expected five blinded Dry Bean pilot selector histories.")
    pilot_initial = pilot["initial_active_family"]
    pilot_final = pilot["final_active_family"]

    def family_counts(values: list[str] | pd.Series) -> list[int]:
        counts = pd.Series(list(values), dtype=str).value_counts()
        return [int(counts.get(family, 0)) for family in ("gaussian", "student_t", "skew_normal")]

    stage_rows = (
        ("Blinded pilot", "Initial fit", 5, family_counts(pilot_initial)),
        ("", "Final cycle", 5, family_counts(pilot_final)),
        (
            "Unseen-seed study",
            "Initial fit",
            20,
            family_counts(stage["initial_active_family"]),
        ),
        ("", "Final cycle", 20, family_counts(stage["final_active_family"])),
    )
    stage_lines = [
        r"\begin{longtable}{llrrrr}",
        r"\caption{Dry Bean active-family transitions. Family selection uses no class label. Families absent from all four rows are omitted.}\label{tab:dry-bean-selector-stages}\\",
        r"\toprule",
        r"Study & Stage & $n$ & Gaussian & Student-$t$ & Skew-normal \\",
        r"\midrule",
    ]
    for row_index, (study, stage_name, n_runs, counts) in enumerate(stage_rows):
        stage_lines.append(
            f"{study} & {stage_name} & {n_runs} & "
            + " & ".join(str(value) for value in counts)
            + r" \\"
        )
        if row_index == 1:
            stage_lines.append(r"\midrule")
    stage_lines.extend([r"\bottomrule", r"\end{longtable}"])
    _write(TABLES / "dry_bean_selector_stages.tex", "\n".join(stage_lines))

    candidates = pd.read_csv(DRY_BEAN / "dry_bean_candidate_audit.csv")
    if int(candidates["attempts"].sum()) != 640:
        raise RuntimeError("Expected 640 adaptive candidate-fit attempts.")
    candidate_lines = [
        r"\begin{longtable}{lrrrr}",
        r"\caption{Candidate-fit audit for the 20-run Dry Bean adaptive procedure. Attempt time includes failed attempts. Every successful fit required the alternate K-means initialization after the primary Gaussian-mixture initialization returned no worked distribution.}\label{tab:dry-bean-candidate-audit}\\",
        r"\toprule",
        r"Candidate & Successful/attempted & Failed & Alternate successes & Mean attempt time (s) \\",
        r"\midrule",
    ]
    family_order = (
        "gaussian",
        "student_t",
        "skew_normal",
        "variance_gamma",
        "hyperbolic",
        "nig",
        "skew_t",
        "full_gh",
    )
    candidates = candidates.set_index("candidate")
    for family in family_order:
        row = candidates.loc[family]
        candidate_lines.append(
            f"{DRY_BEAN_FAMILY_LABELS[family]} & "
            f"{int(row['successful_fits'])}/{int(row['attempts'])} & "
            f"{int(row['failed_fits'])} & "
            f"{int(row['alternate_initialization_successes'])} & "
            f"{row['fit_seconds_mean']:.1f} \\\\"
        )
    candidate_lines.extend([r"\bottomrule", r"\end{longtable}"])
    _write(TABLES / "dry_bean_candidate_audit.tex", "\n".join(candidate_lines))

    boundary_source = DRY_BEAN / "dry_bean_boundary_margin.pdf"
    if not boundary_source.exists():
        raise FileNotFoundError(boundary_source)
    shutil.copy2(boundary_source, FIGURES / "dry_bean_boundary_margin.pdf")
    return {"summary": summary, "methods": methods, "contrasts": contrasts}


def _macros(
    v9_summaries: dict[str, dict[str, float | int]],
    replication: dict[str, object],
    dry_bean: dict[str, object],
) -> None:
    ari = v9_summaries["ari"]
    orientation = replication["boundary_minus_outward_ari_gain"]
    boundary = replication["boundary_nig_ari_gain"]
    score = replication["boundary_nig_score_gain"]
    dry_summary = dry_bean["summary"]
    dry_methods = dry_bean["methods"]
    dry_contrasts = dry_bean["contrasts"]
    assert isinstance(dry_summary, dict)
    assert isinstance(dry_methods, pd.DataFrame)
    assert isinstance(dry_contrasts, pd.DataFrame)
    dry_t = dry_contrasts[dry_contrasts["comparator"].eq("ghvade_mvt")].set_index(
        "metric"
    )
    dry_selector = dry_summary["selector_stage_summary"]
    dry_numerical = dry_summary["numerical_audit"][0]
    lines = [
        f"\\newcommand{{\\VnineAriDelta}}{{{ari['mean']:+.4f}}}",
        f"\\newcommand{{\\VnineAriCi}}{{[{ari['ci_low']:+.4f}, {ari['ci_high']:+.4f}]}}",
        f"\\newcommand{{\\OrientationAriDelta}}{{{orientation['mean']:+.4f}}}",
        f"\\newcommand{{\\OrientationAriCi}}{{[{orientation['ci_low']:+.4f}, {orientation['ci_high']:+.4f}]}}",
        f"\\newcommand{{\\BoundaryAriDelta}}{{{boundary['mean']:+.4f}}}",
        f"\\newcommand{{\\BoundaryAriCi}}{{[{boundary['ci_low']:+.4f}, {boundary['ci_high']:+.4f}]}}",
        f"\\newcommand{{\\BoundaryScoreDelta}}{{{score['mean']:+.4f}}}",
        f"\\newcommand{{\\BoundaryScoreCi}}{{[{score['ci_low']:+.4f}, {score['ci_high']:+.4f}]}}",
        f"\\newcommand{{\\ReplicationInvalidRate}}{{{100.0 * float(replication['invalid_fit_rate']):.2f}\\%}}",
        f"\\newcommand{{\\DryBeanAdaptiveTAccDelta}}{{{dry_t.loc['acc', 'mean_difference']:+.4f}}}",
        f"\\newcommand{{\\DryBeanAdaptiveTAriDelta}}{{{dry_t.loc['ari', 'mean_difference']:+.4f}}}",
        f"\\newcommand{{\\DryBeanAdaptiveTNmiDelta}}{{{dry_t.loc['nmi', 'mean_difference']:+.4f}}}",
        f"\\newcommand{{\\DryBeanBoundaryConcentration}}{{{dry_selector['final_boundary_concentration_mean']:.2f}}}",
        f"\\newcommand{{\\DryBeanDecisionChangeRate}}{{{100.0 * dry_selector['final_decision_change_fraction_mean']:.2f}\\%}}",
        f"\\newcommand{{\\DryBeanFinalShapeMargin}}{{{dry_selector['final_asymmetric_margin_mean']:+.4f}}}",
        f"\\newcommand{{\\DryBeanCandidateFailures}}{{{int(dry_numerical['failed_candidate_fits_total'])}}}",
        f"\\newcommand{{\\DryBeanCandidateRuntimeShare}}{{{100.0 * dry_numerical['candidate_fit_share_of_wall_time']:.2f}\\%}}",
        f"\\newcommand{{\\DryBeanAdaptiveRuntimeHours}}{{{dry_methods.loc['ghvade_adaptive', 'runtime_seconds_mean'] / 3600.0:.2f}}}",
        f"\\newcommand{{\\DryBeanStudentRuntimeMinutes}}{{{dry_methods.loc['ghvade_mvt', 'runtime_seconds_mean'] / 60.0:.2f}}}",
    ]
    _write(TABLES / "results_values.tex", "\n".join(lines))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = _load_v9()
    _overall_table(data)
    _design_table(data)
    summaries = _effect_table(data)
    _selection_table(data)
    replication = _orientation_tables()
    _numerical_table()
    _stage_and_validity_tables()
    _v9_effect_figure(data)
    _orientation_figure()
    _latent_geometry_figure()
    dry_bean = _dry_bean_outputs()
    _macros(summaries, replication, dry_bean)
    print(f"Wrote Paper 2 artifacts to {MANUSCRIPT}")


if __name__ == "__main__":
    main()
