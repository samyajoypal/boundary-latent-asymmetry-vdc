"""Plot the recorded paired contrasts and representation-stage comparisons."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATASETS = ("stl10_resnet50", "cifar10_resnet50", "cifar10020_resnet50")
LABELS = ("STL-10", "CIFAR-10", "CIFAR-100-20")
METHODS = ("ghvade_adaptive", "ghvade_mvt", "ghvade_mvn")
COLORS = ("#147d83", "#b44359", "#555555")


def main():
    source = ROOT / "generated/summary.json"
    if not source.is_file():
        raise FileNotFoundError("Run current/summarize.py first")
    report = json.loads(source.read_text())
    plt.rcParams.update({"font.size":10, "font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={"width_ratios":[1,1.25]},layout="constrained")
    for ax, baseline, title, color in zip(axes,METHODS[1:],("Against Student-t","Against Gaussian"),COLORS):
        records = [next(r for r in report["simulation_primary"] if r["comparator"]==baseline)]
        records += [next(r for r in report["image_primary"] if r["dataset"]==d and r["comparator"]==baseline) for d in DATASETS]
        for i,r in enumerate(records):
            ax.errorbar(r["mean"],i,xerr=[[r["mean"]-r["ci95_low"]],[r["ci95_high"]-r["mean"]]],fmt="o",color=color,capsize=3)
        ax.axvline(0,color=".5",lw=.7)
        ax.set_yticks(range(4),["Simulation",*LABELS])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Adaptive minus comparator ARI")
        ax.grid(axis="x",alpha=.15)
    fig.savefig(ROOT / "generated/paired_ari_results.png",dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(7.2,2.6),layout="constrained")
    for ax,dataset,label in zip(axes,DATASETS,LABELS):
        for method,name,color in zip(METHODS,("Adaptive","Student-t","Full Gaussian"),COLORS):
            values = [next(r for r in report["absolute"] if r["study"]=="image" and r["design"]==dataset
                           and r["regime"]==regime and r["method"]==method)["metrics"]["ari"]["mean"]
                      for regime in ("fixed_representation","joint_training")]
            ax.plot([0,1],values,"o-",label=name,color=color,markersize=4)
        ax.set_xticks([0,1],["Fixed","Joint"])
        ax.set_xlim(-.15,1.15)
        ax.set_title(label)
        ax.grid(axis="y",alpha=.15)
    axes[0].set_ylabel("Mean ARI")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False)
    fig.savefig(ROOT / "generated/real_representation_results.png",dpi=200)
    plt.close(fig)
    print("Wrote two plots under current/generated/; no manuscript files are needed.")


if __name__ == "__main__":
    main()
