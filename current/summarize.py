"""Recompute published means, sample SDs, and paired contrasts from all seed scores."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from run import verify_source

ROOT = Path(__file__).resolve().parent
METRICS = ("ari", "acc", "nmi", "homogeneity")
BASELINES = ("ghvade_mvt", "ghvade_mvn")


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (20,) or not np.isfinite(values).all():
        raise ValueError("Twenty finite values are required")
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)))


def contrast(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (20,) or not np.isfinite(x).all():
        raise ValueError("Twenty finite paired differences are required")
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(stats.t.ppf(.975, 19) * sd / np.sqrt(20))
    p = float(stats.ttest_1samp(x, 0).pvalue) if sd else (1. if mean == 0 else 0.)
    return dict(mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half, p=p)


def holm(rows):
    running = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r: r["p"])):
        running = max(running, (len(rows)-rank)*row["p"])
        row["holm_p"] = min(1., running)


def main():
    verify_source()
    rows = json.loads((ROOT / "results/scores.json").read_text())
    plan = json.loads((ROOT / "protocols/studies.json").read_text())
    reference = json.loads((ROOT / "results/reference_contrasts.json").read_text())
    indexed, groups = {}, defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("study", "design", "regime", "method", "seed"))
        if key in indexed:
            raise ValueError("Duplicate run")
        indexed[key] = row
        groups[key[:-1]].append(row)
    if len(indexed) != 3240:
        raise ValueError("Incomplete result panel")
    absolute = []
    for key, records in sorted(groups.items()):
        expected = set(plan["simulation_seeds" if key[0] == "simulation" else "image_seeds"])
        if {r["seed"] for r in records} != expected:
            raise ValueError("Incomplete seed coverage")
        absolute.append(dict(zip(("study", "design", "regime", "method"), key), metrics={
            m: describe([r[m] for r in records]) for m in METRICS}))

    def differences(study, designs, regime, metric, comparator):
        seeds = plan["simulation_seeds" if study == "simulation" else "image_seeds"]
        return [np.mean([indexed[study, d, regime, "ghvade_adaptive", s][metric]
                         - indexed[study, d, regime, comparator, s][metric] for d in designs]) for s in seeds]

    output = dict(absolute=absolute)
    for category, metrics in (("image_primary", ("ari",)), ("image_secondary", METRICS[1:])):
        records = []
        for dataset in plan["data_hashes"]:
            for metric in metrics:
                for comparator in BASELINES:
                    records.append(dict(dataset=dataset, metric=metric, comparator=comparator,
                        **contrast(differences("image", [dataset], "joint_training", metric, comparator))))
        holm(records)
        for result in records:
            ref = next(r for r in reference[category] if r["dataset"] == result["dataset"]
                       and r["comparator"] == result["comparator"] and r.get("metric", "ari") == result["metric"])
            for k in ("mean", "ci95_low", "ci95_high", "holm_p"):
                np.testing.assert_allclose(result[k], ref[k], rtol=1e-9, atol=1e-12)
        output[category] = records
    for category, metrics, regime in (("primary", ("ari",), "joint_training"),
                                      ("secondary", METRICS[1:], "joint_training"),
                                      ("fixed_representation_secondary", ("ari",), "fixed_representation")):
        records = [dict(metric=metric, comparator=comparator,
            **contrast(differences("simulation", plan["simulation_primary_cells"], regime, metric, comparator)))
            for metric in metrics for comparator in BASELINES]
        holm(records)
        for r in records:
            ref = reference["simulation"][category][r["metric"] + "__vs__" + r["comparator"]]
            np.testing.assert_allclose([r["mean"], r["ci95_low"], r["ci95_high"], r["holm_p"]],
                                       [ref["mean"], *ref["ci95_unadjusted"], ref["p_holm"]], rtol=1e-9, atol=1e-12)
        output["simulation_"+category] = records
    cellwise = []
    cells = sorted({r["design"] for r in rows if r["study"] == "simulation"})
    for cell in cells:
        for comparator in BASELINES:
            cellwise.append(dict(design=cell, comparator=comparator,
                **contrast(differences("simulation", [cell], "joint_training", "ari", comparator))))
    holm(cellwise)
    for r in cellwise:
        ref = reference["simulation"]["cellwise_secondary"][r["design"]+"__vs__"+r["comparator"]]
        np.testing.assert_allclose([r["mean"], r["holm_p"]], [ref["mean"], ref["p_holm"]], rtol=1e-9, atol=1e-12)
    output["simulation_cellwise"] = cellwise
    methods = sorted({r["method"] for r in rows if r["study"] == "simulation" and r["regime"] == "joint_training"})
    output["simulation_average"] = [dict(method=method, metrics={
        m: describe([np.mean([indexed["simulation",c,"joint_training",method,s][m]
                    for c in plan["simulation_primary_cells"]]) for s in plan["simulation_seeds"]])
        for m in METRICS}) for method in methods]
    output["selected_families"] = [dict(study=k[0], design=k[1], regime=k[2],
        counts=dict(sorted(Counter(r["selected_family"] for r in records).items())))
        for k,records in sorted(groups.items()) if k[3] == "ghvade_adaptive"]

    diagnostics = json.loads((ROOT / "results/diagnostics.json").read_text())
    costs = []
    for (study, design, regime, method), records in sorted(groups.items()):
        if study != "image" or method not in (*BASELINES, "ghvade_adaptive"):
            continue
        ref = diagnostics["reference_costs"][design+"/"+regime][method]
        values = {k: describe([r["costs"][k] for r in records]) for k in records[0]["costs"]}
        for key in values:
            np.testing.assert_allclose([values[key]["mean"], values[key]["sd"]],
                                       [ref[key]["mean"], ref[key]["sd"]], rtol=1e-9, atol=1e-9)
        costs.append(dict(dataset=design, regime=regime, method=method, metrics=values))
    output["image_costs_seconds_and_mib"] = costs
    mechanism_groups = defaultdict(list)
    for record in diagnostics["image_mechanisms"]:
        mechanism_groups[record["dataset"],record["regime"]].append(record)
    mechanisms = []
    for (dataset, regime), records in sorted(mechanism_groups.items()):
        assignments = {k: describe([r["assignments"][k] for r in records]) for k in
                       ("decision_change_fraction", "boundary_decision_change_fraction", "interior_decision_change_fraction")}
        moments = {m: {stage: describe([r["moments"][m][stage]["mean_abs_aggregate_q_skewness"] for r in records])
                   for stage in ("pretraining_moments", "final_moments")} for m in (*BASELINES,"ghvade_adaptive")}
        ref = diagnostics["reference_mechanisms"][dataset+"/"+regime]
        for key, value in assignments.items():
            np.testing.assert_allclose([value["mean"],value["sd"]], [ref["assignments"][key]["mean"],ref["assignments"][key]["sd"]])
        for method, stages in moments.items():
            for stage, value in stages.items():
                np.testing.assert_allclose([value["mean"],value["sd"]], [ref["moments"][method][stage]["mean"],ref["moments"][method][stage]["sd"]])
        mechanisms.append(dict(dataset=dataset, regime=regime, assignments=assignments, moments=moments))
    output["image_mechanisms"] = mechanisms
    output["selection_margins"] = []
    for dataset in plan["data_hashes"]:
        values = [r["margin"] for r in diagnostics["selection_margins"] if r["dataset"]==dataset]
        output["selection_margins"].append(dict(dataset=dataset, **describe(values), minimum=min(values), maximum=max(values)))
    latent = {(r["design"],r["seed"],r["family"]):r for r in diagnostics["known_latent"]}
    families = {r["family"] for r in diagnostics["known_latent"]}
    expected = {(c,s,f) for c in cells for s in plan["simulation_seeds"] for f in families}
    if len(families) != 8 or len(latent) != len(diagnostics["known_latent"]) or set(latent) != expected:
        raise ValueError("Expected 200 complete known-latent blocks and eight candidate families")
    output["known_latent_average"] = []
    for family in ("gaussian", "student_t", "full_gh"):
        means = {m: [] for m in (*METRICS,"log_score")}
        for seed in plan["simulation_seeds"]:
            records = [latent[c,seed,family] for c in plan["simulation_primary_cells"]]
            for m in means:
                means[m].append(np.mean([r[m] if m=="log_score" else r["metrics"][m] for r in records]))
        output["known_latent_average"].append(dict(family=family, metrics={m:describe(v) for m,v in means.items()}))
    output["calibration"] = plan["calibration"]
    destination = ROOT / "generated/summary.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Verified 3,240 scores; reproduced primary, secondary, and cellwise contrasts plus supplementary diagnostics.")
    print(destination)


if __name__ == "__main__":
    main()
