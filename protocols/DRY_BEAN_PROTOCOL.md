# Prospective Dry Bean Real-Data Protocol

Status: locked before external clustering metrics are inspected.

Pilot outcome: the original promotion gate failed on 2026-08-21. The original
protocol below is preserved verbatim. Before any external metric was inspected,
the secondary descriptive fallback study was locked in
`DRY_BEAN_PROTOCOL_AMENDMENT.md`; that amendment does not redefine this gate as
successful.

Lock date: 2026-08-20.

Frozen pilot-config SHA-256:
`80ff91465733cf30ede7a540207f7017f61dfbb2221af5b30c7fe1220c879310`.

Frozen confirmatory-config SHA-256:
`761bba0c4c7937061f9003870417fe9fa20451a26e6f56fe90806234b02cf80a`.

The mixture backend is local `fmvmm` revision
`8b0570822a2d774221ca35e01b619d3b6b08ee54`.

## Purpose

This experiment tests the mechanism claimed by Paper 2 on one real dataset. It
does not ask whether generalized hyperbolic (GH) mixtures dominate every deep
clustering method. It asks whether the label-free selector detects asymmetric
latent structure, whether the resulting assignment changes concentrate near a
matched Student-t decision boundary, and, only after that unlabeled gate is
passed, whether those changes improve external clustering metrics.

## Dataset

The sole dataset is UCI Dry Bean (dataset 602; DOI 10.24432/C50S4B). It contains
13,611 bean grains, 16 image-derived morphological measurements, and seven
botanical varieties. The complete dataset is used, so no label-dependent
subsampling occurs. Features are standardized without labels. The number of
components is fixed at the publicly documented value K = 7. The archive is
identified by SHA-256
`0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f`.

Dry Bean was selected before any Paper 2 class metric was computed. Its sample
size supports held-out GH fitting, while its positive morphology measurements
provide a plausible, non-engineered source of within-class skewness. It does
not overlap any real dataset used by Paper 1.

## Frozen Model

The primary model is the unchanged homogeneous adaptive selector used in the
shape-dominant study. The candidate catalogue is Gaussian, Student-t,
skew-normal, variance-gamma, hyperbolic, normal-inverse-Gaussian, skew-t, and
full GH. Selection uses only an independent 20% latent holdout, the frozen
shape gate (mean absolute marginal skewness at least 0.25), and a held-out score
margin of at least 0.01 nats per observation. No label, class metric, or class
name enters representation learning, mixture fitting, model selection,
stopping, numerical screening, or the pilot gate.

The multilayer perceptron has hidden widths 128 and 64 and latent dimension 4.
Deterministic pretraining lasts 80 epochs. Three alternating cycles use at most
40 neural epochs each, five posterior draws for mixture refitting, three draws
per training observation, and 64 draws for final assignment. All other values
are frozen in `configs/paper2_dry_bean_blind_pilot.toml`.

## Blinded Pilot

The pilot consists of seeds 4100--4104 and only the adaptive model. The runner
scrambles the association between observations and evaluation labels before
calling any metric-producing helper, stores only scrambled labels in artifacts,
excludes labels from the dataset hash, and recursively removes clustering
accuracy (ACC), adjusted Rand index (ARI), normalized mutual information (NMI),
and purity from JSON, CSV, console, and summary outputs.

A run is numerically valid when all of the following hold:

1. The task completes with status `ok`, and blinding is active.
2. The final prior satisfies the frozen weight, covariance, condition-number,
   and interior-shape regularity checks.
3. All seven components are active and all seven receive at least one final
   assignment.
4. Student-t and the best asymmetric candidate have finite held-out scores and
   the absolute score difference between Student-t and every successful
   candidate is at most 5 nats per observation.
5. The Student-t versus best-asymmetric boundary diagnostic is finite.

The unlabeled pilot gate passes only if all five tasks are present, no external
metric key is emitted, at least four runs are numerically valid, at least three
valid runs promote an asymmetric family through the frozen shape gate, the
median best-asymmetric-minus-Student-t held-out score is at least 0.01 nats per
observation among promoted runs, and at least three promoted runs both change
at least 0.5% of assignments and concentrate those changes in the lowest 20% of
Student-t margins by a factor greater than 1.5 relative to a uniform change
rate. These criteria are implemented in
`scripts/analyze_paper2_dry_bean_pilot.py`.

If the gate fails, external pilot metrics will not be used to alter thresholds,
seeds, architecture, or dataset. The paper remains a simulation study and the
null application is recorded in the research log.

## Confirmatory Study

If and only if the unlabeled gate passes, the confirmatory study uses unseen
seeds 4200--4219. Labels are revealed once after all tasks finish and after run
validity is fixed using label-free rules. Failed or invalid seeds are retained
and are not replaced.

The table contains raw K-means, raw Gaussian mixture model, deep embedded
clustering (DEC), improved DEC (IDEC), Gaussian VaDE, fixed Student-t GH-VaDE,
and the unchanged adaptive model. ACC, ARI, and NMI are reported as mean plus or
minus standard deviation. The prespecified primary contrast is paired
adaptive-minus-Student-t ARI with a two-sided 95% confidence interval. Paired
ACC and NMI are secondary, with Holm correction across the three external
metrics. The paper will report the result whether positive, null, or negative.

## Manuscript Rule

No Dry Bean number enters the manuscript before the pilot gate is decided. If
confirmed, Paper 2 receives one application subsection, one complete method
table, one family-selection table, and one plot of assignment-change frequency
against Student-t margin. Claims remain conditional: asymmetric GH flexibility
is useful when supported and boundary-relevant, while the selector reduces to a
symmetric family otherwise.
