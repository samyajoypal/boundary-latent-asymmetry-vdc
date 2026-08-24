# Dry Bean Protocol Amendment: Descriptive Fallback Study

Amendment lock time: 2026-08-21T09:51:14+02:00.

This amendment was written after the label-blinded pilot gate was evaluated and
before any Dry Bean external clustering metric was inspected. The frozen pilot
gate file has SHA-256
`02d32d11704803468c533cc036475d3661f68e7d7988d05c31951becabd0d119`.

## Pilot Decision

The original gate failed and remains failed. All five tasks completed, emitted
no external metric, retained seven active and assigned components, and passed
the regularity checks. Nevertheless, no final asymmetric promotion occurred.
The best asymmetric family was skew-normal in every final cycle, but its
held-out gain over Student-t ranged only from 0.00067 to 0.00795 nats per
observation, below the frozen 0.01 threshold.

The result is not explained by an absent boundary effect. Student-t versus
skew-normal assignment changes ranged from 1.54% to 8.67%, and their
concentration in the lowest 20% of Student-t margins ranged from 3.66 to 4.99.
Rather, the asymmetric density advantage weakened during alternating neural
training. Four of five initial prior fits promoted skew-normal, with gains from
0.0123 to 0.0475, but all five final cycles selected Gaussian under the frozen
shape-gated, one-standard-error rule.

No threshold, family catalogue, architecture, seed, dataset, validity rule, or
method implementation is changed in response. Pilot labels remain blinded and
will not be retrospectively evaluated.

## Amended Research Question

The conditional confirmatory superiority study is not conducted. In its place,
the already-frozen unseen-seed configuration is run as a descriptive real-data
fallback study. Its question is whether the deployed adaptive procedure remains
competitive when it rejects asymmetric enrichment and reduces to a simpler
family, while exposing the computational price of testing the larger catalogue.

This is an explicitly secondary application. It cannot support a claim that an
asymmetric GH prior improves Dry Bean clustering. It may support the narrower
claim that the selector deploys extra flexibility only when its label-free
evidence threshold is met. A negative or inferior result will be included.

## Frozen Descriptive Analysis

The unmodified configuration
`configs/paper2_dry_bean_confirmatory.toml` is used. Its SHA-256, fixed before
the pilot ran, is
`761bba0c4c7937061f9003870417fe9fa20451a26e6f56fe90806234b02cf80a`.
It contains unseen seeds 4200--4219 and seven methods: raw K-means, raw Gaussian
mixture, DEC, IDEC, Gaussian VaDE, fixed Student-t GH-VaDE, and adaptive
GH-VaDE. The adaptive family catalogue and all neural and fitting settings are
unchanged.

No result is inspected until all 140 scheduled method-seed tasks have reached a
terminal state. ACC, ARI, and NMI are then reported as mean plus or minus
standard deviation for every method. Paired adaptive-minus-Student-t and
adaptive-minus-Gaussian-VaDE differences, two-sided 95% confidence intervals,
and Holm-adjusted exploratory p-values are reported without converting this
post-gate amendment into a confirmatory test. Failed or invalid runs are shown,
never replaced, and the complete pilot gate failure is reported alongside the
real-data table.

## Manuscript Rule

The application must be described as a prospective descriptive amendment, not
as confirmation of the original gate. The manuscript must state that Dry Bean
selected Gaussian in all five blinded pilot seeds. Initial-stage asymmetric
promotions, final score margins, boundary localization, numerical failures, and
runtime must all remain visible. The real-data section is retained regardless
of whether adaptive GH-VaDE wins, ties, or loses on external metrics.

