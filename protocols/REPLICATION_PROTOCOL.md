# Independent NIG-orientation replication

Frozen on 2026-08-19 before generating any data from seeds 8000--8099.

## Rationale

The first 20-seed family-matched confirmation failed its full locked gate. Its
estimated boundary-minus-outward ARI effect was +0.05267 (standard deviation
0.13709), while the 95% confidence interval included zero. Outward NIG was ARI
equivalent to Student-t, but did not have the positive global density gain
required by the original gate. Those results are retained and will be reported
as a failed confirmatory study.

The independent replication tests the narrower mechanism supported by the
theory: the clustering consequence of the same NIG asymmetry changes with its
orientation relative to a competing decision boundary. It does not claim that
outward NIG must improve global mixture likelihood.

## Design and power

The generator, NIG parameters, Student-t anchor, architecture, optimization,
data splits, and diagnostics are unchanged. Only the unrestricted regime is
replicated. The three cells are a symmetric Student-t control, boundary-facing
NIG, and radial-outward NIG. There are 100 untouched seeds and 300 tasks.

The replication is powered for a standardized paired effect of 0.35, smaller
than the 0.384 observed in the preceding study. With 100 pairs and a one-sided
0.025 test (the same critical value as a two-sided 0.05 test), noncentral-t
power is 0.934.

## Locked analysis

The primary endpoint is the paired difference

`(NIG - Student-t ARI)_boundary - (NIG - Student-t ARI)_outward`

at oracle latent variables. Success requires its 95% confidence interval to be
strictly above zero. Boundary NIG ARI and boundary NIG held-out log score are
confirmatory secondary endpoints. Their one-sided p-values and that of the
primary endpoint are Holm-adjusted as one family; every adjusted p-value must
be below 0.05 and every estimated effect must be positive.

Calibration requirements are:

1. all 300 task files and 4,500 family-stage rows are present;
2. no more than 5% of family-stage fits violate the fixed validity rule;
3. at least 95 paired boundary/outward seeds are valid;
4. at least 90 Student-t control seeds have all three enrichment fits valid;
5. mean best control enrichment is no larger than 0.01 nats per observation;
6. outward NIG ARI is equivalent to Student-t under a TOST margin of 0.01,
   implemented by a 90% confidence interval strictly inside the margin.

Boundary-weighted scores, decision changes, and representation-stage drift are
secondary mechanism analyses. The complete 20-seed failure and the complete
100-seed replication will be reported regardless of the replication outcome.
