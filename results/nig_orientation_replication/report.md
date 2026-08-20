# Paper 2 independent NIG-orientation replication

Overall locked gate: **FAIL**.

- Valid paired seeds: 98/100.
- Invalid family-stage fits: 46/4500 (1.02%).
- Student-t control best enrichment: +0.00454 (n=98).
- Boundary NIG score gain: +0.03661, 95% CI [+0.01669, +0.05653].
- Boundary NIG ARI gain: +0.03375, 95% CI [+0.01622, +0.05128].
- Boundary-minus-outward ARI: +0.02896, 95% CI [+0.00949, +0.04844].
- Outward NIG ARI: +0.00243, 90% CI [-0.00721, +0.01207].

## Holm-adjusted confirmatory tests

- orientation_ari: mean=+0.02896, one-sided p=0.00198351, Holm p=0.00198351
- boundary_ari: mean=+0.03375, one-sided p=0.000116819, Holm p=0.000350456
- boundary_density: mean=+0.03661, one-sided p=0.000213107, Holm p=0.000426215

## Locked checks

- provenance: pass
- complete_primary_pairs: pass
- numerical_stability: pass
- student_t_control: pass
- orientation_ari_ci: pass
- confirmatory_holm_family: pass
- outward_ari_equivalence: fail
