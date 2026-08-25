# Frozen provenance

The independent replication recorded the following values in every one of its
300 completed task manifests:

| Object | SHA-256 or revision |
|---|---|
| Experiment source tree | `47105f4b54b8808a5022e271b9e757d6a37775b61344a9713acf1f899b93a460` |
| Replication TOML | `4f6ef3afc6e1dc3c4075d39101d15c707808584bbc5421b89c235e10889628cc` |
| FMVMM commit | `8b0570822a2d774221ca35e01b619d3b6b08ee54` |

The earlier 600-run shape-dominant study recorded:

| Object | SHA-256 or revision |
|---|---|
| Shape-study TOML | `fc8c721a1e173df58b04489c8fb7978ca13a89d6dee3d982397706415a7d4d90` |
| FMVMM commit | `48a141134a2750bfcb04836ae63d78c3cde08fa9` |

The older runner did not record a source-tree hash for the shape-dominant
study. This absence is retained rather than retrospectively replacing it with
the later replication hash. The exact resolved settings and all 600 aggregate
rows are released.

The 140-run Dry Bean real-data study recorded the following values
in every completed task manifest:

| Object | SHA-256 or revision |
|---|---|
| Experiment source tree | `fff93aed975b4ff74b1588b9250ea20e98d1317f6e370fb7334a6f2b5ec6ba69` |
| Confirmatory TOML | `761bba0c4c7937061f9003870417fe9fa20451a26e6f56fe90806234b02cf80a` |
| Data features, labels, and evaluation mask | `d534cb4e12c1f320638877c040e6c931b3459b947ee21e0219614fdaf4663f9b` |
| FMVMM commit | `8b0570822a2d774221ca35e01b619d3b6b08ee54` |

The source tree is preserved independently under
`experiment_snapshots/dry_bean/`. The failed blinded gate and the pre-label
amendment are retained; the 20-seed external analysis is therefore
descriptive, not relabelled as confirmatory.
