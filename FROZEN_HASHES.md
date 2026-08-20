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
