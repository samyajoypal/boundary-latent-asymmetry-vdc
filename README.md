# Boundary-Relevant Latent Asymmetry in Variational Deep Clustering

This repository is the reproducibility release for the manuscript
**Boundary-Relevant Latent Asymmetry in Variational Deep Clustering**.

The current manuscript snapshot is frozen at tag
`paper2-neurocomputing-r3-2026-08-24`. Its official Elsevier source, compiled
article and supplement, and flat upload archive are in `paper/`; the archive is
`paper/submission_ready_neurocomputing_r3.zip`.

The paper studies when a flexible latent component changes a clustering
partition rather than only improving density fit. It reports a 600-run adaptive
family study, a failed 20-seed NIG orientation confirmation, an independent
100-seed replication, and a locked 20-seed Dry Bean application with seven
methods. The replication supports a boundary-orientation effect and a
boundary-facing NIG advantage. The Dry Bean application is deliberately
retained as a negative fallback result: adaptive GH-VaDE does not outperform
matched Student-t GH-VaDE, and raw-space Gaussian mixtures perform best. The
paper does not claim universal GH superiority or formal outward equivalence.

## Contents

- `src/` and `scripts/`: exact experiment-source snapshot for the independent
  replication.
- `experiment_snapshots/dry_bean/`: exact source tree recorded by every Dry
  Bean task manifest.
- `configs/`: frozen TOML protocols for the 600-run study and both NIG
  orientation studies.
- `results/shape_dominant/`: one aggregate row for every broad-study run.
- `results/nig_orientation_confirmation/`: complete initial-confirmation
  diagnostics and gate record.
- `results/nig_orientation_replication/`: complete replication diagnostics,
  paired differences, validity rows, timing summary, and gate record.
- `results/paper2_dry_bean_confirmatory/`: compact per-run results and manifests
  for all 140 completed tasks, plus the complete frozen analysis.
- `results/paper2_dry_bean_blind_pilot/`: the failed blinded gate and the five
  adaptive pilot selector histories needed to reconstruct the chronology.
- `publication_scripts/`: post-experiment analysis, table/figure generation,
  release verification, and manuscript bundle builder.
- `paper/`: journal manuscript source, PDFs, and flat submission archive.
- `protocols/`: the prospectively written replication contract.

No class label enters mixture fitting or family selection. Labels are retained
only for external ACC, ARI, and NMI evaluation.

## Frozen revisions

- Replication experiment-source SHA-256:
  `47105f4b54b8808a5022e271b9e757d6a37775b61344a9713acf1f899b93a460`
- Replication protocol SHA-256:
  `4f6ef3afc6e1dc3c4075d39101d15c707808584bbc5421b89c235e10889628cc`
- Shape-study protocol SHA-256:
  `fc8c721a1e173df58b04489c8fb7978ca13a89d6dee3d982397706415a7d4d90`
- Replication FMVMM commit:
  `8b0570822a2d774221ca35e01b619d3b6b08ee54`
- Shape-study FMVMM commit:
  `48a141134a2750bfcb04836ae63d78c3cde08fa9`
- Dry Bean experiment-source SHA-256:
  `fff93aed975b4ff74b1588b9250ea20e98d1317f6e370fb7334a6f2b5ec6ba69`
- Dry Bean protocol SHA-256:
  `761bba0c4c7937061f9003870417fe9fa20451a26e6f56fe90806234b02cf80a`
- Dry Bean data SHA-256 (features, labels, and evaluation mask):
  `d534cb4e12c1f320638877c040e6c931b3459b947ee21e0219614fdaf4663f9b`
- Dry Bean FMVMM commit:
  `8b0570822a2d774221ca35e01b619d3b6b08ee54`

The frozen shape-study protocol records the original local FMVMM path because
its checksum is part of the provenance record. Set the `FMVMM_PATH` environment
variable when running it elsewhere; the protocol runner applies that portable
override without modifying the frozen file.

## Verify the release

Create a Python environment and install the recorded dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-reproduction.txt
.venv/bin/pip install -e .
```

FMVMM is a separate dependency. Clone
<https://github.com/samyajoypal/fmvmm>, check out the commit for the study being
rerun, and install it in the same environment.

Verify all row counts, frozen hashes, validity counts, and principal estimates:

```bash
.venv/bin/python publication_scripts/verify_release.py
```

Reconstruct the retained Dry Bean fallback analysis from the released 140
result/manifest pairs and failed blinded-pilot record:

```bash
MPLCONFIGDIR=/tmp/ghvade-matplotlib \
  .venv/bin/python publication_scripts/analyze_paper2_dry_bean_confirmatory.py \
  --analysis-mode descriptive_fallback
```

Regenerate every manuscript table, figure, and result macro from the released
CSV and JSON files:

```bash
MPLCONFIGDIR=/tmp/ghvade-matplotlib \
  .venv/bin/python publication_scripts/make_paper2_outputs.py
```

Build the flat Elsevier source package:

```bash
.venv/bin/python publication_scripts/build_paper2_submission.py \
  --output-dir paper/submission_rebuilt_local
```

## Rerun a frozen task

For example, inspect or run task 0 of the independent replication:

```bash
export FMVMM_PATH=/absolute/path/to/fmvmm
.venv/bin/python scripts/simulation_protocol.py count \
  --protocol configs/paper2_nig_orientation_replication.toml \
  --stage final --tier core
.venv/bin/python scripts/simulation_protocol.py run \
  --protocol configs/paper2_nig_orientation_replication.toml \
  --stage final --tier core --task-id 0
```

The full arrays are computational experiments; the bundled aggregates are
sufficient to reconstruct all reported statistical analyses and manuscript
artifacts without rerunning neural training.

## License

Code is released under the MIT License. Manuscript text and published figures
remain subject to the journal's publication terms.
