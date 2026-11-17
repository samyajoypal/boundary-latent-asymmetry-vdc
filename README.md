# Asymmetric Mixture Priors in Variational Deep Clustering: A Theoretical and Empirical Study

This repository contains the code, recorded configurations, and numerical
results for the current Paper 2 study. All experiment materials are under
`current/`; the root contains only this guide, citation metadata, licensing,
and Git exclusions.

## Reproduction Guide

**Code release: 17 November 2026, version `2026.11.17.1`.**

The [execution guide](current/README.md) covers installation, image feature
preparation, exact-configuration execution, and statistical reproduction for
the 20-seed simulation and the STL-10, CIFAR-10, and CIFAR-100-20 experiments.
The article and supplementary PDF contain the scientific results; neither
manuscript is distributed here, and no separate evidence ZIP is required.

### Reproduce the Results Without Training

```bash
git clone https://github.com/samyajoypal/boundary-latent-asymmetry-vdc.git
cd boundary-latent-asymmetry-vdc
python3.11 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r current/requirements-analysis.txt
.venv-analysis/bin/python current/check_release.py
.venv-analysis/bin/python current/summarize.py
.venv-analysis/bin/python current/plot_results.py
```

This checks source and result integrity, recomputes the statistics from all
3,240 method scores, and writes JSON summaries and the two main result plots to
the ignored `current/generated/` directory. It needs no GPU or original dataset
downloads. Python 3.9--3.11 is supported by the pinned analysis dependencies.

For GPU training, feature downloads, all recorded configurations, and the
external TURTLE comparator, follow [the execution guide](current/README.md).
Training is deliberately opt-in; printing a command does not launch it.

## Repository Layout

| Location | Purpose |
|---|---|
| `current/code/` | Hash-verified scientific code and licensed mixture backend |
| `current/protocols/` | Exact per-run settings, seed lists, and input hashes |
| `current/results/` | Complete current run-level results and diagnostic records |
| `current/run.py` | Portable single-run execution; `--execute` starts training |
| `current/summarize.py` | Statistical recomputation and reference checks |
| `current/plot_results.py` | Regeneration of the main result plots |
| `current/check_release.py` | Code-only publication and integrity checks |
| `current/tests/` | Tests for release safety and statistical reproduction |

These are matched-prior comparisons, not a state-of-the-art claim. Adaptive
improves ARI over matched Student-t on STL-10 and CIFAR-10; CIFAR-100-20 is
inconclusive, and several contextual baselines remain stronger. All methods,
datasets, and seeds in the current study are retained.

## License and Publication Boundary

Code is MIT-licensed; bundled third-party code retains its own copyright and
license. Datasets, pretrained weights, and the upstream TURTLE implementation
are obtained separately under their respective terms. Manuscripts, supplementary
documents, cover letters, journal templates, and submission archives are not
part of this code release. The release checker enforces this boundary and the
current-only repository layout.
