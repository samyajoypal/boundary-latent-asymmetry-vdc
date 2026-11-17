# Current Simulation and Image Studies

This directory accompanies the November 2026 revision. The article and its
supplementary PDF contain the scientific results. This Git tree supplies code,
recorded configurations, compact run-level scores, and execution instructions;
no separate evidence ZIP or manuscript download is needed to use it.

The older root-level release concerns different exploratory experiments. Do not
use its Dry Bean or NIG-orientation commands to reproduce the current tables.
The current simulation has 20 seeds in each of ten designs; the image study has
20 seeds on STL-10, CIFAR-10, and CIFAR-100-20. All 3,240 method scores are retained.
The image inputs are supervised ImageNet-pretrained ResNet-50 features, not raw
pixels. Some prior seed outcomes were inspected and the dataset panel was chosen
through outcome-informed exploratory work. These numerical reruns are not
independent fresh confirmation; correction within the retained panel does not
correct for the preceding search over applications.

## Environment

Run the following from the repository root. The recorded experiments used
Python 3.9.25, PyTorch 2.8.0 with CUDA 12.8, an RTX A4000, and eight CPU workers
for adaptive candidate fits. A compatible CUDA driver is required for training.
The mixture optimization itself remains CPU-based.

```bash
python3.9 -m venv .venv-current
.venv-current/bin/python -m pip install -r current/requirements.txt
.venv-current/bin/python current/summarize.py
```

For analysis only, use `requirements-analysis.txt` instead: it does not install
PyTorch. Python 3.9--3.11 is compatible with these pinned analysis dependencies.

The last command requires no GPU or datasets: it verifies the scientific-source
hashes, checks all seed counts, computes means and sample standard deviations,
and reproduces paired confidence intervals and Holm corrections. It writes
`current/generated/summary.json` and checks the contrasts against the audited
reference. The compact scores retain each original result file's SHA-256; this
is a statistical recomputation, not a new audit of unshipped checkpoints.
Run `current/plot_results.py` afterward to regenerate the two main result plots
as PNGs. Additional compact diagnostic records reproduce the density,
assignment-change, selected-family, and timing summaries without private logs.
The known-latent records are diagnostics, not extra deep-clustering runs; the
portable training launcher below covers the 3,240 main method evaluations.

## Prepare Image Features

```bash
for dataset in stl10_resnet50 cifar10_resnet50 cifar10020_resnet50; do
  PYTHONPATH=current/code/src:current/code \
    .venv-current/bin/python current/code/scripts/prepare_paper2_four_benchmarks.py \
    "$dataset" --data-home current/data/features \
    --image-home current/data/images --device cuda
done
```

This downloads public datasets and the ResNet-50 `IMAGENET1K_V1` weights under
their respective terms. The exact transform is implemented in
`current/code/src/ghmvae/data/publication_benchmarks.py`. Training uses labeled
train and test image splits pooled without using target labels in fitting;
STL-10's additional unlabeled split is not used. Original cache hashes are in
`current/protocols/studies.json`. Serialization and hardware can change cache
bytes; compare preprocessing and numerical content before interpreting a hash
mismatch as a scientific discrepancy.

## Run Recorded Configurations

Commands below only print the full execution command by default. Add `--execute`
to perform the potentially expensive training. No Slurm account or private
filesystem paths are required. The wrapper changes only filesystem locations;
the scientific runner files are byte-matched to the recorded snapshots.

```bash
.venv-current/bin/python current/run.py --study simulation \
  --design nig_easy --seed 16400 --method ghvade_adaptive
.venv-current/bin/python current/run.py --study image \
  --design stl10_resnet50 --seed 16200 --method ghvade_adaptive
```

Use `--regime fixed_representation` for the common-pretraining-only fit; the
default is `joint_training`. The matched methods are `ghvade_adaptive`,
`ghvade_mvt`, and `ghvade_mvn`. `current/protocols/runs.json` lists every valid
design/seed/regime/method combination and all exact fitted settings. Simulation
seeds are 16400--16419; image seeds are 16200--16219. Additional baseline methods
are available in that same file, without changing their recorded schedules.

All matched methods for a dataset/design and seed use the same checkpoint path
under `current/outputs/pretraining/`. Run them serially within each block to
avoid concurrent writes to the first checkpoint; separate blocks can run on
separate GPUs. Use one GPU and eight CPU cores per adaptive run; candidate fits
can otherwise oversubscribe the host. Existing result directories are never overwritten. Set a new
`--output` root for an independent rerun, and use the same root for all methods
in each comparison. Newly generated shared checkpoints preserve the pairing
protocol but are not a claim of bitwise identity to archived cluster runs.
No checkpoints, pretrained image weights, or original images are redistributed.

For example, execute both regimes of the matched methods for one simulation
block (this performs real training and can take substantial time):

```bash
for regime in fixed_representation joint_training; do
  for method in ghvade_mvt ghvade_mvn ghvade_adaptive; do
    .venv-current/bin/python current/run.py --study simulation \
      --design nig_easy --seed 16400 --regime "$regime" --method "$method" --execute
  done
done
```

Enumerate the configurations in `protocols/runs.json` to repeat the complete
panel. The 60 TURTLE runs use the separate entry point below. Check each output's
`status` and retain failures for diagnosis; do not replace difficult seeds or
pool newly generated results with the archived numerical reruns.

## TURTLE Comparator

Obtain the authors' upstream source separately and respect its license:

```bash
git clone https://github.com/mlbio-epfl/turtle.git current/code/vendor/turtle_reference
git -C current/code/vendor/turtle_reference checkout 4674efab38e8f3cd0a4cd1f25d3ffc88be3dcabc
.venv-current/bin/python current/run.py --study image \
  --design stl10_resnet50 --seed 16200 --method turtle_1space
```

Add `--execute` after feature preparation to run the official single-space
optimizer with the recorded 6,000 outer iterations and ten inner steps. This is
the paper's matched-input comparison, not the published CLIP/DINO experiment.

## Interpretation and Release Boundaries

Adaptive improves ARI over matched Student-t on STL-10 and CIFAR-10, but the
CIFAR-100-20 contrast is inconclusive. External baselines can perform better,
and adaptive fitting is much more expensive. Keep all datasets, designs, seeds,
and baselines in any reproduction; do not select only favorable outputs.

`code/vendor/fmvmm_tstable2_20261102/` includes the corrected MIT-licensed
mixture backend with its original license. `source_manifest.json` records the
exact source hashes and study snapshot. The portable launcher and aggregation
entry points were added for this release; they were not the HPC scheduler
wrappers used in the original experiment. GPU training was not rerun merely to
prepare these instructions. Manuscripts, supplements, journal templates, private
scheduler records, and credentials are intentionally excluded from this tree.

## Validate a Code Release

```bash
.venv-analysis/bin/python -m unittest discover -s current/tests -v
.venv-analysis/bin/python current/check_release.py
```

The tests cover statistical recomputation, all recorded run identities, shared
checkpoint paths, and rejection of editorial files or detected credentials.
Maintainers can additionally audit the exact staged Git content with
`current/check_release.py --index` before committing. Scientific-source hashes
are checked separately from `RELEASE_SHA256.json`, which covers the entire
current code-and-records release. Changes require a deliberate manifest refresh;
do not refresh it merely to suppress a failed reproduction check.
