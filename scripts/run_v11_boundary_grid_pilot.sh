#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"
export FMVMM_PATH="${FMVMM_PATH:-/home/samyajoypal/fmvmm}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fmvmm-mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for orientation in clockwise_neighbor radial; do
    for strength in 0.6 1.2 2.4; do
        .venv/bin/python scripts/run_simulation_benchmark.py \
            --scenarios parametric \
            --design-id "v11_${orientation}_skew_${strength}_pilot" \
            --methods ghvade_mvt ghvade_component_adaptive \
            --seeds 0 1 2 \
            --n-samples 900 \
            --input-dim 12 \
            --n-components 3 \
            --latent-dim 2 \
            --matched-center-scale 1.4 \
            --matched-cluster-scale 0.6 \
            --skew-strength "$strength" \
            --tail-df 5 \
            --dgp-family student_t \
            --dgp-component-families student_t student_t skew_t \
            --dgp-skew-orientation "$orientation" \
            --mixture-pretrain-epochs 20 \
            --mixture-epochs 10 \
            --mixture-selection-fraction 0.2 \
            --component-enrichment-families mgst \
            --shape-score-margin 0.0 \
            --output-dir results/method_v11_pilot/boundary_grid
    done
done
