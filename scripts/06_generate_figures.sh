#!/bin/bash
# Regenerate the 5 figures in paper/figures/:
#  - mcmc_trajectory.png: 2-panel GPGM vs RFFGM posterior in (θa, θb) plane (LV, N=25, seed 42)
#  - trajectory.png: LV trajectory recovery at N=25 (RFFGM / GPGM / R MAGI)
#  - scaling.png: LVC scaling (RMSD and Rhat vs K)
#  - timing.png: wall time vs N for LV
#  - ablation.png: L ablation on LV
set -euo pipefail
cd "$(dirname "$0")/.."

export JULIA_NUM_THREADS=1

# Delegates to the project's figure-rendering script; outputs to paper/figures/.
julia --project=. scripts/plot_figures.jl \
    --output_dir paper/figures/

echo "figures written to paper/figures/"
ls -lh paper/figures/
