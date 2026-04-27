# Reproduction scripts for the final paper

This directory reproduces the experiments reported in the paper
"Random Fourier Features-Based Gradient Matching for Efficient ODE Parameter Inference"
(Miyazawa & Mochihashi, NeurIPS 2026 submission).

This directory contains only the final settings used for the reported numbers, tables, and figures.

## Prerequisites

- Julia 1.10 or newer with the project environment at the repository root
  (run `julia --project=. -e 'using Pkg; Pkg.instantiate()'` from the repo root once)
- Python 3.9+ with `scipy` (used by `07_generate_tables.py` for Welch t-tests)
- R 4.0+ with the `magi` package v1.2.5 (only needed for `04_r_magi_baseline.sh`)
- Approximately 30 CPU cores available for the parallel runners

Set environment variables to pin single-threaded BLAS per job:

```bash
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Scripts

| Script | Produces | Approx. wall time (30 cores) |
|---|---|---|
| `01_main_comparison.sh` | `results/main/{rffgm,gpgm,magi}_N{10,25,50}.csv` (Tables 1, 2) | ~24 h |
| `02_kernel_flexibility.sh` | `results/kernel/rffgm_kernels.csv` (Table 5, §4.4) | ~8 h |
| `03_scaling_lvc.sh` | `results/scaling/lvc_K{2,5,10,15}.csv` (Table 6, Fig 3) | ~18 h |
| `04_r_magi_baseline.sh` | `results/rmagi/rmagi_{lv,sir,pst,fn}.csv` (Table 1 R MAGI row, App J) | ~6 h |
| `05_auto_gamma.sh` | `results/auto_gamma/results.csv` (App K Table 9) | ~13 h |
| `06_generate_figures.sh` | `../paper/figures/{mcmc_trajectory,scaling,trajectory,timing,ablation}.png` | minutes |
| `07_generate_tables.py` | stdout: LaTeX table bodies (Tables 1, 2, 5, 6, 9) | seconds |

Run order:
1. Execute `01`--`05` to produce the raw result CSVs (can be run in parallel or interleaved;
   each script writes to its own `results/<subdir>/` and uses its own scheduler).
2. Execute `06_generate_figures.sh` (reads from CSVs produced by `01`, `03`, `05`).
3. Execute `07_generate_tables.py` to print the final LaTeX table bodies; paste or diff
   against the current `paper/main.tex`.

## Data generation

Observation data is generated deterministically by the Julia ODE solvers inside each
experiment script (given the seed), so there is no separate "download-data" step.
The exact per-ODE settings (noise level, time span, true parameters, 3 $\theta$ patterns)
are in [`common.jl`](common.jl) — that file is
loaded by every script here.

## Expected resource use

`01_main_comparison.sh` is the longest wall-clock step: 3 methods × 2 kernels × 4 ODEs ×
3 N × 3 $\theta$-patterns × 10 seeds × 5 γ = 10 800 Markov chains of 20 000 iterations each.
A 30-core machine takes on the order of a day; the script is idempotent and supports resume.

## Reproducibility notes

- All random seeds are fixed in the scripts (`SEEDS="42 123 456 789 1234 2345 3456 4567 5678 6789"`).
- The RFF frequency draw is seeded by the data seed, so the Table 5 kernel comparison
  uses one ω-draw per seed rather than a fresh draw per γ.
- R MAGI runs use `magi` v1.2.5 with default step-size factor 0.01 and 200 leapfrog steps
  (matching [`baselines/magi/setup.R`](../baselines/magi/setup.R)).
- Julia MAGI uses midpoint discretization (`discretization_level=1`) and the R MAGI-style
  per-dimension step-size adaptation documented in the paper's §3.
