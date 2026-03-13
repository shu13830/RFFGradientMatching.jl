#!/bin/bash
# Parallel ELMC vs HMC from converged HMC state (N=25)
# Phase 1: HMC burn-in+sample (10k iter, with ε adaptation) → converged state
# Phase 2: From converged state, burn-in with ε adaptation (5k) + sample (5k)
# Grid: α={0.5,1.0,2.0} × ε_init={0.002,0.005,0.008,0.01} × n_lf={5,10,15}

cd /home/miyazawa/RFFGradientMatching.jl

N_OBS=25
OUTDIR="experiments/pgm2026/results/elmc_grid"
mkdir -p "$OUTDIR"

SCRIPT="experiments/pgm2026/bench_elmc_from_hmc.jl"
STATEFILE="$OUTDIR/hmc_converged_N${N_OBS}_mat52_seed42.jls"

# Phase 1: HMC convergence (only if state file doesn't exist)
if [ ! -f "$STATEFILE" ]; then
    echo "=== Phase 1: Running HMC to convergence (N=$N_OBS) ==="
    julia --project=. "$SCRIPT" HMC 0.0 0.0 50 "$N_OBS" 2>/dev/null
    echo "Phase 1 complete."
fi

# Phase 2: Parallel grid search
OUTCSV="$OUTDIR/from_hmc_results_N${N_OBS}_mat52_adaptive.csv"
HEADER="method,alpha,epsilon_init,n_leapfrog,N,accept_rate,rmsd,theta1,theta2,theta3,theta4,wall_sec,logdens_mean,adapted_eps"
echo "$HEADER" > "$OUTCSV"

ALPHAS="0.5 1.0 2.0"
EPSILONS="0.002 0.005 0.008 0.01"
N_LFS="5 10 15"

PIDS=()
TMPFILES=()

# HMC baseline (one job, n_leapfrog=50)
tmpf=$(mktemp "$OUTDIR/tmp_hmc_XXXX.csv")
TMPFILES+=("$tmpf")
echo "Launching HMC baseline (N=$N_OBS, n_lf=50, ε_init=0.01)..."
julia --project=. "$SCRIPT" HMC 0.0 0.01 50 "$N_OBS" > "$tmpf" 2>/dev/null &
PIDS+=($!)

# ELMC grid: α × ε × n_leapfrog
for alpha in $ALPHAS; do
    for eps in $EPSILONS; do
        for nlf in $N_LFS; do
            tmpf=$(mktemp "$OUTDIR/tmp_elmc_${alpha}_${eps}_${nlf}_XXXX.csv")
            TMPFILES+=("$tmpf")
            echo "Launching ELMC α=$alpha, ε_init=$eps, n_lf=$nlf ..."
            julia --project=. "$SCRIPT" ELMC "$alpha" "$eps" "$nlf" "$N_OBS" \
                > "$tmpf" 2>/dev/null &
            PIDS+=($!)
        done
    done
done

echo "Launched ${#PIDS[@]} jobs (1 HMC + $((${#PIDS[@]}-1)) ELMC). Waiting..."

# Wait and collect
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    status=$?
    if [ $status -eq 0 ] && [ -s "${TMPFILES[$i]}" ]; then
        cat "${TMPFILES[$i]}" >> "$OUTCSV"
    else
        echo "FAILED: ${TMPFILES[$i]} (exit=$status)"
        FAILED=$((FAILED + 1))
    fi
    rm -f "${TMPFILES[$i]}"
done

echo ""
echo "=== Results ==="
column -t -s',' "$OUTCSV"
echo ""
echo "Done. $FAILED failures out of ${#PIDS[@]} jobs."
