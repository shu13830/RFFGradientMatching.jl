#!/bin/bash
# Parallel kernel comparison: RFFGM (5 kernels) + GPGM (2 kernels)
# LV, N=25, n_rff=50, HMC 10k iter

cd /home/miyazawa/RFFGradientMatching.jl

N_OBS=25
N_RFF=50
OUTDIR="experiments/pgm2026/results/kernel_comparison"
mkdir -p "$OUTDIR"

SCRIPT="experiments/pgm2026/bench_kernel_comparison.jl"
OUTCSV="$OUTDIR/lv_N${N_OBS}_L${N_RFF}.csv"
HEADER="method,kernel,N,n_rff,rmsd,ess_mean,ess_per_sec,wall_sec,theta1,theta2,theta3,theta4"
echo "$HEADER" > "$OUTCSV"

# RFFGM: all 5 kernels
RFFGM_KERNELS="RBF Matern52 Laplace GenCauchy ExpPower"
# GPGM: only RBF and Matern52 (others lack analytical K'/K'')
GPGM_KERNELS="RBF Matern52"

PIDS=()
TMPFILES=()

for kern in $RFFGM_KERNELS; do
    tmpf=$(mktemp "$OUTDIR/tmp_rffgm_${kern}_XXXX.csv")
    TMPFILES+=("$tmpf")
    echo "Launching RFFGM + $kern ..."
    julia --project=. "$SCRIPT" RFFGM "$kern" "$N_OBS" "$N_RFF" \
        > "$tmpf" 2>/dev/null &
    PIDS+=($!)
done

for kern in $GPGM_KERNELS; do
    tmpf=$(mktemp "$OUTDIR/tmp_gpgm_${kern}_XXXX.csv")
    TMPFILES+=("$tmpf")
    echo "Launching GPGM + $kern ..."
    julia --project=. "$SCRIPT" GPGM "$kern" "$N_OBS" "$N_RFF" \
        > "$tmpf" 2>/dev/null &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} jobs. Waiting..."

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
