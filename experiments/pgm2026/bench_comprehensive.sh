#!/bin/bash
# Comprehensive RFFGM vs GPGM vs MAGI comparison (PGM 2026)
# Full experiment matrix with multiple seeds
#
# Matrix:
#   RFFGM:  5 kernels × 3 ODEs × 4 N × 2 L × 5 seeds = 600 jobs
#   GPGM:   2 kernels × 3 ODEs × (3 N × 5 seeds + 1 N=100 × 1 seed) = 96 jobs
#   MAGI:   2 kernels × 3 ODEs × (3 N × 5 seeds + 1 N=100 × 1 seed) = 96 jobs
#   Total:  792 jobs

cd /home/miyazawa/RFFGradientMatching.jl

MAX_PARALLEL=50

OUTDIR="experiments/pgm2026/results/comprehensive_v2"
mkdir -p "$OUTDIR"

SCRIPT="experiments/pgm2026/bench_comprehensive.jl"
OUTCSV="$OUTDIR/comprehensive_results_v2.csv"
HEADER="method,kernel,ode,N,n_rff,seed,rmsd,ess_mean,ess_per_sec,rhat_max,wall_sec,coverage,traj_rmse,theta_mean,theta_std"
echo "$HEADER" > "$OUTCSV"

SEEDS="42 123 456 789 1234"
ODES="LV FN PST"

# Collect all jobs as array of arguments
JOBS=()

# ── RFFGM jobs: 5 kernels × 3 ODEs × 4 N × 2 L × 5 seeds ──
for ode in $ODES; do
    for kernel in RBF Matern52 Laplace GenCauchy ExpPower; do
        for N in 10 25 50 100; do
            for L in 50 100; do
                for seed in $SEEDS; do
                    JOBS+=("RFFGM $kernel $ode $N $L $seed")
                done
            done
        done
    done
done

# ── GPGM jobs: 2 kernels × 3 ODEs × (N≤50: 5 seeds, N=100: 1 seed) ──
for ode in $ODES; do
    for kernel in RBF Matern52; do
        for N in 10 25 50; do
            for seed in $SEEDS; do
                JOBS+=("GPGM $kernel $ode $N 0 $seed")
            done
        done
        # N=100: single seed only (compute cost)
        JOBS+=("GPGM $kernel $ode 100 0 42")
    done
done

# ── MAGI jobs: same structure as GPGM ──
for ode in $ODES; do
    for kernel in RBF Matern52; do
        for N in 10 25 50; do
            for seed in $SEEDS; do
                JOBS+=("MAGI $kernel $ode $N 0 $seed")
            done
        done
        # N=100: single seed only (compute cost)
        JOBS+=("MAGI $kernel $ode 100 0 42")
    done
done

TOTAL=${#JOBS[@]}
echo "Total jobs: $TOTAL (max $MAX_PARALLEL parallel)"
echo "Output: $OUTCSV"
echo ""

# Run in batches
COMPLETED=0
FAILED=0
BATCH_START=0

while [ $BATCH_START -lt $TOTAL ]; do
    BATCH_END=$((BATCH_START + MAX_PARALLEL))
    if [ $BATCH_END -gt $TOTAL ]; then
        BATCH_END=$TOTAL
    fi

    BATCH_SIZE=$((BATCH_END - BATCH_START))
    BATCH_NUM=$((BATCH_START / MAX_PARALLEL + 1))
    echo "=== Batch $BATCH_NUM: jobs $((BATCH_START+1))-$BATCH_END of $TOTAL ==="

    PIDS=()
    TMPFILES=()
    DESCS=()

    for i in $(seq $BATCH_START $((BATCH_END - 1))); do
        args=(${JOBS[$i]})
        method=${args[0]}
        kernel=${args[1]}
        ode=${args[2]}
        N=${args[3]}
        L=${args[4]}
        seed=${args[5]}

        tmpf=$(mktemp "$OUTDIR/tmp_XXXX.csv")
        TMPFILES+=("$tmpf")
        desc="${method} ${kernel} ${ode} N=${N} L=${L} seed=${seed}"
        DESCS+=("$desc")
        echo "  Launching $desc"
        julia --project=. "$SCRIPT" "$method" "$kernel" "$ode" "$N" "$L" "$seed" \
            > "$tmpf" 2>/dev/null &
        PIDS+=($!)
    done

    # Wait for batch
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}"
        status=$?
        if [ $status -eq 0 ] && [ -s "${TMPFILES[$j]}" ]; then
            cat "${TMPFILES[$j]}" >> "$OUTCSV"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  FAILED: ${DESCS[$j]} (exit=$status)"
            FAILED=$((FAILED + 1))
        fi
        rm -f "${TMPFILES[$j]}"
    done

    echo "  Batch done. Progress: $((COMPLETED + FAILED))/$TOTAL (ok=$COMPLETED, fail=$FAILED)"
    echo ""
    BATCH_START=$BATCH_END
done

echo "=== Summary ==="
echo "Completed: $COMPLETED, Failed: $FAILED, Total: $TOTAL"
echo ""
echo "=== Results (first 30 lines) ==="
column -t -s',' "$OUTCSV" | head -30
echo ""
echo "Full results: $OUTCSV"
