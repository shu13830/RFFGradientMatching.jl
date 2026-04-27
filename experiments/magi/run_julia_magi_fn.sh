#!/bin/bash
# Julia MAGI FN reproduction: 10 seeds in parallel
set -euo pipefail
cd /home/miyazawa/RFFGradientMatching.jl

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRIPT="experiments/magi/run_julia_magi_fn.jl"
OUTDIR="experiments/magi/results"
mkdir -p "$OUTDIR"

SEEDS="42 123 456 789 1234 2345 3456 4567 5678 6789"

echo "Julia MAGI FN reproduction (10 seeds, 10 parallel)"
echo "Started: $(date)"
echo ""

PIDS=()
for seed in $SEEDS; do
    echo "  Launching seed=$seed"
    julia --project=. "$SCRIPT" "$seed" \
        > "$OUTDIR/julia_magi_fn_seed${seed}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for all 10 jobs..."

FAILED=0
for i in "${!PIDS[@]}"; do
    seed=$(echo $SEEDS | cut -d' ' -f$((i+1)))
    wait "${PIDS[$i]}"
    status=$?
    if [ $status -eq 0 ]; then
        echo "  seed=$seed: OK"
    else
        echo "  seed=$seed: FAILED (exit=$status)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Results ==="
for seed in $SEEDS; do
    log="$OUTDIR/julia_magi_fn_seed${seed}.log"
    if [ -f "$log" ]; then
        grep "RMSD" "$log" 2>/dev/null || echo "  seed=$seed: no RMSD found"
    fi
done

echo ""
echo "Finished: $(date)"
echo "Failed: $FAILED / 10"
