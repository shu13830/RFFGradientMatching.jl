#!/bin/bash
# N-scaling: RFFGM vs GPGM at N={10,25,50,100}, Matern52 kernel
# RFFGM uses L=50 RFF features throughout

cd /home/miyazawa/RFFGradientMatching.jl

N_RFF=50
KERNEL="Matern52"
OUTDIR="experiments/pgm2026/results/n_scaling"
mkdir -p "$OUTDIR"

SCRIPT="experiments/pgm2026/bench_n_scaling.jl"
OUTCSV="$OUTDIR/lv_${KERNEL}_L${N_RFF}.csv"
HEADER="method,kernel,N,n_rff,rmsd,ess_mean,ess_per_sec,wall_sec,theta1,theta2,theta3,theta4"
echo "$HEADER" > "$OUTCSV"

N_VALUES="10 25 50 100"
METHODS="RFFGM GPGM"

PIDS=()
TMPFILES=()

for method in $METHODS; do
    for n in $N_VALUES; do
        tmpf=$(mktemp "$OUTDIR/tmp_${method}_N${n}_XXXX.csv")
        TMPFILES+=("$tmpf")
        echo "Launching $method N=$n ..."
        julia --project=. "$SCRIPT" "$method" "$n" "$N_RFF" "$KERNEL" \
            > "$tmpf" 2>/dev/null &
        PIDS+=($!)
    done
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
