#!/bin/bash
# R MAGI baseline for Tables 1, 2 (R MAGI row) and Appendix J (MAGI Validation).
# Uses baselines/magi/run_all_rmagi.R which supports 3 θ-patterns and 5 fixed seeds.
#
# CLI of run_all_rmagi.R: <ode> <N> <theta_id>  (theta_id=0 → all patterns)
# Output is written to baselines/magi/results/ by the R script; we copy it out.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pin BLAS / OMP threads to 1 so that each R process uses exactly one core.
# Without this, R's default OpenBLAS silently spawns one thread per core
# inside each R process, leading to N_LANES x cores oversubscription.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTDIR="scripts/results/rmagi"
mkdir -p "$OUTDIR"

# 4 ODE × 3 N × 3 θ_id × 5 fixed seeds = 180 chains (R MAGI ~5-15 min each → ~5h with 30 lanes).
N_VALUES="25 50"  # Table 1 uses N=25, Table 2 uses N=50; skip N=10 for time budget
ODES="lv sir pst fn"

echo "=== R MAGI baseline: ODEs=$ODES, N=$N_VALUES, θ_id=1,2,3, 5 seeds each ==="
echo "Started: $(date)"

# Kick off 30-lane pool. Each job = (ode, N, theta_id) — 5 seeds loop inside R.
JOBS=()
for ode in $ODES; do
    for N in $N_VALUES; do
        for tid in 1 2 3; do
            JOBS+=("$ode $N $tid")
        done
    done
done
TOTAL=${#JOBS[@]}
echo "Total job groups: $TOTAL"

N_LANES=30
LOCK="$OUTDIR/.lock"
COUNTFILE="$OUTDIR/.count"
echo "0" > "$COUNTFILE"

run_job() {
    local lane_id=$1
    local spec=$2
    local args=($spec)
    local tag="${args[0]}_N${args[1]}_t${args[2]}"
    local log="$OUTDIR/rmagi_${tag}.log"
    local done_marker="$OUTDIR/rmagi_${tag}.done"
    if [ -f "$done_marker" ]; then
        (flock 200; cnt=$(cat "$COUNTFILE"); echo $((cnt+1)) > "$COUNTFILE"; echo "  [L$lane_id] SKIP (resume) ${tag} ($((cnt+1))/$TOTAL)") 200>"$LOCK"
        return 0
    fi
    if Rscript baselines/magi/run_all_rmagi.R "${args[@]}" > "$log" 2>&1; then
        touch "$done_marker"
        (flock 200; cnt=$(cat "$COUNTFILE"); echo $((cnt+1)) > "$COUNTFILE"; echo "  [L$lane_id] OK ${tag} ($((cnt+1))/$TOTAL)") 200>"$LOCK"
    else
        (flock 200; echo "  [L$lane_id] FAIL ${tag}") 200>"$LOCK"
    fi
}
export -f run_job
export OUTDIR LOCK COUNTFILE TOTAL

# Lane assignment
for l in $(seq 0 $((N_LANES - 1))); do eval "declare -a LANE_${l}=()"; done
for i in $(seq 0 $((TOTAL - 1))); do
    lane=$((i % N_LANES))
    eval "LANE_${lane}+=(\"${JOBS[$i]}\")"
done
# Launch
for l in $(seq 0 $((N_LANES - 1))); do
    (
        eval "lane_jobs=(\"\${LANE_${l}[@]}\")"
        for spec in "${lane_jobs[@]}"; do
            run_job $l "$spec"
        done
    ) &
done
wait

# Consolidate — the R script writes baselines/magi/results/rmagi_<ode>_N<N>_seed<seed>_samples.csv
# plus a summary row into an all_results CSV. Copy any CSVs produced.
if [ -d baselines/magi/results ]; then
    cp baselines/magi/results/*.csv "$OUTDIR/" 2>/dev/null || true
fi

rm -f "$LOCK" "$COUNTFILE"
echo "=== R MAGI done: $(date) ==="
ls -la "$OUTDIR/" | head -20
