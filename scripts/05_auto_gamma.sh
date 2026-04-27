#!/bin/bash
# Auto-γ selection heuristic validation: 4 ODEs × 3 N × 3 θ × 5 γ × 5 seeds = 900 RFFGM-RBF chains.
# Produces Appendix K Table 9.
set -euo pipefail
cd "$(dirname "$0")/.."

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_LANES=30
SCRIPT="scripts/bench_gamma_select.jl"
NITER="${1:-20000}"

OUTDIR="scripts/results/auto_gamma"
mkdir -p "$OUTDIR"
OUTCSV="$OUTDIR/results.csv"
HEADER="ode,N,seed,theta_id,gamma,rmsd,ess_mean,ess_per_sec,rhat_max,wall_sec,coverage,resid_mean,ule_mean,theta_mean"
if [ ! -f "$OUTCSV" ] || [ "$(wc -l < "$OUTCSV")" -le 1 ]; then
    echo "$HEADER" > "$OUTCSV"
fi
DONE_KEYS=$(tail -n +2 "$OUTCSV" | awk -F',' '{print $1","$2","$3","$4","$5}')
LOCKFILE="$OUTDIR/.lock"

GAMMAS="0.001 0.01 0.03 0.1 0.3"
SEEDS="42 123 456 789 1234"
ODES="LV SIR PST FN"

JOBS=()
for seed in $SEEDS; do
    for tid in 1 2 3; do
        for N in 10 25 50; do
            for ode in $ODES; do
                for gamma in $GAMMAS; do
                    key="${ode},${N},${seed},${tid},$(printf '%.6f' "$gamma")"
                    if ! echo "$DONE_KEYS" | grep -Fxq "$key"; then
                        JOBS+=("$ode $N $seed $tid $gamma $NITER")
                    fi
                done
            done
        done
    done
done
TOTAL=${#JOBS[@]}
[ $TOTAL -eq 0 ] && { echo "all done"; exit 0; }
echo "auto-γ: $TOTAL jobs across $N_LANES lanes"

run_lane() {
    local lane_id=$1; shift
    for j in "$@"; do
        local args=($j)
        local tmpf
        tmpf=$(mktemp "$OUTDIR/tmp_XXXX.csv")
        if julia --project=. "$SCRIPT" "${args[@]}" > "$tmpf" 2>/dev/null && [ -s "$tmpf" ]; then
            (flock 200; cat "$tmpf" >> "$OUTCSV") 200>"$LOCKFILE"
        fi
        rm -f "$tmpf"
    done
}
export -f run_lane
export OUTDIR OUTCSV LOCKFILE SCRIPT

for l in $(seq 0 $((N_LANES - 1))); do eval "declare -a LANE_${l}=()"; done
for i in $(seq 0 $((TOTAL - 1))); do
    lane=$((i % N_LANES))
    eval "LANE_${lane}+=(\"${JOBS[$i]}\")"
done
for l in $(seq 0 $((N_LANES - 1))); do eval "run_lane $l \"\${LANE_${l}[@]}\"" & done
wait
echo "done"
