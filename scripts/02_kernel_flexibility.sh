#!/bin/bash
# Kernel flexibility: RFFGM with Laplace, GenCauchy, ExpPower(β=1.5) on 4 ODEs × 3 N × 3 θ × 10 seeds × 5 γ.
# Produces the non-smooth-kernel cells of Table 5 (paper §4.4).
set -euo pipefail
cd "$(dirname "$0")/.."

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

N_LANES=30
SCRIPT="scripts/bench_core.jl"
KERNELS="Laplace GenCauchy ExpPower"
ODES="LV SIR PST FN"
GAMMAS="0.001 0.01 0.05 0.1 0.5"
SEEDS="42 123 456 789 1234 2345 3456 4567 5678 6789"

OUTDIR="scripts/results/kernel"
mkdir -p "$OUTDIR"
OUTCSV="$OUTDIR/rffgm_kernels.csv"
HEADER="method,kernel,ode,N,seed,theta_id,gamma,rmsd,ess_mean,ess_per_sec,rhat_max,wall_sec,coverage,traj_rmse,resid_mean,gp_marglik,theta_mean,theta_std,convergence"
[ -f "$OUTCSV" ] && [ $(wc -l < "$OUTCSV") -gt 1 ] || echo "$HEADER" > "$OUTCSV"
DONE_KEYS=$(tail -n +2 "$OUTCSV" | awk -F',' '{print $1","$2","$3","$4","$5","$6","$7}')
LOCKFILE="$OUTDIR/.lock"

JOBS=()
for tid in 1 2 3; do
    for N in 10 25 50; do
        for ode in $ODES; do
            for kernel in $KERNELS; do
                for gamma in $GAMMAS; do
                    for seed in $SEEDS; do
                        key="RFFGM,${kernel},${ode},${N},${seed},${tid},$(printf '%.6f' $gamma)"
                        if ! echo "$DONE_KEYS" | grep -Fxq "$key"; then
                            JOBS+=("RFFGM $kernel $ode $N $seed $tid $gamma")
                        fi
                    done
                done
            done
        done
    done
done
TOTAL=${#JOBS[@]}
[ $TOTAL -eq 0 ] && { echo "all done"; exit 0; }
echo "kernel flex: $TOTAL jobs across $N_LANES lanes"

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
