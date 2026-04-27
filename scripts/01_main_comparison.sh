#!/bin/bash
# Main comparison: RFFGM, GPGM, Julia MAGI on 4 ODEs × 3 N × 3 θ-patterns × 10 seeds × 5 γ.
# Produces the raw data behind Tables 1 and 2 (paper/main.tex) and Appendix H θ-robustness.
#
# Usage: bash scripts/01_main_comparison.sh [method]
#   method: one of RFFGM, GPGM, MAGI, or all (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

METHOD="${1:-all}"
N_LANES=30
SCRIPT="scripts/bench_core.jl"
KERNELS="RBF Matern52"
ODES="LV SIR PST FN"
GAMMAS="0.001 0.01 0.05 0.1 0.5"
SEEDS="42 123 456 789 1234 2345 3456 4567 5678 6789"

run_method() {
    local m=$1
    local OUTDIR="scripts/results/main/${m,,}"
    mkdir -p "$OUTDIR"
    local OUTCSV="$OUTDIR/results.csv"
    local HEADER="method,kernel,ode,N,seed,theta_id,gamma,rmsd,ess_mean,ess_per_sec,rhat_max,wall_sec,coverage,traj_rmse,resid_mean,gp_marglik,theta_mean,theta_std,convergence"
    if [ ! -f "$OUTCSV" ] || [ "$(wc -l < "$OUTCSV")" -le 1 ]; then
        echo "$HEADER" > "$OUTCSV"
    fi
    local DONE_KEYS
    DONE_KEYS=$(tail -n +2 "$OUTCSV" | awk -F',' '{print $1","$2","$3","$4","$5","$6","$7}')
    local LOCKFILE="$OUTDIR/.lock"

    local JOBS=()
    for tid in 1 2 3; do
        for N in 10 25 50; do
            for ode in $ODES; do
                for kernel in $KERNELS; do
                    for gamma in $GAMMAS; do
                        for seed in $SEEDS; do
                            local key="${m},${kernel},${ode},${N},${seed},${tid},$(printf '%.6f' "$gamma")"
                            if ! echo "$DONE_KEYS" | grep -Fxq "$key"; then
                                JOBS+=("$m $kernel $ode $N $seed $tid $gamma")
                            fi
                        done
                    done
                done
            done
        done
    done
    local TOTAL=${#JOBS[@]}
    [ "$TOTAL" -eq 0 ] && { echo "[$m] all done"; return; }
    echo "[$m] $TOTAL jobs across $N_LANES lanes"

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
        local lane=$((i % N_LANES))
        eval "LANE_${lane}+=(\"${JOBS[$i]}\")"
    done
    for l in $(seq 0 $((N_LANES - 1))); do
        eval "run_lane $l \"\${LANE_${l}[@]}\"" &
    done
    wait
    echo "[$m] done"
}

case "$METHOD" in
    all)
        run_method RFFGM
        run_method GPGM
        run_method MAGI
        ;;
    RFFGM|GPGM|MAGI)
        run_method "$METHOD"
        ;;
    *)
        echo "Unknown method: $METHOD (use RFFGM, GPGM, MAGI, or all)"; exit 1;;
esac
