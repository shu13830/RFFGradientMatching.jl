#!/bin/bash
# LVC scaling experiment: RFFGM vs GPGM at K ∈ {2, 5, 10, 15}, 10 seeds, N=40, γ=0.1.
# Produces Table 6 and Figure 3 data. R MAGI rows are filled in by `04_r_magi_baseline.sh`.
set -euo pipefail
cd "$(dirname "$0")/.."

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

MAX_PARALLEL=8

OUTDIR="scripts/results/scaling"
mkdir -p "$OUTDIR"
OUTCSV="$OUTDIR/results.csv"
HEADER="method,K,n_params,seed,gamma,rmsd,ess_mean,rhat_max,wall_sec"

if [ ! -f "$OUTCSV" ] || [ ! -s "$OUTCSV" ]; then
    echo "$HEADER" > "$OUTCSV"
fi
LOCKFILE="$OUTDIR/.lock"

SEEDS=(42 123 456 789 1234 2345 3456 4567 5678 6789)
K_VALUES=(2 5 10 15)
GAMMA=0.1

DONE_KEYS=""
if [ -f "$OUTCSV" ] && [ "$(wc -l < "$OUTCSV")" -gt 1 ]; then
    DONE_KEYS=$(tail -n +2 "$OUTCSV" | awk -F',' '{print $1","$2","$4}')
fi

JOBS=()
for K in "${K_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for method in RFFGM GPGM; do
            key="${method},${K},${seed}"
            if ! echo "$DONE_KEYS" | grep -Fxq "$key"; then
                JOBS+=("$method $K $seed")
            fi
        done
    done
done
TOTAL=${#JOBS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "LVC scaling: all done (resume), nothing to do"
    exit 0
fi
echo "LVC scaling: $TOTAL jobs (MAX_PARALLEL=$MAX_PARALLEL)"

IDX=0
while [ $IDX -lt $TOTAL ]; do
    BATCH_END=$((IDX + MAX_PARALLEL))
    [ $BATCH_END -gt $TOTAL ] && BATCH_END=$TOTAL
    PIDS=()
    TMPFILES=()
    for i in $(seq $IDX $((BATCH_END - 1))); do
        args=(${JOBS[$i]})
        method=${args[0]}
        K=${args[1]}
        seed=${args[2]}
        tmpf=$(mktemp "$OUTDIR/tmp_XXXX.csv")
        TMPFILES+=("$tmpf")
        julia --project=. -e "
include(\"scripts/common.jl\")
config, aux = make_lvc_config($K)
Random.seed!($seed)
times, y_obs, y_clean, prob = generate_data(config; N=40, seed=$seed)
method_type = Dict(\"RFFGM\"=>RFFGM, \"GPGM\"=>GPGM)[\"$method\"]
gm = setup_model(method_type, config, times, y_obs, prob; n_rff=100)
gm.odegrad.γ = $GAMMA
cache_e_cov_chol!(gm)
bs = create_blocked_sampler(gm)
t0 = time()
chain, logdens = AbstractMCMC.sample(gm, bs, 20000; num_burnin=10000, anneal=true)
wall = time() - t0
θ_chain = get_θ(gm, chain[10001:end])
θ_mean = vec(mean(θ_chain, dims=1))
rmsd = sqrt(mean((θ_mean .- config.θ_true).^2))
chn = Chains(θ_chain)
ess_vals = MCMCDiagnosticTools.ess(chn).nt.ess
rhat_vals = MCMCDiagnosticTools.rhat(chn).nt.rhat
using Printf
@printf(\"%s,%d,%d,%d,%.3f,%.4f,%.1f,%.3f,%.1f\n\",
    \"$method\", $K, $((K * (K - 1))), $seed, $GAMMA,
    rmsd, mean(ess_vals), maximum(rhat_vals), wall)
" > "$tmpf" 2>/dev/null &
        PIDS+=($!)
    done
    for j in "${!PIDS[@]}"; do
        wait "${PIDS[$j]}" || true
        if [ -s "${TMPFILES[$j]}" ]; then
            (flock 200; cat "${TMPFILES[$j]}" >> "$OUTCSV") 200>"$LOCKFILE"
        fi
        rm -f "${TMPFILES[$j]}"
    done
    IDX=$BATCH_END
done
rm -f "$LOCKFILE"
echo "done"
