#!/usr/bin/env julia
# ELMC vs HMC from converged HMC state
# Phase 1: HMC burn-in+sample (10k iter) → converged state
# Phase 2: From that state, run HMC or ELMC for additional 5k iter
#
# CLI: julia bench_elmc_from_hmc.jl <phase2_method> <alpha> <epsilon> [n_leapfrog_elmc] [N_obs]
#   phase2_method: "HMC" or "ELMC"
#   alpha: ELMC α (ignored if HMC)
#   epsilon: ELMC step_size (ignored if HMC)
#   N_obs: number of observations (default 25)

include(joinpath(@__DIR__, "common.jl"))
using RFFGradientMatching: BlockedSamplerState
using Printf, Serialization

phase2_method = ARGS[1]  # "HMC" or "ELMC"
α_elmc = parse(Float64, ARGS[2])
ε_elmc = parse(Float64, ARGS[3])
n_lf_elmc = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
N_obs = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 25
seed = 42
n_hmc_total = 10_000
n_hmc_warmup = 5_000
n_phase2_burnin = 5_000   # ε adaptation period in Phase 2 (β=1 maintained)
n_phase2_sample = 5_000   # sampling period in Phase 2
n_phase2 = n_phase2_burnin + n_phase2_sample

# Kernel: Matern52 instead of RBF to reduce smoothness bias
const ELMC_KERNEL = 1.0 * with_lengthscale(Matern52Kernel(), 1.0)

# State file: shared across runs with same seed/N/kernel
statefile = joinpath(@__DIR__, "results", "elmc_grid", "hmc_converged_N$(N_obs)_mat52_seed$(seed).jls")

function run_hmc_phase1()
    Random.seed!(seed)
    config = ODE_CONFIGS["LV"]
    times, y_obs, _, prob = generate_data(config; N=N_obs, seed=seed)
    gm = setup_model(GPGM, config, times, y_obs, prob;
        kernel=ELMC_KERNEL, anneal_length=min(999, n_hmc_warmup - 1))

    hmc_blk = HMCBlock(gm, [:X, :θ]; n_leapfrog=50, step_size=0.01)
    bs = BlockedSampler([[hmc_blk]], [1.0])

    chain, logdens = AbstractMCMC.sample(gm, bs, n_hmc_total;
        num_burnin=n_hmc_warmup, anneal=true)

    # θ stats from HMC
    θ_chain = get_θ(gm, chain[n_hmc_warmup+1:end])
    θ_mean = vec(mean(θ_chain, dims=1))

    # Save final state (last sample's param_dict)
    final_dict = pack_param_dict(gm)  # gm is updated to last sample
    mkpath(dirname(statefile))
    serialize(statefile, (final_dict=final_dict, θ_mean_hmc=θ_mean))

    return θ_mean
end

function run_phase2(method, α, ε, n_lf)
    Random.seed!(seed)
    config = ODE_CONFIGS["LV"]
    times, y_obs, _, prob = generate_data(config; N=N_obs, seed=seed)
    gm = setup_model(GPGM, config, times, y_obs, prob; kernel=ELMC_KERNEL, anneal_length=1)
    gm.anneal_iter[1] = gm.anneal_length
    gm.β[1] = 1.0

    # Load converged state
    saved = deserialize(statefile)
    update_model_with_dict!(gm, [:X, :θ], saved.final_dict)

    if method == "HMC"
        blk = HMCBlock(gm, [:X, :θ]; n_leapfrog=50, step_size=0.01)
    else
        blk = ELMCBlock(gm, [:X, :θ]; n_leapfrog=n_lf, step_size=ε, α=α)
    end
    bs = BlockedSampler([[blk]], [1.0])

    t0 = time()
    chain, logdens = AbstractMCMC.sample(gm, bs, n_phase2;
        num_burnin=n_phase2_burnin, anneal=false)
    wall = time() - t0

    # Adapted step_low after burn-in
    adapted_ε = if hasfield(typeof(blk), :step_low)
        blk.step_low
    else
        NaN
    end

    # Accept rate from sampling period only (last n_phase2_sample samples)
    # accept_history is a rolling window of last 100, so compute from chain
    # We track acceptance by comparing consecutive chain entries
    sample_chain = chain[n_phase2_burnin+1:end]
    n_accept = 0
    for i in 2:length(sample_chain)
        if sample_chain[i][:X] !== sample_chain[i-1][:X]
            n_accept += 1
        end
    end
    acc_rate = length(sample_chain) > 1 ? n_accept / (length(sample_chain) - 1) : NaN

    θ_chain = get_θ(gm, sample_chain)
    θ_mean = vec(mean(θ_chain, dims=1))
    θ_true = config.θ_true
    rmsd = sqrt(mean((θ_mean .- θ_true).^2))

    # Log density stats (sampling period only)
    ld_sample = logdens[n_phase2_burnin+1:end]
    ld = [l for l in ld_sample if isfinite(l)]
    ld_mean = length(ld) > 0 ? mean(ld) : NaN

    return θ_mean, rmsd, acc_rate, wall, ld_mean, adapted_ε
end

function main()
    # Phase 1: ensure HMC converged state exists
    if !isfile(statefile)
        @info "Phase 1: Running HMC to convergence..."
        θ_hmc = run_hmc_phase1()
        @info "HMC converged: θ_mean=$(round.(θ_hmc, digits=4))"
    end

    # Phase 2
    θ_mean, rmsd, acc, wall, ld, adapted_ε = run_phase2(phase2_method, α_elmc, ε_elmc, n_lf_elmc)

    # Output CSV line (added adapted_eps column)
    @printf("%s,%.2f,%.4f,%d,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%.6f\n",
            phase2_method, α_elmc, ε_elmc, n_lf_elmc, N_obs,
            isnan(acc) ? -1.0 : acc, rmsd,
            θ_mean[1], θ_mean[2], θ_mean[3], θ_mean[4],
            wall, ld, isnan(adapted_ε) ? -1.0 : adapted_ε)
end

main()
