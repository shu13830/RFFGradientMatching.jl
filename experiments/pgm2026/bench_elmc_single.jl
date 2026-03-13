#!/usr/bin/env julia
# Single ELMC grid search run: takes α, ε, n_leapfrog as CLI args
# Outputs one-line CSV result to stdout

include(joinpath(@__DIR__, "common.jl"))
using RFFGradientMatching: BlockedSamplerState
using Printf

α = parse(Float64, ARGS[1])
ε = parse(Float64, ARGS[2])
n_lf = parse(Int, ARGS[3])
N_obs = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
n_iter = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2000
n_warmup = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1000

Random.seed!(42)
config = ODE_CONFIGS["LV"]
times, y_obs, _, prob = generate_data(config; N=N_obs, seed=42)

gm = setup_model(GPGM, config, times, y_obs, prob; anneal_length=min(999, n_warmup-1))

bs = let
    block = ELMCBlock(gm, [:X, :θ]; n_leapfrog=n_lf, step_size=ε, α=α)
    BlockedSampler([[block]], [1.0])
end

t0 = time()
chain, logdens = AbstractMCMC.sample(gm, bs, n_iter; num_burnin=n_warmup, anneal=true)
wall = time() - t0

blk = bs.blocks[1][1]
n_acc = sum(blk.accept_history)
n_tot = length(blk.accept_history)
acc_rate = n_tot > 0 ? n_acc / n_tot : NaN

θ_chain = get_θ(gm, chain[n_warmup+1:end])
θ_true = [0.67, 1.33, 1.0, 1.0]
n_samples = size(θ_chain, 1)
θ_mean = n_samples > 0 ? vec(mean(θ_chain, dims=1)) : fill(NaN, 4)
rmsd = sqrt(mean((θ_mean .- θ_true).^2))

@printf("%.2f,%.4f,%d,%d,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f\n",
        α, ε, n_lf, N_obs, acc_rate, rmsd, θ_mean[1], θ_mean[2], θ_mean[3], θ_mean[4], wall)
