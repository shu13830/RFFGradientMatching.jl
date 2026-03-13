#!/usr/bin/env julia
# Benchmark: ELMC vs HMC per-iteration cost

include(joinpath(@__DIR__, "common.jl"))
using RFFGradientMatching: BlockedSamplerState

Random.seed!(42)
config = ODE_CONFIGS["LV"]
times, y_obs, y_clean, prob = generate_data(config; N=10, seed=42)
gm = setup_model(RFFGM, config, times, y_obs, prob; n_rff=50, anneal_length=1)
gm.anneal_iter[1] = gm.anneal_length
gm.β[1] = 1.0

param_dict = pack_param_dict(gm)
vars_joint = [:W, :θ]
println("RFFGM Dimension D = ", length(pack_param_vec(gm, vars_joint)))

# --- HMC ---
hmc_block = HMCBlock(gm, vars_joint; n_leapfrog=50, step_size=0.001)
state_hmc = BlockedSamplerState(param_dict)
bs_hmc = BlockedSampler([[hmc_block]], [1.0])
AbstractMCMC.step(gm, bs_hmc, state_hmc, true)  # compile

t_hmc = @elapsed for _ in 1:10
    AbstractMCMC.step(gm, bs_hmc, state_hmc, true)
end
println("HMC: $(round(t_hmc/10, digits=4)) sec/iter")

# --- ELMC ---
elmc_block = ELMCBlock(gm, vars_joint; n_leapfrog=50, step_size=0.001, α=1.0)
state_elmc = BlockedSamplerState(param_dict)
bs_elmc = BlockedSampler([[elmc_block]], [1.0])
AbstractMCMC.step(gm, bs_elmc, state_elmc, true)  # compile

t_elmc = @elapsed for _ in 1:10
    AbstractMCMC.step(gm, bs_elmc, state_elmc, true)
end
println("ELMC: $(round(t_elmc/10, digits=4)) sec/iter")
println("ELMC/HMC ratio: $(round((t_elmc/10)/(t_hmc/10), digits=1))x")

# --- GPGM ---
gm_gp = setup_model(GPGM, config, times, y_obs, prob; anneal_length=1)
gm_gp.anneal_iter[1] = gm_gp.anneal_length
gm_gp.β[1] = 1.0
vars_gp = [:X, :θ]
param_dict_gp = pack_param_dict(gm_gp)
println("\nGPGM Dimension D = ", length(pack_param_vec(gm_gp, vars_gp)))

hmc_gp = HMCBlock(gm_gp, vars_gp; n_leapfrog=50, step_size=0.001)
st_gp = BlockedSamplerState(param_dict_gp)
bs_hmc_gp = BlockedSampler([[hmc_gp]], [1.0])
AbstractMCMC.step(gm_gp, bs_hmc_gp, st_gp, true)
t_hmc_gp = @elapsed for _ in 1:10
    AbstractMCMC.step(gm_gp, bs_hmc_gp, st_gp, true)
end
println("GPGM HMC: $(round(t_hmc_gp/10, digits=4)) sec/iter")

elmc_gp = ELMCBlock(gm_gp, vars_gp; n_leapfrog=50, step_size=0.001, α=1.0)
st_gp2 = BlockedSamplerState(param_dict_gp)
bs_elmc_gp = BlockedSampler([[elmc_gp]], [1.0])
AbstractMCMC.step(gm_gp, bs_elmc_gp, st_gp2, true)
t_elmc_gp = @elapsed for _ in 1:10
    AbstractMCMC.step(gm_gp, bs_elmc_gp, st_gp2, true)
end
println("GPGM ELMC: $(round(t_elmc_gp/10, digits=4)) sec/iter")
println("GPGM ELMC/HMC ratio: $(round((t_elmc_gp/10)/(t_hmc_gp/10), digits=1))x")

# --- Estimated wall time ---
println("\n--- Estimated wall time for 20,000 iterations ---")
println("RFFGM HMC:  $(round(t_hmc/10 * 20000 / 60, digits=1)) min")
println("RFFGM ELMC: $(round(t_elmc/10 * 20000 / 60, digits=1)) min")
println("GPGM HMC:   $(round(t_hmc_gp/10 * 20000 / 60, digits=1)) min")
println("GPGM ELMC:  $(round(t_elmc_gp/10 * 20000 / 60, digits=1)) min")
