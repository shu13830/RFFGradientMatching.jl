#=====================================================================
# ***Benchmark RFFGM vs GPGM on noisy predator–prey time–series***
---
=====================================================================#
using Random, LinearAlgebra, Statistics, Dates, Printf
using DifferentialEquations
using KernelFunctions
using Distributions
using RFFGradientMatching
using AbstractMCMC, MCMCChains, MCMCDiagnosticTools
using JLD2, CSV, DataFrames, Plots
using Plots.PlotMeasures
using ProgressMeter

default(size=(800,400))

#---------------------------------------------------------------------
# 0.  GLOBAL SETTINGS
#---------------------------------------------------------------------
const N_REP       = 1                 # replicates / cell
const OBS_NOISE   = 0.10
const TSPAN       = (0.0, 2.0)
const MCMC_ITERS  = 10_000
const BURN_IN     = 5_000
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)
PLOT_DIR    = joinpath(RESULTS_DIR, "plots")
mkpath(PLOT_DIR)

const N_GRID = [10, 25, 40]   # data length
const L_GRID = [50, 100, 200]       # RFF dimension
linestyles = Dict(10=>:solid, 25=>:dash, 40=>:dot)

#---------------------------------------------------------------#
# 1. TRUE MODEL (constant across cells)
#---------------------------------------------------------------#
true_params = [2.0, 1.0, 4.0, 1.0]          # α,β,δ,γ
u0          = [5.0, 3.0]
pp!         = RFFGradientMatching.lotkavolterrapredatorprey!

α², ℓ = 1.0, 1.0
kernel = α² * with_lengthscale(RBFKernel(), ℓ)

#---------------------------------------------------------------#
# 2. HELPERS
#---------------------------------------------------------------#
function ess_per_sec(chain::Chains)
    ess = MCMCDiagnosticTools.ess(chain).nt.ess
    return mean(ess) / (MCMC_ITERS - BURN_IN)
end

rmse(θ̄) = sqrt(mean((θ̄ .- true_params).^2))

#---------------------------------------------------------------#
# 3. SINGLE‑RUN WORKER
#---------------------------------------------------------------#
function run_gpgm(N::Int, rep::Int)
    rng        = MersenneTwister(10_000*N + rep)
    t_obs      = collect(range(TSPAN[1], TSPAN[2], length=N))
    prob_true  = ODEProblem(pp!, u0, TSPAN, true_params)
    y_true     = Array(solve(prob_true, Tsit5(), saveat=t_obs))
    y_obs      = y_true .+ OBS_NOISE * randn(rng, size(y_true))

    gm_gp = GPGM(t_obs, y_obs, prob_true, "PredatorPrey";
                 k=kernel, state_noise_std=1e-3, obs_noise_std=OBS_NOISE)
    optimize_ϕ_and_σ!(gm_gp)
    bs = BlockedSampler([[HMCBlock(gm_gp, [:X,:θ], n_leapfrog=10, step_size=0.05, metric=:diag)]],[1.0])
    init_gm_gp = pack_param_dict(gm_gp)
    chain_gp, logdens_gp = sample(rng, gm_gp, bs, MCMC_ITERS; init_params=init_gm_gp, num_burnin=BURN_IN)

    θ_samples = get_θ(gm_gp, chain_gp[BURN_IN+1:end])
    ess_gp  = ess_per_sec(Chains(θ_samples))
    sqdiff_θ_samples = [(s .- true_params).^2 |> mean for s in θ_samples]
    mean_θ = mean(θ_samples, dims=1)[:]
    std_θ = std(θ_samples, dims=1)[:]
    rmsd_gp = sqrt(mean(sqdiff_θ_samples))

    return t_obs, y_obs, prob_true,
        (N=N, rep=rep, logdens=logdens_gp, ess=ess_gp, rmsd=rmsd_gp,
        mean_θ=mean_θ, std_θ=std_θ, chain=chain_gp, gm=gm_gp)
end

function run_rffgm(t_obs, y_obs, prob, N::Int, L::Int, rep::Int, seed::Int)
    rng = MersenneTwister(seed)
    gm_rf = RFFGM(t_obs, y_obs, prob, "PredatorPrey";
                 k=kernel, n_rff=L, state_noise_std=1e-3, obs_noise_std=OBS_NOISE)
    optimize_ϕ_and_σ!(gm_rf)
    bs = BlockedSampler([[HMCBlock(gm_rf, [:W,:θ], n_leapfrog=10, step_size=0.05, metric=:diag)]], [1.0])
    init_gm_rf = pack_param_dict(gm_rf)
    chain_rf, logdens_rf = sample(rng, gm_rf, bs, MCMC_ITERS; init_params=init_gm_rf, num_burnin=BURN_IN)

    θ_samples = get_θ(gm_rf, chain_rf[BURN_IN+1:end])
    ess_rf  = ess_per_sec(Chains(θ_samples))
    sqdiff_θ_samples = [(s .- true_params).^2 |> mean for s in θ_samples]
    mean_θ = mean(θ_samples, dims=1)[:]
    std_θ = std(θ_samples, dims=1)[:]
    rmsd_rf = sqrt(mean(sqdiff_θ_samples))

    return (N=N, L=L, rep=rep,
            logdens=logdens_rf, ess=ess_rf, rmsd=rmsd_rf,
            mean_θ=mean_θ, std_θ=std_θ,
            chain=chain_rf, gm=gm_rf)
end

#---------------------------------------------------------------#
# 4. FACTORIAL EXECUTION
#---------------------------------------------------------------#
@showprogress "Running experiments…" for N in N_GRID, rep in 1:N_REP
    ## -- GPGM once per (N,rep)
    t_obs, y_obs, prob, res_gp = run_gpgm(N, rep)

    ## save GPGM artefacts
    gdir = joinpath(RESULTS_DIR, "GPGM", @sprintf("N=%03d", N))
    mkpath(gdir)
    @save joinpath(gdir, "rep$(rep).jld2") res_gp

    ## -- RFFGM for every L on the SAME data
    for L in L_GRID
        seed = 1_000_000*N + 1_000*L + rep
        res_rf = run_rffgm(t_obs, y_obs, prob, N, L, rep, seed)

        rdir = joinpath(RESULTS_DIR, "RFFGM", @sprintf("N=%03d-L=%03d",N,L))
        mkpath(rdir)
        @save joinpath(rdir, "rep$(rep).jld2") res_rf
    end
end

#---------------------------------------------------------------#
# 5. WRITE CSVs
#---------------------------------------------------------------#
gpgm_df  = DataFrame();  rffgm_df = DataFrame()
for N in N_GRID, rep in 1:N_REP
    @load joinpath(RESULTS_DIR, "GPGM", @sprintf("N=%03d", N), "rep$(rep).jld2") res_gp
    push!(gpgm_df, (N=N, rep=rep, ess=res_gp.ess, rmsd=res_gp.rmsd))
end
for N in N_GRID, L in L_GRID, rep in 1:N_REP
    @load joinpath(RESULTS_DIR, "RFFGM", @sprintf("N=%03d-L=%03d", N, L), "rep$(rep).jld2") res_rf
    push!(rffgm_df, (N=N, L=L, rep=rep, ess=res_rf.ess, rmsd=res_rf.rmsd))
end
CSV.write(joinpath(RESULTS_DIR, "gpgm_summary.csv"), gpgm_df)
CSV.write(joinpath(RESULTS_DIR, "rffgm_summary.csv"), rffgm_df)

println("✓ Finished.  Results in $(RESULTS_DIR)")

#---------------------------------------------------------------#
# 6. PLOT LOGDENSITY TRACE
#---------------------------------------------------------------#
function load_gpgm_logdens()
    traces = Dict{Int,Vector{Float64}}()
    for N in N_GRID
        @load joinpath(RESULTS_DIR, "GPGM", @sprintf("N=%03d",N), "rep1.jld2") res_gp
        logd = RFFGradientMatching.get_logdensity(res_gp.gm, res_gp.chain)
        traces[N] = logd
    end
    return traces
end

function load_rffgm_logdens()
    traces = Dict{Int, Dict{Int,Vector{Float64}}}()
    for N in N_GRID
        traces[N] = Dict{Int,Vector{Float64}}()
        for L in L_GRID
            @load joinpath(RESULTS_DIR, "RFFGM", @sprintf("N=%03d-L=%03d",N,L), "rep1.jld2") res_rf
            logd = RFFGradientMatching.get_logdensity(res_rf.gm, res_rf.chain)
            traces[N][L] = logd
        end
    end
    return traces
end

gp_logdens_traces = load_gpgm_logdens()
rf_logdens_traces = load_rffgm_logdens()

pl = plot(gp_logdens_traces[10], c=:blue, lw=1.5, label="GPGM(N=10)")
plot!(gp_logdens_traces[25], c=:blue, ls=:dashdot, lw=1.5, label="GPGM(N=25)")
plot!(gp_logdens_traces[40], c=:blue, ls=:dash, lw=1.5, label="GPGM(N=40)")
plot!(rf_logdens_traces[10][100], c=:red, lw=1.5, label="RFFGM(N=10)")
plot!(rf_logdens_traces[25][100], c=:red, ls=:dashdot, lw=1.5, label="RFFGM(N=25)")
plot!(rf_logdens_traces[40][100], c=:red, ls=:dash, lw=1.5, label="RFFGM(N=40)")
ylims!(-5000, 1000)
plot!(
    xlabel="MCMC Iteration", 
    ylabel="Log density", 
    # legend=:outerright,
    size=(1200,500), 
    margin=10mm,
    fontfamily="Times",
    labelfontsize=20,
    tickfontsize=15,
    legendfontsize=15
)

# 
savefig(pl, joinpath(RESULTS_DIR, "plots", "GPGMvsRFFGM_convergence.png"))
