# -----------------------------------------------------------------
# PGM 2026 Experiments — Common Infrastructure
#
# Include this file from each experiment script:
#   include(joinpath(@__DIR__, "common.jl"))
# -----------------------------------------------------------------

using Random, LinearAlgebra, Statistics, Printf, Dates, JSON
using DifferentialEquations
using KernelFunctions
using Distributions
using RFFGradientMatching
using AbstractMCMC, MCMCChains, MCMCDiagnosticTools
using CSV, DataFrames
using ArgParse
using GeneralizedRFF

import RFFGradientMatching:
    lotkavolterrapredatorprey!, lotkavolterracompetition!,
    fitzhughnagumo!, signaltransductioncascade!, lorenz96!, sir!,
    SigmoidKernel, pack_param_dict, get_logdensity,
    build_rff_basis, rff_approx_error,
    Hmat, W2X, calc_destandardized_X, get_y_std

# ── Constants ──────────────────────────────────────────────────────

const MCMC_ITERATIONS  = 10_000
const MCMC_WARMUP      = 5_000
const ANNEAL_LENGTH    = 1_000
const STATE_NOISE_STD  = 1e-3
const DEFAULT_N_RFF    = 100

const DEFAULT_SEEDS = [42, 123, 456, 789, 1234, 2345, 3456, 4567, 5678, 6789]

const RESULTS_BASE = joinpath(@__DIR__, "results")

# ── ODE Configuration ─────────────────────────────────────────────

struct ODEConfig
    name::String
    f!::Function
    θ_true::Vector{Float64}
    u0::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    noise_std::Float64
    kernel::KernelFunctions.Kernel
    param_names::Vector{String}
    component_names::Vector{String}
end

const ODE_CONFIGS = Dict(
    "LV" => ODEConfig(
        "LV",
        lotkavolterrapredatorprey!,
        [2.0, 1.0, 4.0, 1.0],
        [5.0, 3.0],
        (0.0, 2.0),
        0.5,
        1.0 * with_lengthscale(SqExponentialKernel(), 1.0),
        ["a", "b", "c", "d"],
        ["prey", "predator"],
    ),
    "FN" => ODEConfig(
        "FN",
        fitzhughnagumo!,
        [3.0, 0.2, 0.2],
        [-1.0, 1.0],
        (0.0, 2.0),
        0.5,
        1.0 * with_lengthscale(SqExponentialKernel(), 1.0),
        ["theta1", "theta2", "theta3"],
        ["V", "R"],
    ),
    "PST" => ODEConfig(
        "PST",
        signaltransductioncascade!,
        [0.07, 0.6, 0.05, 0.3, 0.017, 0.3],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        (0.0, 100.0),
        0.01,
        1.0 * SigmoidKernel(1.0, 0.0),
        ["k1", "k2", "k3", "k4", "V", "Km"],
        ["S", "dS", "R", "Rs", "Rpp"],
    ),
)

# ── Kernel Configurations (Exp3) ─────────────────────────────────

struct KernelConfig
    name::String
    short_name::String
    kernel::KernelFunctions.Kernel
    methods::Vector{DataType}
end

const KERNEL_CONFIGS = Dict(
    "RBF" => KernelConfig(
        "RBF (Gaussian)", "RBF",
        1.0 * with_lengthscale(SqExponentialKernel(), 1.0),
        [RFFGM, GPGM],
    ),
    "Matern52" => KernelConfig(
        "Matern-5/2", "Mat52",
        1.0 * with_lengthscale(Matern52Kernel(), 1.0),
        [RFFGM, GPGM],
    ),
    "Laplace" => KernelConfig(
        "Laplace (Matern-1/2)", "Lap",
        1.0 * with_lengthscale(ExponentialPowerKernel(γ=1.0), 1.0),
        [RFFGM],
    ),
    "GenCauchy" => KernelConfig(
        "Generalized Cauchy (a=1.5, b=1.5)", "GCau",
        1.0 * with_lengthscale(GeneralizedCauchyKernel(1.5, 1.5), 1.0),
        [RFFGM],
    ),
    "ExpPower" => KernelConfig(
        "Exponential Power (g=1.5)", "ExpP",
        1.0 * with_lengthscale(ExponentialPowerKernel(γ=1.5), 1.0),
        [RFFGM],
    ),
)

# Ordered kernel list for consistent output
const KERNEL_ORDER = ["RBF", "Matern52", "Laplace", "GenCauchy", "ExpPower"]

function parse_kernels(kernel_str::String)
    kernel_str == "ALL" ? KERNEL_ORDER : String.(split(kernel_str, ","))
end

# ── RFF Approximation Error (Exp9) ──────────────────────────────

function compute_rff_approx_error(gm::RFFGM)
    mean(rff_approx_error(gp.h, gp.k, gp.z) for gp in gm.gp)
end

# ── LVC Closure ODE (Exp4) ────────────────────────────────────────

"""
    make_lvc_ode(K, r_true, Ks_true; diag_alpha=ones(K))

Create a closure ODE function for LV Competition where only off-diagonal
alpha entries are estimated. Returns an ODE function with
`p = [off-diagonal alpha]` of length K*(K-1).

Off-diagonal elements are packed row-major:
  [α_{1,2}, ..., α_{1,K}, α_{2,1}, α_{2,3}, ..., α_{K,K-1}]
"""
function make_lvc_ode(K::Int, r_true::Vector{Float64}, Ks_true::Vector{Float64};
    diag_alpha::Vector{Float64}=ones(K))
    @assert length(r_true) == K
    @assert length(Ks_true) == K
    @assert length(diag_alpha) == K

    function lvc_offdiag!(du, u, p_offdiag, t)
        α = zeros(eltype(p_offdiag), K, K)
        idx = 1
        for i in 1:K
            for j in 1:K
                if i == j
                    α[i, j] = diag_alpha[i]
                else
                    α[i, j] = p_offdiag[idx]
                    idx += 1
                end
            end
        end
        for i in 1:K
            interaction = sum(α[i, j] * u[j] for j in 1:K)
            du[i] = r_true[i] * u[i] * (1 - interaction / Ks_true[i])
        end
    end
    return lvc_offdiag!
end

"""
    make_lvc_config(K; seed_base=42, noise_std=0.3, tspan=(0.0, 5.0), N_default=40)

Generate a deterministic LVC configuration for K species.
True parameters are generated from seed_base so they are reproducible.
"""
function make_lvc_config(K::Int;
    seed_base::Int=42,
    noise_std::Float64=0.3,
    tspan::Tuple{Float64,Float64}=(0.0, 5.0),
)
    rng = MersenneTwister(seed_base)

    r_true = 1.0 .+ 0.5 .* rand(rng, K)           # growth rates ~ U[1.0, 1.5]
    Ks_true = 10.0 .+ 5.0 .* rand(rng, K)           # carrying capacities ~ U[10, 15]
    diag_alpha = ones(K)
    alpha_offdiag_true = 0.1 .+ 0.3 .* rand(rng, K * (K - 1))  # off-diag ~ U[0.1, 0.4]

    u0 = 5.0 .* ones(K)

    f! = make_lvc_ode(K, r_true, Ks_true; diag_alpha=diag_alpha)

    component_names = ["x$i" for i in 1:K]
    param_names = String[]
    for i in 1:K, j in 1:K
        i != j && push!(param_names, "alpha_$(i)_$(j)")
    end

    config = ODEConfig(
        "LVC_K$(K)", f!, alpha_offdiag_true, u0, tspan,
        noise_std,
        1.0 * with_lengthscale(SqExponentialKernel(), 1.0),
        param_names, component_names,
    )

    # Store auxiliary info for MAGI data export
    aux = Dict(
        "K" => K,
        "r_true" => r_true,
        "Ks_true" => Ks_true,
        "diag_alpha" => diag_alpha,
        "alpha_offdiag_true" => alpha_offdiag_true,
    )

    return config, aux
end

# ── Data Generation ───────────────────────────────────────────────

function generate_data(config::ODEConfig; N::Int, seed::Int)
    rng = MersenneTwister(seed)
    times = collect(range(config.tspan[1], config.tspan[2], length=N))
    prob = ODEProblem(config.f!, config.u0, config.tspan, config.θ_true)
    sol = solve(prob, Tsit5(), saveat=times)
    y_clean = Array(sol)       # K × N
    y_obs = y_clean .+ config.noise_std .* randn(rng, size(y_clean))
    return times, y_obs, y_clean, prob
end

# ── Kernel fallback for RFFGM ─────────────────────────────────────

# GenCauchy: heavy-tailed, flexible smoothness — suitable for PST's sharp transitions
const RFFGM_SIGMOID_FALLBACK = 1.0 * with_lengthscale(GeneralizedCauchyKernel(1.5, 1.5), 1.0)

"""Return a kernel compatible with RFFGM (GenCauchy fallback for SigmoidKernel)."""
function rffgm_compatible_kernel(kernel::KernelFunctions.Kernel)
    base_k = kernel isa KernelFunctions.TransformedKernel ? kernel.kernel :
             kernel isa KernelFunctions.ScaledKernel ? kernel.kernel : kernel
    base_k = base_k isa KernelFunctions.ScaledKernel ? base_k.kernel : base_k
    if base_k isa SigmoidKernel
        @info "SigmoidKernel not supported for RFFGM; using GenCauchy(1.5,1.5) fallback"
        return RFFGM_SIGMOID_FALLBACK
    end
    return kernel
end

# ── Model Setup ───────────────────────────────────────────────────

function setup_model(::Type{RFFGM}, config::ODEConfig, times, y_obs, prob;
    kernel=config.kernel, n_rff=DEFAULT_N_RFF, anneal_length=ANNEAL_LENGTH)

    kernel = rffgm_compatible_kernel(kernel)
    gm = RFFGM(times, y_obs, prob, config.name;
        k=kernel, state_noise_std=STATE_NOISE_STD, obs_noise_std=config.noise_std,
        n_rff=n_rff, anneal_length=anneal_length)

    n_θ = length(prob.p)
    set_priortransform_on_θ!(gm, fill(Normal(0.0, 1.0), n_θ), fill(log, n_θ))
    optimize_ϕ_and_σ!(gm)
    optimize_u!(gm)
    return gm
end

function setup_model(::Type{GPGM}, config::ODEConfig, times, y_obs, prob;
    kernel=config.kernel, anneal_length=ANNEAL_LENGTH, kwargs...)

    gm = GPGM(times, y_obs, prob, config.name;
        k=kernel, state_noise_std=STATE_NOISE_STD, obs_noise_std=config.noise_std,
        anneal_length=anneal_length)

    n_θ = length(prob.p)
    set_priortransform_on_θ!(gm, fill(Normal(0.0, 1.0), n_θ), fill(log, n_θ))
    optimize_ϕ_and_σ!(gm)
    optimize_u!(gm)
    return gm
end

# ── Sampler Creation ──────────────────────────────────────────────

function create_blocked_sampler(gm::RFFGM;
    step_size_latent=0.05, step_size_joint=0.01, step_size_theta=0.05,
    n_leapfrog=10)

    block_W  = HMCBlock(gm, [:W];     n_leapfrog=n_leapfrog, step_size=step_size_latent, metric=:diag)
    block_Wθ = HMCBlock(gm, [:W, :θ]; n_leapfrog=n_leapfrog, step_size=step_size_joint,  metric=:diag)
    block_θ  = HMCBlock(gm, [:θ];     n_leapfrog=n_leapfrog, step_size=step_size_theta,  metric=:diag)
    return BlockedSampler([[block_W], [block_Wθ], [block_θ]], [0.4, 0.4, 0.2])
end

function create_blocked_sampler(gm::GPGM;
    step_size_latent=0.05, step_size_joint=0.01, step_size_theta=0.05,
    n_leapfrog=10)

    block_X  = HMCBlock(gm, [:X];     n_leapfrog=n_leapfrog, step_size=step_size_latent, metric=:diag)
    block_Xθ = HMCBlock(gm, [:X, :θ]; n_leapfrog=n_leapfrog, step_size=step_size_joint,  metric=:diag)
    block_θ  = HMCBlock(gm, [:θ];     n_leapfrog=n_leapfrog, step_size=step_size_theta,  metric=:diag)
    return BlockedSampler([[block_X], [block_Xθ], [block_θ]], [0.4, 0.4, 0.2])
end

# ── Single Experiment Runner ──────────────────────────────────────

function run_single_experiment(
    method::Type{<:Union{RFFGM,GPGM}}, config::ODEConfig;
    N::Int, seed::Int,
    kernel=config.kernel,
    n_rff::Int=DEFAULT_N_RFF,
    n_iterations::Int=MCMC_ITERATIONS,
    n_warmup::Int=MCMC_WARMUP,
    anneal_length::Int=ANNEAL_LENGTH,
)
    Random.seed!(seed)
    times, y_obs, y_clean, prob = generate_data(config; N=N, seed=seed)

    # Auto-clamp anneal_length to fit within warmup
    _anneal = min(anneal_length, max(n_warmup - 1, 0))
    gm = setup_model(method, config, times, y_obs, prob;
        kernel=kernel, n_rff=n_rff, anneal_length=_anneal)
    bs = create_blocked_sampler(gm)

    t_start = time()
    chain, logdens = AbstractMCMC.sample(gm, bs, n_iterations;
        num_burnin=n_warmup, anneal=true)
    wall_time = time() - t_start

    # Extract post-warmup θ samples
    θ_chain = get_θ(gm, chain[n_warmup+1:end])  # n_samples × n_params

    return (;
        θ_chain, chain, logdens, gm,
        wall_time, times, y_obs, y_clean,
        config, N, seed, method=string(nameof(method)),
    )
end

# ── Metrics ───────────────────────────────────────────────────────

function compute_rmsd(θ_chain::AbstractMatrix, θ_true::Vector{Float64})
    θ_mean = vec(mean(θ_chain, dims=1))
    return sqrt(mean((θ_mean .- θ_true).^2))
end

function compute_ess(θ_chain::AbstractMatrix)
    chn = Chains(θ_chain)
    ess_df = MCMCDiagnosticTools.ess(chn)
    return ess_df.nt.ess
end

function compute_rhat(θ_chain::AbstractMatrix)
    chn = Chains(θ_chain)
    rhat_df = MCMCDiagnosticTools.rhat(chn)
    return rhat_df.nt.rhat
end

function compute_all_metrics(result, θ_true::Vector{Float64})
    θ_chain = result.θ_chain
    rmsd = compute_rmsd(θ_chain, θ_true)
    ess_vals = compute_ess(θ_chain)
    ess_mean = mean(ess_vals)
    ess_per_sec = ess_mean / result.wall_time
    θ_mean = vec(mean(θ_chain, dims=1))
    θ_std = vec(std(θ_chain, dims=1))
    rhat_vals = compute_rhat(θ_chain)

    return (;
        rmsd, ess_vals, ess_mean, ess_per_sec,
        θ_mean, θ_std, rhat_vals,
        wall_time=result.wall_time,
        N=result.N, seed=result.seed, method=result.method,
    )
end

# ── Result I/O ────────────────────────────────────────────────────

function make_result_row(metrics; ode_key::String, extra_cols=Dict{String,Any}())
    row = Dict{String,Any}(
        "ode"         => ode_key,
        "method"      => metrics.method,
        "N"           => metrics.N,
        "seed"        => metrics.seed,
        "rmsd"        => metrics.rmsd,
        "ess_mean"    => metrics.ess_mean,
        "time_sec"    => metrics.wall_time,
        "ess_per_sec" => metrics.ess_per_sec,
        "rhat_max"    => maximum(metrics.rhat_vals),
    )
    for (i, val) in enumerate(metrics.θ_mean)
        row["theta_$(i)_mean"] = val
    end
    for (i, val) in enumerate(metrics.θ_std)
        row["theta_$(i)_std"] = val
    end
    for (i, val) in enumerate(metrics.ess_vals)
        row["ess_$(i)"] = val
    end
    for (i, val) in enumerate(metrics.rhat_vals)
        row["rhat_$(i)"] = val
    end
    merge!(row, extra_cols)
    return row
end

function save_summary_csv(outdir::String, rows::Vector{<:Dict}; filename="summary.csv")
    mkpath(outdir)
    df = DataFrame(rows)
    fpath = joinpath(outdir, filename)
    CSV.write(fpath, df)
    @info "Summary saved to $fpath"
    return fpath
end

function save_samples_csv(outdir::String, θ_chain::AbstractMatrix, param_names::Vector{String};
    prefix::String="", N::Int=0, seed::Int=0)
    mkpath(outdir)
    df = DataFrame(θ_chain, Symbol.(param_names))
    fname = joinpath(outdir, "$(prefix)_N$(N)_seed$(seed)_samples.csv")
    CSV.write(fname, df)
    return fname
end

function save_logdens_csv(outdir::String, logdens::Vector{Float64};
    prefix::String="", N::Int=0, seed::Int=0)
    mkpath(outdir)
    df = DataFrame(iteration=1:length(logdens), logdens=logdens)
    fname = joinpath(outdir, "$(prefix)_N$(N)_seed$(seed)_logdens.csv")
    CSV.write(fname, df)
    return fname
end

# ── CLI Argument Parsing ──────────────────────────────────────────

function add_common_args!(s::ArgParseSettings)
    @add_arg_table! s begin
        "--seed"
            help = "Random seeds (comma-separated)"
            default = join(DEFAULT_SEEDS, ",")
        "--n_iterations"
            help = "Number of MCMC iterations"
            arg_type = Int
            default = MCMC_ITERATIONS
        "--n_warmup"
            help = "Number of warmup iterations"
            arg_type = Int
            default = MCMC_WARMUP
        "--method"
            help = "Methods to run: RFFGM, GPGM, or ALL"
            default = "ALL"
    end
end

function parse_seeds(s::String)
    return parse.(Int, split(s, ","))
end

function parse_methods(s::String)
    s == "ALL" && return [RFFGM, GPGM]
    s == "RFFGM" && return [RFFGM]
    s == "GPGM" && return [GPGM]
    error("Unknown method: $s. Use RFFGM, GPGM, or ALL.")
end

# ── Logging ───────────────────────────────────────────────────────

function log_run(; ode, method, N, seed, extra="")
    ts = Dates.format(now(), "HH:MM:SS")
    @info "[$ts] $ode | $method | N=$N | seed=$seed $extra"
end

function log_metrics(metrics)
    @info @sprintf("  RMSD=%.4f  ESS_mean=%.1f  R̂_max=%.3f  time=%.1fs  ESS/s=%.2f",
        metrics.rmsd, metrics.ess_mean, maximum(metrics.rhat_vals),
        metrics.wall_time, metrics.ess_per_sec)
end

# ── Trajectory Reconstruction (Exp6/Exp8) ────────────────────────

"""
    compute_trajectory_stats(gm::RFFGM, chain, times_dense; n_warmup=0)

Compute posterior mean and 95% CI of trajectories on a dense time grid.
Returns `(mean, lower, upper)` each of size `K × N_dense` (destandardized).
"""
function compute_trajectory_stats(gm::RFFGM, chain, times_dense; n_warmup::Int=0)
    post_chain = chain[n_warmup+1:end]
    W_samples = get_W(gm, post_chain)   # n_samples × K × L
    n_samples, K, _ = size(W_samples)
    N_dense = length(times_dense)

    # Precompute feature matrices on the dense grid for each component
    H_dense = [Hmat(gm.gp[k], times_dense) for k in 1:K]   # each N_dense × L

    # Reconstruct trajectory for each posterior sample
    X_all = Array{Float64}(undef, n_samples, K, N_dense)
    for i in 1:n_samples
        for k in 1:K
            wk = @view W_samples[i, k, :]
            X_all[i, k, :] = H_dense[k] * wk
        end
    end

    # Destandardize
    for k in 1:K
        gpk = gm.gp[k]
        X_all[:, k, :] .= X_all[:, k, :] .* gpk.y_std .+ gpk.y_mean
    end

    X_mean  = dropdims(mean(X_all, dims=1), dims=1)
    X_lower = dropdims(mapslices(x -> quantile(x, 0.025), X_all, dims=1), dims=1)
    X_upper = dropdims(mapslices(x -> quantile(x, 0.975), X_all, dims=1), dims=1)
    return X_mean, X_lower, X_upper
end

"""
    compute_trajectory_stats(gm::GPGM, chain, times_dense; n_warmup=0)

GPGM version: uses GP conditional mean at dense time points.
Returns `(mean, lower, upper)` each of size `K × N_dense` (destandardized).
"""
function compute_trajectory_stats(gm::GPGM, chain, times_dense; n_warmup::Int=0)
    post_chain = chain[n_warmup+1:end]
    X_samples = get_X(gm, post_chain)   # n_samples × K × N_obs
    n_samples, K, N_obs = size(X_samples)
    N_dense = length(times_dense)

    # Precompute GP prediction matrices: K(t*, t) K(t,t)^{-1} for each component
    pred_mats = [kernelmatrix(gm.gp[k].k, times_dense, gm.gp[k].z) * gm.gp[k].K⁻¹
                 for k in 1:K]   # each N_dense × N_obs

    # Predict trajectory at dense grid for each posterior sample
    X_all = Array{Float64}(undef, n_samples, K, N_dense)
    for i in 1:n_samples
        for k in 1:K
            xk = @view X_samples[i, k, :]       # standardized X at obs points
            X_all[i, k, :] = pred_mats[k] * xk  # GP conditional mean
        end
    end

    # Destandardize
    for k in 1:K
        gpk = gm.gp[k]
        X_all[:, k, :] .= X_all[:, k, :] .* gpk.y_std .+ gpk.y_mean
    end

    X_mean  = dropdims(mean(X_all, dims=1), dims=1)
    X_lower = dropdims(mapslices(x -> quantile(x, 0.025), X_all, dims=1), dims=1)
    X_upper = dropdims(mapslices(x -> quantile(x, 0.975), X_all, dims=1), dims=1)
    return X_mean, X_lower, X_upper
end

# ── Per-Parameter RMSD (Exp7) ────────────────────────────────────

"""
    compute_per_param_rmsd(θ_chain, θ_true)

Per-parameter absolute error |E[θ̂_i] - θ*_i|.
"""
function compute_per_param_rmsd(θ_chain::AbstractMatrix, θ_true::Vector{Float64})
    θ_mean = vec(mean(θ_chain, dims=1))
    return abs.(θ_mean .- θ_true)
end

# ── Lorenz96 Configuration (Exp10) ──────────────────────────────

"""
    make_lorenz96_config(K; F_true=8.0, noise_std=0.5, tspan=(0.0, 1.0))

Create an ODEConfig for the K-dimensional Lorenz96 system with single parameter F.
"""
function make_lorenz96_config(K::Int;
    F_true::Float64=8.0,
    noise_std::Float64=0.5,
    tspan::Tuple{Float64,Float64}=(0.0, 1.0))

    rng = MersenneTwister(42)
    u0 = F_true .* ones(K) .+ 0.01 .* randn(rng, K)  # near equilibrium + perturbation

    return ODEConfig(
        "L96_K$(K)", lorenz96!, [F_true], u0, tspan,
        noise_std,
        1.0 * with_lengthscale(SqExponentialKernel(), 1.0),
        ["F"],
        ["x$i" for i in 1:K],
    )
end

# ── Lynx-Hare Real Data (Exp8) ──────────────────────────────────

# Hudson's Bay Company pelt trading records (in thousands), 1900-1920
const LYNXHARE_YEARS = collect(1900.0:1920.0)
const LYNXHARE_HARE  = [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0,
                         25.4, 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6,
                         14.6, 16.2, 24.7]
const LYNXHARE_LYNX  = [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3,
                         9.1, 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8,
                         9.7, 10.1, 8.6]

"""
    make_lynxhare_config(; kernel=SqExponentialKernel())

Create an ODEConfig for the Lynx-Hare LV model with real data.
Time is scaled to [0, 1], data is divided by 100 for numerical stability.
Initial θ guess = [0.5, 0.025, 0.8, 0.025] (typical LV for this data).
"""
function make_lynxhare_config(; kernel::KernelFunctions.Kernel=1.0*with_lengthscale(SqExponentialKernel(), 1.0))
    # Scale time to [0, 1]
    times = (LYNXHARE_YEARS .- 1900.0) ./ 20.0

    # Scale data for numerical stability (divide by 100)
    y_obs = vcat(LYNXHARE_HARE', LYNXHARE_LYNX') ./ 100.0  # 2 × 21

    # Rough initial θ for LV on this scaled data
    θ_init = [0.5, 0.025, 0.8, 0.025]
    u0 = y_obs[:, 1]

    config = ODEConfig(
        "LynxHare", lotkavolterrapredatorprey!, θ_init, u0,
        (times[1], times[end]),
        0.05,   # noise_std (rough estimate for scaled data)
        kernel,
        ["a", "b", "c", "d"],
        ["hare", "lynx"],
    )
    return config, times, y_obs
end
