#!/usr/bin/env julia
# Julia MAGI experiment for comparison with R MAGI
# Runs on LV (Lotka-Volterra) with N=10, seed=42
#
# Aligned with R MAGI settings:
#   1. Discretization: midpoints between observations (10 → 19 inducing points)
#   2. Matern52 kernel (closer to R's generalMatern)
#   3. γ ≈ 0 (R MAGI has no explicit γ; ODE constraint via Cov[dX/dt|X] only)
#   4. σ sampling via :σ block (R estimates obs noise jointly)
#
# Usage:
#   julia --project=. experiments/magi/run_julia_magi.jl

using RFFGradientMatching
import RFFGradientMatching: lotkavolterrapredatorprey!
using DifferentialEquations
using KernelFunctions
using Distributions
using Random
using AbstractMCMC
using CSV
using DataFrames
using Statistics
using Printf

# ── Paths ─────────────────────────────────────────────────────────────────
project_root = dirname(dirname(@__DIR__))
data_file    = joinpath(project_root, "baselines", "data", "lv_N10_seed42.csv")
results_dir  = joinpath(@__DIR__, "results")
mkpath(results_dir)

# ── Load data ─────────────────────────────────────────────────────────────
dat = CSV.read(data_file, DataFrame)
times = Vector{Float64}(dat.time)
y_obs = Matrix{Float64}(hcat(dat.prey, dat.predator)')  # 2 x N
N = length(times)
@info "Data loaded" N

# ── [1] Discretization: add midpoints (like R's setDiscretization level=1) ─
midpoints = [(times[i] + times[i+1]) / 2 for i in 1:length(times)-1]
dense_times = sort(vcat(times, midpoints))
@info "Discretization" N_obs=N N_inducing=length(dense_times)

# ── MAGI model ────────────────────────────────────────────────────────────
θ_true = [2.0, 1.0, 4.0, 1.0]
prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), θ_true)

# [2] Matern52 kernel (closer to R's generalMatern)
k = with_lengthscale(Matern52Kernel(), 1.0)
gm = MAGI(times, y_obs, prob, "LV";
    k=k,
    state_noise_std=1e-3,
    obs_noise_std=0.5,
    inducing_points=dense_times,
    anneal_length=1000,
    γ_init=0.1
)

# Prior: θ > 0 via log transform, Normal(0,1) in transformed space
set_priortransform_on_θ!(gm, fill(Normal(0.0, 1.0), 4), fill(log, 4))

# γ sampled (log-transformed)
set_priortransform_on_γ!(gm, Normal(0.0, 1.0), log)

# σ sampling (R MAGI estimates σ jointly)
set_priortransform_on_σ!(gm, fill(Normal(0.0, 1.0), 2), fill(log, 2))

# ── Optimize GP hyperparameters and initial trajectory ────────────────────
@info "Optimizing GP hyperparameters..."
optimize_ϕ_and_σ!(gm)
@info "Optimizing initial trajectory..."
optimize_u!(gm)

# ── Sampler ───────────────────────────────────────────────────────────────
# Single block: X + θ + γ + σ jointly, 45 dims total
block_all = HMCBlock(gm, [:X, :θ, :γ, :σ]; n_leapfrog=10, step_size=0.01, metric=:diag)
bs = BlockedSampler([[block_all]], [1.0])

# ── MCMC sampling ─────────────────────────────────────────────────────────
n_iter   = 10_000
n_burnin = 5_000

@info "Starting MCMC" n_iter n_burnin anneal_length=gm.anneal_length
Random.seed!(42)
t_start = time()
chain, logdens = AbstractMCMC.sample(gm, bs, n_iter;
    num_burnin=n_burnin, anneal=true)
t_elapsed = time() - t_start

# ── Extract results ───────────────────────────────────────────────────────
θ_chain = get_θ(gm, chain)      # n_iter x 4
σ_chain = get_σ(gm, chain)      # n_iter x 2
X_chain = get_X(gm, chain)      # n_iter x n_components x N_inducing

param_names = ["a", "b", "c", "d"]
θ_mean = vec(mean(θ_chain, dims=1))
θ_std  = vec(std(θ_chain, dims=1))
θ_q025 = [quantile(θ_chain[:, i], 0.025) for i in 1:4]
θ_q975 = [quantile(θ_chain[:, i], 0.975) for i in 1:4]

# ── Save θ samples ────────────────────────────────────────────────────────
CSV.write(joinpath(results_dir, "julia_magi_theta_samples.csv"),
    DataFrame(θ_chain, param_names))

# ── Save σ samples ────────────────────────────────────────────────────────
CSV.write(joinpath(results_dir, "julia_magi_sigma_samples.csv"),
    DataFrame(σ_chain, ["sigma_prey", "sigma_predator"]))

# ── Save trajectory statistics ────────────────────────────────────────────
# X_chain: (n_samples, n_components, N_inducing) — 3D array
# Extract only observation time points for comparison with R
N_ind = size(X_chain, 3)
obs_idx = [findfirst(t -> isapprox(t, ti; atol=1e-10), dense_times) for ti in times]

X_prey = X_chain[:, 1, obs_idx]   # n_iter x N_obs
X_pred = X_chain[:, 2, obs_idx]   # n_iter x N_obs

traj_mean_prey  = vec(mean(X_prey, dims=1))
traj_lower_prey = [quantile(X_prey[:, j], 0.025) for j in 1:N]
traj_upper_prey = [quantile(X_prey[:, j], 0.975) for j in 1:N]

traj_mean_pred  = vec(mean(X_pred, dims=1))
traj_lower_pred = [quantile(X_pred[:, j], 0.025) for j in 1:N]
traj_upper_pred = [quantile(X_pred[:, j], 0.975) for j in 1:N]

traj_df = DataFrame(
    t = times,
    mean_prey      = traj_mean_prey,
    lower_prey     = traj_lower_prey,
    upper_prey     = traj_upper_prey,
    mean_predator  = traj_mean_pred,
    lower_predator = traj_lower_pred,
    upper_predator = traj_upper_pred
)
CSV.write(joinpath(results_dir, "julia_magi_trajectory.csv"), traj_df)

# ── Print summary ─────────────────────────────────────────────────────────
println()
println("═" ^ 55)
println("Julia MAGI Results (LV, N=10, seed=42)")
println("═" ^ 55)
@printf("Time: %.1f sec\n", t_elapsed)
@printf("Post burn-in samples: %d\n", n_iter)
@printf("Inducing points: %d (obs: %d + midpoints: %d)\n",
    length(dense_times), N, length(midpoints))
println()
@printf("%-6s  %6s  %12s  %14s  \n", "param", "true", "mean±sd", "95% CI")
println("-" ^ 55)
for i in 1:4
    @printf("%-6s  %6.2f  %5.3f±%.3f  [%5.3f, %5.3f]\n",
        param_names[i], θ_true[i],
        θ_mean[i], θ_std[i],
        θ_q025[i], θ_q975[i])
end
rmsd = sqrt(mean((θ_mean .- θ_true).^2))
@printf("\nRMSD(θ): %.4f\n", rmsd)
println("═" ^ 55)
