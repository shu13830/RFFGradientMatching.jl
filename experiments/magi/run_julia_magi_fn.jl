#!/usr/bin/env julia
# Julia MAGI on FitzHugh-Nagumo — reproduce R MAGI paper results
#
# R MAGI paper settings:
#   θ = (a, b, c) = (0.2, 0.2, 3.0)  [our ordering: (c, a, b) = (3.0, 0.2, 0.2)]
#   tspan = [0, 20], N = 41, σ = 0.2
#   kernel = generalMatern
#   20,000 HMC iterations, 200 leapfrog steps, 50% burn-in
#   discretization level=1 (midpoints inserted)
#   σ sampled, no γ (manifold constraint)
#
# Usage:
#   julia --project=. experiments/magi/run_julia_magi_fn.jl [seed]

using RFFGradientMatching
import RFFGradientMatching: fitzhughnagumo!
using DifferentialEquations
using KernelFunctions
using Distributions
using Random
using AbstractMCMC
using Statistics
using Printf

seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 42

# ── Generate FN data (matching MAGI paper) ──
θ_true = [3.0, 0.2, 0.2]  # (c, a, b) in our ordering
u0 = [-1.0, 1.0]
tspan = (0.0, 20.0)
σ_obs = 0.2

prob = ODEProblem(fitzhughnagumo!, u0, tspan, θ_true)

# N=41 evenly spaced observations
N_obs = 41
times = collect(range(tspan[1], tspan[2], length=N_obs))

# Solve ODE and add noise
Random.seed!(seed)
sol = solve(prob, Tsit5(); saveat=times)
y_clean = reduce(hcat, sol.u)'  # N × K → need K × N
y_obs = y_clean .+ σ_obs .* randn(size(y_clean))
y_obs = Matrix{Float64}(y_obs')  # K × N

@info "Data generated" N=N_obs tspan seed σ=σ_obs

# ── MAGI model (no discretization, N=41 directly) ──
k = with_lengthscale(Matern52Kernel(), 1.0)
gm = MAGI(times, y_obs, prob, "FN";
    k=k,
    state_noise_std=1e-3,
    obs_noise_std=σ_obs,
    anneal_length=1,       # β=1 fixed (MAGI default)
    γ_init=1e-3            # small jitter (R MAGI uses 1e-7)
)

# Prior on θ: log-transform, weakly informative
n_θ = length(θ_true)
set_priortransform_on_θ!(gm, fill(Normal(0.0, 10.0), n_θ), fill(log, n_θ))

# σ sampling with weakly informative prior
K = length(gm.gp)
set_priortransform_on_σ!(gm, fill(Normal(0.0, 10.0), K), fill(log, K))

# ── Optimize ──
@info "Optimizing GP hyperparameters..."
optimize_ϕ_and_σ!(gm)
@info "Optimizing initial trajectory..."
optimize_u!(gm)

# ── Sampler: Joint HMC (X + θ + σ) ──
block_all = HMCBlock(gm, [:X, :θ, :σ]; n_leapfrog=200, step_size=5e-5, metric=:diag)
bs = BlockedSampler([[block_all]], [1.0])

# ── MCMC ──
n_iter   = 20_000
n_burnin = 10_000

@info "Starting MCMC" n_iter n_burnin seed
Random.seed!(seed)
t_start = time()
chain, logdens = AbstractMCMC.sample(gm, bs, n_iter;
    num_burnin=n_burnin, anneal=true)
t_elapsed = time() - t_start

# ── Extract results (post burn-in) ──
post_chain = chain[n_burnin+1:end]
θ_chain = get_θ(gm, post_chain)
σ_chain = get_σ(gm, post_chain)

param_names = ["c", "a", "b"]
θ_mean = vec(mean(θ_chain, dims=1))
θ_std  = vec(std(θ_chain, dims=1))
θ_q025 = [quantile(θ_chain[:, i], 0.025) for i in 1:n_θ]
θ_q975 = [quantile(θ_chain[:, i], 0.975) for i in 1:n_θ]

σ_mean = vec(mean(σ_chain, dims=1))

# ── Print summary ──
println()
println("═" ^ 60)
@printf("Julia MAGI FN Results (N=%d, seed=%d)\n", N_obs, seed)
println("═" ^ 60)
@printf("Time: %.1f sec\n", t_elapsed)
@printf("Post burn-in samples: %d\n", length(post_chain))
@printf("Inducing points: %d\n", N_obs)
@printf("σ_mean: V=%.4f, R=%.4f (true=%.2f)\n", σ_mean[1], σ_mean[2], σ_obs)
println()
@printf("%-6s  %6s  %12s  %14s\n", "param", "true", "mean±sd", "95% CI")
println("-" ^ 60)
for i in 1:n_θ
    @printf("%-6s  %6.2f  %5.3f±%.3f  [%5.3f, %5.3f]\n",
        param_names[i], θ_true[i],
        θ_mean[i], θ_std[i],
        θ_q025[i], θ_q975[i])
end
rmsd = sqrt(mean((θ_mean .- θ_true).^2))
@printf("\nRMSD(θ): %.4f\n", rmsd)
println("═" ^ 60)
