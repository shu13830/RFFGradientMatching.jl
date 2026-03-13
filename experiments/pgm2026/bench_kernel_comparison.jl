#!/usr/bin/env julia
# Kernel comparison: RFFGM (all kernels) vs GPGM (RBF, Matern52)
# Supports multiple ODE problems
#
# CLI: julia bench_kernel_comparison.jl <method> <kernel_name> <ode> [N_obs] [n_rff]
#   method: "RFFGM" or "GPGM"
#   kernel_name: "RBF", "Matern52", "Laplace", "GenCauchy", "ExpPower"
#   ode: "LV" or "FN"
#   N_obs: number of observations (default 25)
#   n_rff: RFF features (default 50)

include(joinpath(@__DIR__, "common.jl"))
using Printf

method_str = ARGS[1]       # "RFFGM" or "GPGM"
kernel_name = ARGS[2]      # kernel key in KERNEL_CONFIGS
ode_name = length(ARGS) >= 3 ? ARGS[3] : "LV"
N_obs = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 25
n_rff = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 50

method = method_str == "RFFGM" ? RFFGM : GPGM
seed = 42

kconf = KERNEL_CONFIGS[kernel_name]
kernel = kconf.kernel
config = ODE_CONFIGS[ode_name]

# Run experiment using common.jl infrastructure
result = run_single_experiment(method, config;
    N=N_obs, seed=seed, kernel=kernel, n_rff=n_rff,
    n_iterations=10_000, n_warmup=5_000)

metrics = compute_all_metrics(result, config.θ_true)

# CSV output: method,kernel,ode,N,n_rff,rmsd,ess_mean,ess_per_sec,wall_sec,theta_mean...
θ_str = join([@sprintf("%.4f", t) for t in metrics.θ_mean], ",")
@printf("%s,%s,%s,%d,%d,%.4f,%.1f,%.4f,%.1f,%s\n",
    method_str, kernel_name, ode_name, N_obs, n_rff,
    metrics.rmsd, metrics.ess_mean, metrics.ess_per_sec, metrics.wall_time,
    θ_str)
