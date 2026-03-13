#!/usr/bin/env julia
# N-scaling experiment: RFFGM vs GPGM as N grows
# LV problem, Matern52 kernel, HMC
#
# CLI: julia bench_n_scaling.jl <method> <N_obs> [n_rff] [kernel_name]
#   method: "RFFGM" or "GPGM"
#   N_obs: number of observations
#   n_rff: RFF features (default 50, ignored for GPGM)
#   kernel_name: "RBF" or "Matern52" (default "Matern52")

include(joinpath(@__DIR__, "common.jl"))
using Printf

method_str = ARGS[1]
N_obs = parse(Int, ARGS[2])
n_rff = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
kernel_name = length(ARGS) >= 4 ? ARGS[4] : "Matern52"

method = method_str == "RFFGM" ? RFFGM : GPGM
seed = 42

kconf = KERNEL_CONFIGS[kernel_name]
kernel = kconf.kernel
config = ODE_CONFIGS["LV"]

# Run experiment
result = run_single_experiment(method, config;
    N=N_obs, seed=seed, kernel=kernel, n_rff=n_rff,
    n_iterations=10_000, n_warmup=5_000)

metrics = compute_all_metrics(result, config.θ_true)

# CSV: method,kernel,N,n_rff,rmsd,ess_mean,ess_per_sec,wall_sec,theta1..4
@printf("%s,%s,%d,%d,%.4f,%.1f,%.4f,%.1f,%.4f,%.4f,%.4f,%.4f\n",
    method_str, kernel_name, N_obs, n_rff,
    metrics.rmsd, metrics.ess_mean, metrics.ess_per_sec, metrics.wall_time,
    metrics.θ_mean[1], metrics.θ_mean[2], metrics.θ_mean[3], metrics.θ_mean[4])
