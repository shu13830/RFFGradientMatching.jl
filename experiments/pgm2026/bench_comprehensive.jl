#!/usr/bin/env julia
# Comprehensive RFFGM vs GPGM vs MAGI comparison
# Supports all ODE problems, kernels, N values, L values, and seeds
#
# CLI: julia bench_comprehensive.jl <method> <kernel_name> <ode> <N_obs> <n_rff> <seed>
#   method: "RFFGM", "GPGM", or "MAGI"
#   kernel_name: "RBF", "Matern52", "Laplace", "GenCauchy", "ExpPower", "Sigmoid"
#   ode: "LV", "FN", or "PST"
#   N_obs: number of observations
#   n_rff: RFF features (0 for GPGM/MAGI)
#   seed: random seed

include(joinpath(@__DIR__, "common.jl"))
using Printf

method_str = ARGS[1]
kernel_name = ARGS[2]
ode_name = ARGS[3]
N_obs = parse(Int, ARGS[4])
n_rff = parse(Int, ARGS[5])
seed = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 42

method = if method_str == "RFFGM"
    RFFGM
elseif method_str == "GPGM"
    GPGM
elseif method_str == "MAGI"
    MAGI
else
    error("Unknown method: $method_str")
end

# Kernel selection
if kernel_name == "Sigmoid"
    kernel = 1.0 * SigmoidKernel(1.0, 0.0)
else
    kconf = KERNEL_CONFIGS[kernel_name]
    kernel = kconf.kernel
end

config = ODE_CONFIGS[ode_name]

# Run experiment
result = run_single_experiment(method, config;
    N=N_obs, seed=seed, kernel=kernel, n_rff=n_rff,
    n_iterations=10_000, n_warmup=5_000)

metrics = compute_all_metrics(result, config.θ_true)

# Coverage
coverage = compute_coverage(result.θ_chain, config.θ_true)

# Trajectory RMSE (skip warmup samples)
traj_rmse = compute_trajectory_rmse(result.gm, result.chain, config; n_warmup=5_000)

# CSV output: method,kernel,ode,N,n_rff,seed,rmsd,ess_mean,ess_per_sec,rhat_max,wall_sec,coverage,traj_rmse,theta_mean,theta_std
θ_mean_str = join([@sprintf("%.4f", t) for t in metrics.θ_mean], ";")
θ_std_str = join([@sprintf("%.4f", t) for t in metrics.θ_std], ";")
@printf("%s,%s,%s,%d,%d,%d,%.4f,%.1f,%.4f,%.4f,%.1f,%.4f,%.4f,%s,%s\n",
    method_str, kernel_name, ode_name, N_obs, n_rff, seed,
    metrics.rmsd, metrics.ess_mean, metrics.ess_per_sec,
    maximum(metrics.rhat_vals), metrics.wall_time,
    coverage, traj_rmse,
    θ_mean_str, θ_std_str)
