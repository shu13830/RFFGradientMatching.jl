#!/usr/bin/env julia
# γ selection experiment: RFFGM RBF, multiple γ values
# Records ulogpdf_e posterior mean for γ selection criterion
#
# CLI: julia bench_gamma_select.jl <ode> <N> <seed> <theta_id> <gamma>

include(joinpath(@__DIR__, "common.jl"))
using Printf
import RFFGradientMatching: ulogpdf_e, calc_destandardized_X, W2X, eval_ẋ, dfdt_mean, dfdt_cov, get_y_std, calc_θ, get_W

const ITER = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20000
const WARMUP = div(ITER, 2)
const ANNEAL = div(WARMUP, 4)
const LF = 25
const STEP_SIZE = 0.01 / LF
const L = 100

ode_name = ARGS[1]
N_obs = parse(Int, ARGS[2])
seed = parse(Int, ARGS[3])
theta_id = parse(Int, ARGS[4])
gamma_val = parse(Float64, ARGS[5])

config = ODE_CONFIGS[ode_name]
θ_true = get_θ_true(config, theta_id)
kernel = with_lengthscale(SqExponentialKernel(), 1.0)

config_with_θ = ODEConfig(config.name, config.f!, θ_true, config.θ_patterns,
    config.u0, config.tspan, config.noise_std, config.kernel,
    config.param_names, config.component_names)

Random.seed!(seed)
times, y_obs, y_clean, prob = generate_data(config_with_θ; N=N_obs, seed=seed)

gm = setup_model(RFFGM, config_with_θ, times, y_obs, prob;
    kernel=kernel, n_rff=L, anneal_length=ANNEAL)

gm.odegrad.γ = gamma_val
cache_e_cov_chol!(gm)

b1 = HMCBlock(gm, [:W]; n_leapfrog=LF, step_size=STEP_SIZE, metric=:diag)
b2 = HMCBlock(gm, [:θ]; n_leapfrog=LF, step_size=STEP_SIZE, metric=:diag)
bs = BlockedSampler([[b1, b2]], [1.0])

t_start = time()
chain, logdens = AbstractMCMC.sample(gm, bs, ITER;
    num_burnin=WARMUP, anneal=true)
wall_time = time() - t_start

post_chain = chain[WARMUP+1:end]
θ_chain = get_θ(gm, post_chain)

# Compute ODE residual (γ-independent) and ulogpdf_e
resid_vals = Float64[]
ule_vals = Float64[]
for c in post_chain[1:100:end]  # subsample every 100th for speed
    W = c[:W]
    X = W2X(gm.gp, W)
    θ = calc_θ(gm.odegrad, c[:θ])
    X_ds = calc_destandardized_X(gm.gp, X)
    y_std = get_y_std(gm.gp)
    ẋode = eval_ẋ(gm.odegrad, X_ds, θ) ./ y_std
    ẋgp = dfdt_mean(gm.gp, W)
    # ODE residual: mean ||f(x,θ) - x'||²
    resid = mean(sum((ẋode[k,:] .- ẋgp[k]).^2 for k in 1:size(ẋode,1)))
    push!(resid_vals, resid)
    ule = ulogpdf_e(ẋode, ẋgp, dfdt_cov(gm.gp), gamma_val)
    push!(ule_vals, ule)
end
resid_mean = mean(resid_vals)
ule_mean = mean(ule_vals)

result = (; θ_chain, chain=post_chain, logdens, gm, wall_time, times, y_obs, y_clean,
    config=config_with_θ, N=N_obs, seed, method="RFFGM")
metrics = compute_all_metrics(result, θ_true)
coverage = compute_coverage(result.θ_chain, θ_true)

θ_mean_str = join([@sprintf("%.4f", t) for t in metrics.θ_mean], ";")
@printf("%s,%d,%d,%d,%.6f,%.4f,%.1f,%.4f,%.4f,%.1f,%.4f,%.6f,%.2f,%s\n",
    ode_name, N_obs, seed, theta_id, gamma_val,
    metrics.rmsd, metrics.ess_mean, metrics.ess_per_sec,
    maximum(metrics.rhat_vals), metrics.wall_time,
    coverage, resid_mean, ule_mean, θ_mean_str)
