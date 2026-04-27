#!/usr/bin/env julia
# Core experiment: unified script for all methods
# Records convergence history, ODE residual, GP marginal likelihood
#
# CLI: julia bench_core.jl <method> <kernel> <ode> <N> <seed> <theta_id> <gamma> [n_rff]

include(joinpath(@__DIR__, "common.jl"))
using Printf
import RFFGradientMatching: ulogpdf_e, calc_destandardized_X, W2X, eval_ẋ, dfdt_mean, dfdt_cov, get_y_std, calc_θ, get_X, calc_X

const ITER = 20000
const WARMUP = 10000
const ANNEAL = 2500
const LF = 25
const STEP_SIZE = 0.01 / LF
const BAND_TABLE = Dict("LV" => 20, "SIR" => 20, "PST" => 20, "FN" => 20)
const CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000]

method_str = ARGS[1]
kernel_name = ARGS[2]
ode_name = ARGS[3]
N_obs = parse(Int, ARGS[4])
seed = parse(Int, ARGS[5])
theta_id = parse(Int, ARGS[6])
gamma_val = parse(Float64, ARGS[7])
n_rff = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 100

method = Dict("RFFGM"=>RFFGM, "GPGM"=>GPGM, "MAGI"=>MAGI)[method_str]
kconf = KERNEL_CONFIGS[kernel_name]
kernel = kconf.kernel
config = ODE_CONFIGS[ode_name]
θ_true = get_θ_true(config, theta_id)

config_with_θ = ODEConfig(config.name, config.f!, θ_true, config.θ_patterns,
    config.u0, config.tspan, config.noise_std, config.kernel,
    config.param_names, config.component_names)

Random.seed!(seed)
times, y_obs, y_clean, prob = generate_data(config_with_θ; N=N_obs, seed=seed)

_anneal = method == MAGI ? 1 : ANNEAL
gm = setup_model(method, config_with_θ, times, y_obs, prob;
    kernel=kernel, n_rff=n_rff, anneal_length=_anneal)

# GP marginal likelihood (before γ change)
gp_marglik = try
    sum(logpdf(gpk.fz, gpk.u) for gpk in gm.gp)
catch
    NaN
end

# Set γ and band approximation (MAGI only; GPGM uses dense like FGPGM original)
gm.odegrad.γ = gamma_val
if method == MAGI
    set_bandsize!(gm, get(BAND_TABLE, ode_name, 20))
end
cache_e_cov_chol!(gm)

# σ sampling for MAGI only
if method == MAGI
    K = length(gm.gp)
    set_priortransform_on_σ!(gm, fill(Normal(0.0, 10.0), K), fill(log, K))
end

# Sampler
latent_sym = method == RFFGM ? :W : :X
blocks = [
    HMCBlock(gm, [latent_sym]; n_leapfrog=LF, step_size=STEP_SIZE, metric=:diag),
    HMCBlock(gm, [:θ]; n_leapfrog=LF, step_size=STEP_SIZE, metric=:diag),
]
if method == MAGI
    push!(blocks, HMCBlock(gm, [:σ]; n_leapfrog=LF, step_size=STEP_SIZE, metric=:diag))
end
bs = BlockedSampler([blocks], [1.0])

t_start = time()
chain, logdens = AbstractMCMC.sample(gm, bs, ITER;
    num_burnin=WARMUP, anneal=true)
wall_time = time() - t_start

# Convergence history
conv_records = []
for cp in CHECKPOINTS
    cp > ITER && break
    win = min(1000, cp)
    sub = chain[max(1, cp-win+1):cp]
    θ_sub = get_θ(gm, sub)
    θ_mean_cp = vec(mean(θ_sub, dims=1))
    rmsd_cp = sqrt(mean((θ_mean_cp .- θ_true).^2))
    push!(conv_records, (cp, rmsd_cp))
end

# Post-warmup metrics
post_chain = chain[WARMUP+1:end]
θ_chain = get_θ(gm, post_chain)

# ODE residual
resid_vals = Float64[]
for c in post_chain[1:100:end]
    if method == RFFGM
        W = c[:W]; X = W2X(gm.gp, W)
        θ = calc_θ(gm.odegrad, c[:θ])
        X_ds = calc_destandardized_X(gm.gp, X)
        y_std = get_y_std(gm.gp)
        ẋode = eval_ẋ(gm.odegrad, X_ds, θ) ./ y_std
        ẋgp = dfdt_mean(gm.gp, W)
    else
        X = calc_X(gm.gp, c[:X])
        θ = calc_θ(gm.odegrad, c[:θ])
        X_ds = calc_destandardized_X(gm.gp, X)
        y_std = get_y_std(gm.gp)
        ẋode = eval_ẋ(gm.odegrad, X_ds, θ) ./ y_std
        ẋgp = dfdt_mean(gm.gp, X)
    end
    resid = mean(sum((ẋode[k,:] .- ẋgp[k]).^2 for k in 1:size(ẋode,1)))
    push!(resid_vals, resid)
end
resid_mean = mean(resid_vals)

result = (; θ_chain, chain=post_chain, logdens, gm, wall_time, times, y_obs, y_clean,
    config=config_with_θ, N=N_obs, seed, method=string(nameof(method)))
metrics = compute_all_metrics(result, θ_true)
coverage = compute_coverage(result.θ_chain, θ_true)
traj_rmse = compute_trajectory_rmse(result.gm, result.chain, config_with_θ; n_warmup=0)

θ_mean_str = join([@sprintf("%.4f", t) for t in metrics.θ_mean], ";")
θ_std_str = join([@sprintf("%.4f", t) for t in metrics.θ_std], ";")
conv_str = join([@sprintf("%d:%.4f", cp, r) for (cp, r) in conv_records], ";")

@printf("%s,%s,%s,%d,%d,%d,%.6f,%.4f,%.1f,%.4f,%.4f,%.1f,%.4f,%.4f,%.6f,%.4f,%s,%s,%s\n",
    method_str, kernel_name, ode_name, N_obs, seed, theta_id, gamma_val,
    metrics.rmsd, metrics.ess_mean, metrics.ess_per_sec,
    maximum(metrics.rhat_vals), metrics.wall_time,
    coverage, traj_rmse, resid_mean, gp_marglik,
    θ_mean_str, θ_std_str, conv_str)
