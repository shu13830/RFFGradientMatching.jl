#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp8: Real Data — Lynx-Hare (Hudson's Bay Company, 1900-1920)
#
# Purpose: Demonstrate inference on real data (App.Fig.2 + App.Tab.3)
# Settings: 4 configs: GPGM-RBF, RFFGM-RBF, RFFGM-Laplace, RFFGM-GenCauchy
# No true θ → RMSD is NaN; evaluate ESS, R̂, and trajectory visual
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp8_realdata.jl
#   julia --project=. experiments/pgm2026/run_exp8_realdata.jl \
#       --seed 42 --kernel RBF --method RFFGM --n_iterations 100 --n_warmup 50
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── Exp8 kernel configurations ────────────────────────────────────

const EXP8_KERNEL_CONFIGS = Dict(
    "RBF" => (
        kernel = 1.0 * with_lengthscale(SqExponentialKernel(), 1.0),
        methods = [RFFGM, GPGM],
    ),
    "Laplace" => (
        kernel = 1.0 * with_lengthscale(ExponentialPowerKernel(γ=1.0), 1.0),
        methods = [RFFGM],
    ),
    "GenCauchy" => (
        kernel = 1.0 * with_lengthscale(GeneralizedCauchyKernel(1.5, 1.5), 1.0),
        methods = [RFFGM],
    ),
)
const EXP8_KERNEL_ORDER = ["RBF", "Laplace", "GenCauchy"]

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp8_args()
    s = ArgParseSettings(description="Exp8: Lynx-Hare real data experiment")
    add_common_args!(s)
    @add_arg_table! s begin
        "--kernel"
            help = "Kernels (comma-separated, or ALL)"
            default = "ALL"
        "--n_dense"
            help = "Dense grid points for trajectory"
            arg_type = Int
            default = 200
    end
    args = ArgParse.parse_args(s)
    kernel_str = args["kernel"]
    kernel_keys = kernel_str == "ALL" ? EXP8_KERNEL_ORDER : String.(split(kernel_str, ","))
    return (
        kernel_keys  = kernel_keys,
        n_dense      = args["n_dense"],
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# ── Lynx-Hare model setup (bypasses generate_data) ───────────────

function setup_lynxhare_model(method::Type{<:Union{RFFGM,GPGM}}, config, times, y_obs;
    kernel=config.kernel, n_rff=DEFAULT_N_RFF, anneal_length=ANNEAL_LENGTH)

    prob = ODEProblem(config.f!, config.u0, config.tspan, config.θ_true)

    if method === RFFGM
        gm = RFFGM(times, y_obs, prob, config.name;
            k=kernel, state_noise_std=STATE_NOISE_STD, obs_noise_std=config.noise_std,
            n_rff=n_rff, anneal_length=anneal_length)
    else
        gm = GPGM(times, y_obs, prob, config.name;
            k=kernel, state_noise_std=STATE_NOISE_STD, obs_noise_std=config.noise_std,
            anneal_length=anneal_length)
    end

    # Wider priors for real data (no ground truth)
    n_θ = length(prob.p)
    set_priortransform_on_θ!(gm, fill(Normal(0.0, 2.0), n_θ), fill(log, n_θ))
    optimize_ϕ_and_σ!(gm)
    optimize_u!(gm)
    return gm, prob
end

function run_lynxhare_experiment(method::Type{<:Union{RFFGM,GPGM}}, config, times, y_obs;
    seed::Int, kernel=config.kernel,
    n_rff::Int=DEFAULT_N_RFF,
    n_iterations::Int=MCMC_ITERATIONS,
    n_warmup::Int=MCMC_WARMUP,
    anneal_length::Int=ANNEAL_LENGTH)

    Random.seed!(seed)
    _anneal = min(anneal_length, max(n_warmup - 1, 0))
    gm, prob = setup_lynxhare_model(method, config, times, y_obs;
        kernel=kernel, n_rff=n_rff, anneal_length=_anneal)
    bs = create_blocked_sampler(gm)

    t_start = time()
    chain, logdens = AbstractMCMC.sample(gm, bs, n_iterations;
        num_burnin=n_warmup, anneal=true)
    wall_time = time() - t_start

    θ_chain = get_θ(gm, chain[n_warmup+1:end])

    return (;
        θ_chain, chain, logdens, gm,
        wall_time, times, y_obs,
        config, seed, method=string(nameof(method)),
    )
end

# ── Metrics (no true θ) ──────────────────────────────────────────

function compute_realdata_metrics(result)
    θ_chain = result.θ_chain
    ess_vals = compute_ess(θ_chain)
    ess_mean = mean(ess_vals)
    ess_per_sec = ess_mean / result.wall_time
    θ_mean = vec(mean(θ_chain, dims=1))
    θ_std = vec(std(θ_chain, dims=1))
    rhat_vals = compute_rhat(θ_chain)

    return (;
        rmsd=NaN, ess_vals, ess_mean, ess_per_sec,
        θ_mean, θ_std, rhat_vals,
        wall_time=result.wall_time,
        N=length(result.times), seed=result.seed, method=result.method,
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp8_args()
    outdir = joinpath(RESULTS_BASE, "exp8")
    mkpath(outdir)

    # Load Lynx-Hare data
    config, times, y_obs = make_lynxhare_config()
    N = length(times)

    # Save raw data for reference
    data_df = DataFrame(
        t=times,
        year=LYNXHARE_YEARS,
        hare=LYNXHARE_HARE ./ 100.0,
        lynx=LYNXHARE_LYNX ./ 100.0,
    )
    CSV.write(joinpath(outdir, "exp8_data.csv"), data_df)

    summary_rows = Dict{String,Any}[]

    for kname in args.kernel_keys
        kconfig = EXP8_KERNEL_CONFIGS[kname]
        for seed in args.seeds
            for method in args.methods
                # Skip methods that don't support this kernel
                if !(method in kconfig.methods)
                    @info "Skipping $(kname) for $(nameof(method)) (unsupported)"
                    continue
                end

                log_run(ode="LynxHare", method=nameof(method), N=N, seed=seed,
                    extra="kernel=$kname")

                try
                    result = run_lynxhare_experiment(method, config, times, y_obs;
                        seed=seed,
                        kernel=kconfig.kernel,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup)

                    metrics = compute_realdata_metrics(result)

                    @info @sprintf("  ESS_mean=%.1f  R̂_max=%.3f  time=%.1fs  ESS/s=%.2f",
                        metrics.ess_mean, maximum(metrics.rhat_vals),
                        metrics.wall_time, metrics.ess_per_sec)

                    # Save θ posterior samples
                    save_samples_csv(outdir, result.θ_chain, config.param_names;
                        prefix="LynxHare_$(nameof(method))_$(kname)", N=N, seed=seed)

                    # Summary row
                    push!(summary_rows, make_result_row(metrics;
                        ode_key="LynxHare",
                        extra_cols=Dict{String,Any}("kernel" => kname)))

                    # Trajectory output for representative seed (first seed only)
                    if seed == args.seeds[1]
                        times_dense = collect(range(times[1], times[end], length=args.n_dense))
                        X_mean, X_lower, X_upper = compute_trajectory_stats(
                            result.gm, result.chain, times_dense;
                            n_warmup=args.n_warmup)

                        mname = string(nameof(method))
                        K_comp = size(X_mean, 1)
                        traj_df = DataFrame(t=times_dense)
                        for k in 1:K_comp
                            cname = config.component_names[k]
                            traj_df[!, "mean_$(cname)"]  = X_mean[k, :]
                            traj_df[!, "lower_$(cname)"] = X_lower[k, :]
                            traj_df[!, "upper_$(cname)"] = X_upper[k, :]
                        end
                        traj_fname = joinpath(outdir,
                            "exp8_trajectory_$(mname)_$(kname).csv")
                        CSV.write(traj_fname, traj_df)
                        @info "Trajectory saved to $traj_fname"
                    end
                catch e
                    @warn "Run failed" kname method=nameof(method) seed exception=e
                    push!(summary_rows, Dict{String,Any}(
                        "ode"         => "LynxHare",
                        "method"      => string(nameof(method)),
                        "N"           => N,
                        "seed"        => seed,
                        "kernel"      => kname,
                        "rmsd"        => NaN,
                        "ess_mean"    => NaN,
                        "time_sec"    => NaN,
                        "ess_per_sec" => NaN,
                        "rhat_max"    => NaN,
                    ))
                end
            end
        end
    end

    save_summary_csv(outdir, summary_rows; filename="exp8_summary.csv")
    @info "Exp8 complete."
end

main()
