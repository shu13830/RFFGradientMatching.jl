#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp10: Lorenz96 Preliminary — High-dimensional chaotic system
#
# Purpose: Scaling test on chaotic ODE with K dimensions, |θ|=1 (App.Fig.4)
# Grid:    K ∈ {10, 20, 40} × 5 seeds × {RFFGM, GPGM}
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp10_lorenz96.jl
#   julia --project=. experiments/pgm2026/run_exp10_lorenz96.jl \
#       --K 10 --seed 42 --method RFFGM --n_iterations 100 --n_warmup 50
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp10_args()
    s = ArgParseSettings(description="Exp10: Lorenz96 preliminary scaling test")
    add_common_args!(s)
    @add_arg_table! s begin
        "--K"
            help = "Lorenz96 dimensions (comma-separated)"
            default = "10,20,40"
        "--N"
            help = "Number of observation points"
            arg_type = Int
            default = 50
    end
    args = ArgParse.parse_args(s)
    return (
        K_values     = parse.(Int, split(args["K"], ",")),
        N            = args["N"],
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp10_args()
    outdir = joinpath(RESULTS_BASE, "exp10")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for K in args.K_values
        config = make_lorenz96_config(K)
        ode_key = "L96_K$(K)"

        for seed in args.seeds
            for method in args.methods
                log_run(ode=ode_key, method=nameof(method), N=args.N, seed=seed)

                try
                    result = run_single_experiment(method, config;
                        N=args.N, seed=seed,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup)

                    metrics = compute_all_metrics(result, config.θ_true)
                    log_metrics(metrics)

                    # Compute trajectory RMSD (posterior mean trajectory vs clean data)
                    traj_rmsd = NaN
                    try
                        X_mean, _, _ = compute_trajectory_stats(
                            result.gm, result.chain, result.times;
                            n_warmup=args.n_warmup)
                        traj_rmsd = sqrt(mean((X_mean .- result.y_clean).^2))
                    catch e
                        @warn "Trajectory RMSD computation failed" exception=e
                    end

                    # Save θ posterior samples
                    save_samples_csv(outdir, result.θ_chain, config.param_names;
                        prefix="$(ode_key)_$(nameof(method))", N=args.N, seed=seed)

                    row = make_result_row(metrics;
                        ode_key=ode_key,
                        extra_cols=Dict{String,Any}(
                            "K" => K,
                            "trajectory_rmsd" => traj_rmsd,
                        ))
                    push!(rows, row)

                catch e
                    @warn "Run failed" ode_key method=nameof(method) seed exception=e
                    push!(rows, Dict{String,Any}(
                        "ode"             => ode_key,
                        "method"          => string(nameof(method)),
                        "N"               => args.N,
                        "seed"            => seed,
                        "K"               => K,
                        "rmsd"            => NaN,
                        "ess_mean"        => NaN,
                        "time_sec"        => NaN,
                        "ess_per_sec"     => NaN,
                        "rhat_max"        => NaN,
                        "trajectory_rmsd" => NaN,
                    ))
                end
            end
        end
    end

    save_summary_csv(outdir, rows; filename="exp10_summary.csv")
    @info "Exp10 complete."
end

main()
