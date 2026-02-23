#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp6: Trajectory Recovery Visualisation (LV)
#
# Purpose: Visualise posterior trajectory estimates (App.Fig.1, App.Tab.1)
# Grid:    LV, N=25, seed=42 × {RFFGM, GPGM}
#          MAGI trajectory loaded from baselines/magi/results/ if available
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp6_trajectory.jl
#   julia --project=. experiments/pgm2026/run_exp6_trajectory.jl \
#       --seed 42 --method ALL --n_iterations 100 --n_warmup 50
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp6_args()
    s = ArgParseSettings(description="Exp6: Trajectory recovery (LV)")
    add_common_args!(s)
    @add_arg_table! s begin
        "--N"
            help = "Number of observation points"
            arg_type = Int
            default = 25
        "--n_dense"
            help = "Number of dense grid points for trajectory"
            arg_type = Int
            default = 200
    end
    args = ArgParse.parse_args(s)
    return (
        N            = args["N"],
        n_dense      = args["n_dense"],
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# ── Trajectory CSV I/O ────────────────────────────────────────────

function save_trajectory_csv(outdir::String, times_dense, X_mean, X_lower, X_upper,
    component_names::Vector{String}; prefix::String="")
    mkpath(outdir)
    K, N_dense = size(X_mean)
    df = DataFrame(t=times_dense)
    for k in 1:K
        cname = component_names[k]
        df[!, "mean_$(cname)"]  = X_mean[k, :]
        df[!, "lower_$(cname)"] = X_lower[k, :]
        df[!, "upper_$(cname)"] = X_upper[k, :]
    end
    fname = joinpath(outdir, "$(prefix)_trajectory.csv")
    CSV.write(fname, df)
    @info "Trajectory saved to $fname"
    return fname
end

function save_observations_csv(outdir::String, times, y_obs, y_clean,
    component_names::Vector{String}; prefix::String="exp6")
    mkpath(outdir)
    K, N = size(y_obs)
    df = DataFrame(t=times)
    for k in 1:K
        cname = component_names[k]
        df[!, "obs_$(cname)"]   = y_obs[k, :]
        df[!, "clean_$(cname)"] = y_clean[k, :]
    end
    fname = joinpath(outdir, "$(prefix)_observations.csv")
    CSV.write(fname, df)
    @info "Observations saved to $fname"
    return fname
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp6_args()
    config = ODE_CONFIGS["LV"]
    outdir = joinpath(RESULTS_BASE, "exp6")
    mkpath(outdir)

    times_dense = collect(range(config.tspan[1], config.tspan[2], length=args.n_dense))
    summary_rows = Dict{String,Any}[]

    for seed in args.seeds
        # Generate data (same for all methods within this seed)
        Random.seed!(seed)
        times, y_obs, y_clean, _ = generate_data(config; N=args.N, seed=seed)

        # Save observation data
        save_observations_csv(outdir, times, y_obs, y_clean, config.component_names;
            prefix="exp6_N$(args.N)_seed$(seed)")

        for method in args.methods
            log_run(ode="LV", method=nameof(method), N=args.N, seed=seed,
                extra="(trajectory)")

            result = run_single_experiment(method, config;
                N=args.N, seed=seed,
                n_iterations=args.n_iterations,
                n_warmup=args.n_warmup)

            metrics = compute_all_metrics(result, config.θ_true)
            log_metrics(metrics)

            # Compute trajectory statistics on dense grid
            X_mean, X_lower, X_upper = compute_trajectory_stats(
                result.gm, result.chain, times_dense;
                n_warmup=args.n_warmup)

            # Save trajectory CSV
            mname = string(nameof(method))
            save_trajectory_csv(outdir, times_dense, X_mean, X_lower, X_upper,
                config.component_names;
                prefix="exp6_$(mname)_N$(args.N)_seed$(seed)")

            # Save θ posterior samples
            save_samples_csv(outdir, result.θ_chain, config.param_names;
                prefix="LV_$(mname)", N=args.N, seed=seed)

            push!(summary_rows, make_result_row(metrics; ode_key="LV"))
        end
    end

    # Try loading MAGI trajectory (generated separately by R script)
    magi_traj_file = joinpath(@__DIR__, "..", "..", "baselines", "magi", "results",
        "magi_lv_trajectory.csv")
    if isfile(magi_traj_file)
        @info "MAGI trajectory found: $magi_traj_file"
        cp(magi_traj_file, joinpath(outdir, "exp6_MAGI_trajectory.csv"); force=true)
    else
        @warn "MAGI trajectory not found at $magi_traj_file — run baselines/magi/run_lv.R --trajectory first"
    end

    save_summary_csv(outdir, summary_rows; filename="exp6_summary.csv")
    @info "Exp6 complete."
end

main()
