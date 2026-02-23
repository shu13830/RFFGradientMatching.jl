#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp7: PST Per-Parameter RMSD Detail
#
# Purpose: Show per-parameter RMSD for PST's 6 parameters (App.Tab.2)
# Grid:    PST, N=15, 10 seeds × {RFFGM, GPGM}
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp7_pst_detail.jl
#   julia --project=. experiments/pgm2026/run_exp7_pst_detail.jl \
#       --seed 42,123 --method RFFGM --n_iterations 100 --n_warmup 50
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp7_args()
    s = ArgParseSettings(description="Exp7: PST per-parameter RMSD detail")
    add_common_args!(s)
    @add_arg_table! s begin
        "--N"
            help = "Number of observation points"
            arg_type = Int
            default = 15
    end
    args = ArgParse.parse_args(s)
    return (
        N            = args["N"],
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp7_args()
    config = ODE_CONFIGS["PST"]
    outdir = joinpath(RESULTS_BASE, "exp7")
    mkpath(outdir)

    summary_rows = Dict{String,Any}[]
    perparam_rows = Dict{String,Any}[]

    for seed in args.seeds
        for method in args.methods
            log_run(ode="PST", method=nameof(method), N=args.N, seed=seed)

            result = run_single_experiment(method, config;
                N=args.N, seed=seed,
                n_iterations=args.n_iterations,
                n_warmup=args.n_warmup)

            metrics = compute_all_metrics(result, config.θ_true)
            log_metrics(metrics)

            # Per-parameter absolute error
            per_param = compute_per_param_rmsd(result.θ_chain, config.θ_true)

            # Summary row (aggregate)
            row = make_result_row(metrics; ode_key="PST")
            for (i, pname) in enumerate(config.param_names)
                row["rmsd_$(pname)"] = per_param[i]
            end
            push!(summary_rows, row)

            # Per-parameter long-format rows
            for (i, pname) in enumerate(config.param_names)
                push!(perparam_rows, Dict{String,Any}(
                    "method"     => string(nameof(method)),
                    "seed"       => seed,
                    "param_name" => pname,
                    "param_true" => config.θ_true[i],
                    "param_mean" => metrics.θ_mean[i],
                    "param_std"  => metrics.θ_std[i],
                    "param_rmsd" => per_param[i],
                    "ess"        => metrics.ess_vals[i],
                    "rhat"       => metrics.rhat_vals[i],
                ))
            end

            # Save θ posterior samples
            save_samples_csv(outdir, result.θ_chain, config.param_names;
                prefix="PST_$(nameof(method))", N=args.N, seed=seed)
        end
    end

    save_summary_csv(outdir, summary_rows; filename="exp7_summary.csv")
    save_summary_csv(outdir, perparam_rows; filename="exp7_per_param.csv")
    @info "Exp7 complete."
end

main()
