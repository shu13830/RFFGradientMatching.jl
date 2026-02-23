#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp4: High-Dimensional Scaling — LV Competition  ★ MOST IMPORTANT ★
#
# Purpose: Demonstrate RFFGM advantage as K grows (Fig.2 + Tab.4)
# ODE:     Lotka-Volterra Competition, off-diagonal α only
#          K ∈ {2, 5, 10, 20}  →  |θ| ∈ {2, 20, 90, 380}
# Grid:    K × 10 seeds × {RFFGM, GPGM}
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp4_scaling.jl
#   julia --project=. experiments/pgm2026/run_exp4_scaling.jl \
#       --K 2,5 --N 40 --seed 42,123 --method RFFGM
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp4_args()
    s = ArgParseSettings(description="Exp4: LVC high-dimensional scaling")
    add_common_args!(s)
    @add_arg_table! s begin
        "--K"
            help = "Species counts (comma-separated)"
            default = "2,5,10,20"
        "--N"
            help = "Number of observation points"
            arg_type = Int
            default = 40
        "--n_rff"
            help = "Number of RFF features"
            arg_type = Int
            default = DEFAULT_N_RFF
        "--timeout"
            help = "Timeout per run in seconds (0 = no timeout)"
            arg_type = Int
            default = 3600
    end
    args = ArgParse.parse_args(s)
    return (
        K_values     = parse.(Int, split(args["K"], ",")),
        N            = args["N"],
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
        n_rff        = args["n_rff"],
        timeout      = args["timeout"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp4_args()
    outdir = joinpath(RESULTS_BASE, "exp4")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for K in args.K_values
        n_params = K * (K - 1)
        config, aux = make_lvc_config(K)
        @info "=== LVC K=$K  |θ|=$n_params ==="

        # Save LVC auxiliary params for reproducibility
        params_file = joinpath(outdir, "lvc_K$(K)_params.json")
        open(params_file, "w") do io
            JSON.print(io, aux, 2)
        end

        for seed in args.seeds
            for method in args.methods
                log_run(ode="LVC_K$(K)", method=nameof(method),
                    N=args.N, seed=seed,
                    extra="|θ|=$n_params")

                try
                    result = run_single_experiment(method, config;
                        N=args.N, seed=seed,
                        n_rff=args.n_rff,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup)

                    metrics = compute_all_metrics(result, config.θ_true)
                    log_metrics(metrics)

                    # Save θ posterior samples
                    save_samples_csv(outdir, result.θ_chain, config.param_names;
                        prefix="LVC_K$(K)_$(nameof(method))",
                        N=args.N, seed=seed)

                    push!(rows, make_result_row(metrics;
                        ode_key="LVC_K$(K)",
                        extra_cols=Dict{String,Any}(
                            "K" => K, "n_params" => n_params)))

                catch e
                    @warn "Run failed" K method=nameof(method) seed exception=e
                    # Record failed run as NaN
                    push!(rows, Dict{String,Any}(
                        "ode"         => "LVC_K$(K)",
                        "method"      => string(nameof(method)),
                        "N"           => args.N,
                        "seed"        => seed,
                        "K"           => K,
                        "n_params"    => n_params,
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

    save_summary_csv(outdir, rows; filename="exp4_summary.csv")
    @info "Exp4 complete."
end

main()
