#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp1: Convergence Comparison — RFFGM vs GPGM on Lotka-Volterra
#
# Purpose: Visualise faster HMC mixing of RFFGM (Fig.1 + Tab.1)
# Grid:    N ∈ {10, 25, 40} × 5 seeds × {RFFGM, GPGM}
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp1_convergence.jl
#   julia --project=. experiments/pgm2026/run_exp1_convergence.jl \
#       --N 10,25,40 --seed 42,123,456,789,1234 --method ALL
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp1_args()
    s = ArgParseSettings(description="Exp1: Convergence comparison (LV)")
    add_common_args!(s)
    @add_arg_table! s begin
        "--N"
            help = "Observation points (comma-separated)"
            default = "10,25,40"
    end
    args = ArgParse.parse_args(s)
    return (
        N_values     = parse.(Int, split(args["N"], ",")),
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp1_args()
    config = ODE_CONFIGS["LV"]
    outdir = joinpath(RESULTS_BASE, "exp1")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for N in args.N_values
        for seed in args.seeds
            for method in args.methods
                log_run(ode="LV", method=nameof(method), N=N, seed=seed)

                result = run_single_experiment(method, config;
                    N=N, seed=seed,
                    n_iterations=args.n_iterations,
                    n_warmup=args.n_warmup)

                metrics = compute_all_metrics(result, config.θ_true)
                log_metrics(metrics)

                # Save log-density trace (for convergence plot, Fig.1)
                logdens_all = get_logdensity(result.gm, result.chain)
                save_logdens_csv(outdir, logdens_all;
                    prefix=string(nameof(method)), N=N, seed=seed)

                # Save θ posterior samples
                save_samples_csv(outdir, result.θ_chain, config.param_names;
                    prefix=string(nameof(method)), N=N, seed=seed)

                push!(rows, make_result_row(metrics; ode_key="LV"))
            end
        end
    end

    save_summary_csv(outdir, rows; filename="exp1_summary.csv")
    @info "Exp1 complete."
end

main()
