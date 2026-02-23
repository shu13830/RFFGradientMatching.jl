#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp2: Parameter Estimation Accuracy — RMSD on LV / FN / PST
#
# Purpose: Confirm RFFGM accuracy matches GPGM/MAGI (Tab.2)
# Grid:    {LV, FN, PST} × N ∈ {10, 25, 40} × 10 seeds × {RFFGM, GPGM}
#          (PST default: N=15, σ=0.01)
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp2_accuracy.jl
#   julia --project=. experiments/pgm2026/run_exp2_accuracy.jl \
#       --ode LV --N 10,25,40 --seed 42,123 --method RFFGM
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp2_args()
    s = ArgParseSettings(description="Exp2: RMSD accuracy comparison (LV/FN/PST)")
    add_common_args!(s)
    @add_arg_table! s begin
        "--ode"
            help = "ODE systems (comma-separated, or ALL)"
            default = "ALL"
        "--N"
            help = "Observation points (comma-separated)"
            default = "10,25,40"
    end
    args = ArgParse.parse_args(s)
    ode_str = args["ode"]
    ode_keys = ode_str == "ALL" ? ["LV", "FN", "PST"] : split(ode_str, ",")
    return (
        ode_keys     = String.(ode_keys),
        N_values     = parse.(Int, split(args["N"], ",")),
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp2_args()
    outdir = joinpath(RESULTS_BASE, "exp2")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for ode_key in args.ode_keys
        config = ODE_CONFIGS[ode_key]
        for N in args.N_values
            for seed in args.seeds
                for method in args.methods
                    log_run(ode=ode_key, method=nameof(method), N=N, seed=seed)

                    result = run_single_experiment(method, config;
                        N=N, seed=seed,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup)

                    metrics = compute_all_metrics(result, config.θ_true)
                    log_metrics(metrics)

                    # Save θ posterior samples
                    save_samples_csv(outdir, result.θ_chain, config.param_names;
                        prefix="$(ode_key)_$(nameof(method))", N=N, seed=seed)

                    push!(rows, make_result_row(metrics; ode_key=ode_key))
                end
            end
        end
    end

    save_summary_csv(outdir, rows; filename="exp2_summary.csv")
    @info "Exp2 complete."
end

main()
