#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp5: Computation Time — Wall-Clock + ESS/sec
#
# Purpose: Verify theoretical scaling O(NL) vs O(N³) (Tab.5)
# ODE:     Lotka-Volterra (K=2)
# Grid:    N ∈ {50, 100, 200} × 5 seeds × {RFFGM, GPGM}
#          5 timing repeats per cell (report median)
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp5_timing.jl
#   julia --project=. experiments/pgm2026/run_exp5_timing.jl \
#       --N 50,100 --seed 42 --n_repeats 3
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp5_args()
    s = ArgParseSettings(description="Exp5: Computation time comparison (LV)")
    add_common_args!(s)
    @add_arg_table! s begin
        "--N"
            help = "Observation points (comma-separated)"
            default = "50,100,200"
        "--n_repeats"
            help = "Number of timing repeats per cell (report median)"
            arg_type = Int
            default = 5
    end
    args = ArgParse.parse_args(s)
    return (
        N_values     = parse.(Int, split(args["N"], ",")),
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
        n_repeats    = args["n_repeats"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp5_args()
    config = ODE_CONFIGS["LV"]
    outdir = joinpath(RESULTS_BASE, "exp5")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for N in args.N_values
        for seed in args.seeds
            for method in args.methods
                @info "=== Timing: LV | $(nameof(method)) | N=$N | seed=$seed ==="

                # Run n_repeats times, collect wall times
                wall_times = Float64[]
                local last_result

                for rep in 1:args.n_repeats
                    log_run(ode="LV", method=nameof(method), N=N, seed=seed,
                        extra="rep=$rep/$(args.n_repeats)")

                    result = run_single_experiment(method, config;
                        N=N, seed=seed,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup)

                    push!(wall_times, result.wall_time)
                    last_result = result
                end

                # Use median wall time
                median_time = sort(wall_times)[div(length(wall_times), 2) + 1]

                # Compute metrics from the last run
                metrics = compute_all_metrics(last_result, config.θ_true)

                # Override wall_time and ess_per_sec with median
                ess_per_sec_median = metrics.ess_mean / median_time

                @info @sprintf("  Median time=%.2fs (min=%.2f, max=%.2f)  ESS/s=%.2f",
                    median_time, minimum(wall_times), maximum(wall_times),
                    ess_per_sec_median)

                row = make_result_row(metrics; ode_key="LV")
                row["time_sec"] = median_time
                row["ess_per_sec"] = ess_per_sec_median
                row["time_min"] = minimum(wall_times)
                row["time_max"] = maximum(wall_times)
                row["n_repeats"] = args.n_repeats

                push!(rows, row)
            end
        end
    end

    save_summary_csv(outdir, rows; filename="exp5_summary.csv")
    @info "Exp5 complete."
end

main()
