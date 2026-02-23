#!/usr/bin/env julia
# Aggregate results from all methods (RFFGM, GPGM, MAGI) into comparison tables.
#
# Reads:
#   baselines/magi/results/{ode}_summary.csv
#   experiments/results/E4/{ode}_rffgm_summary.csv  (when available)
#   experiments/results/E4/{ode}_gpgm_summary.csv   (when available)
#
# Outputs:
#   baselines/results_comparison.csv
#   stdout: formatted comparison tables
#
# Usage:
#   julia --project=. baselines/compare.jl

using CSV
using DataFrames
using Printf
using Statistics

const BASE_DIR = @__DIR__
const MAGI_RESULTS = joinpath(BASE_DIR, "magi", "results")
const JULIA_RESULTS = joinpath(dirname(BASE_DIR), "experiments", "results", "E4")
const OUTPUT_FILE = joinpath(BASE_DIR, "results_comparison.csv")

# ── Load results ───────────────────────────────────────────────────────────

function load_magi_results()
    dfs = DataFrame[]
    for ode in ["lv", "fn", "pst"]
        f = joinpath(MAGI_RESULTS, "$(ode)_summary.csv")
        if isfile(f)
            push!(dfs, CSV.read(f, DataFrame))
        end
    end
    isempty(dfs) ? DataFrame() : vcat(dfs..., cols=:union)
end

function load_julia_results()
    dfs = DataFrame[]
    if !isdir(JULIA_RESULTS)
        return DataFrame()
    end
    for f in readdir(JULIA_RESULTS; join=true)
        endswith(f, "_summary.csv") || continue
        push!(dfs, CSV.read(f, DataFrame))
    end
    isempty(dfs) ? DataFrame() : vcat(dfs..., cols=:union)
end

# ── Summarize ──────────────────────────────────────────────────────────────

function summarize(df::DataFrame)
    isempty(df) && return DataFrame()

    cols = names(df)
    has_col(c) = c in cols

    gdf = groupby(df, [:ode, :method, :N])
    summary = combine(gdf,
        :rmsd => mean => :rmsd_mean,
        :rmsd => std => :rmsd_std,
        :ess_mean => mean => :ess_mean,
        :time_sec => mean => :time_mean,
        :ess_per_sec => mean => :ess_per_sec_mean,
        nrow => :n_seeds,
    )
    sort!(summary, [:ode, :N, :method])
    return summary
end

# ── Print table ────────────────────────────────────────────────────────────

function print_table(summary::DataFrame)
    isempty(summary) && return

    for ode in unique(summary.ode)
        sub = filter(r -> r.ode == ode, summary)
        println("\n", "="^70)
        @printf("  %s\n", uppercase(ode))
        println("="^70)
        @printf("%-8s %-8s %8s %8s %8s %10s %8s\n",
                "Method", "N", "RMSD", "±std", "ESS", "Time(s)", "ESS/s")
        println("-"^70)
        for r in eachrow(sub)
            @printf("%-8s %-8d %8.4f %8.4f %8.1f %10.1f %8.2f\n",
                    r.method, r.N,
                    r.rmsd_mean, coalesce(r.rmsd_std, 0.0),
                    r.ess_mean, r.time_mean, r.ess_per_sec_mean)
        end
    end
    println()
end

# ── Main ───────────────────────────────────────────────────────────────────

function main()
    println("Loading results...")

    magi_df = load_magi_results()
    julia_df = load_julia_results()

    if isempty(magi_df) && isempty(julia_df)
        println("No results found. Run experiments first:")
        println("  julia --project=. experiments/generate_data.jl")
        println("  Rscript baselines/magi/run_lv.R")
        println("  # etc.")
        return
    end

    all_df = vcat(magi_df, julia_df, cols=:union)
    summary = summarize(all_df)

    # Print to stdout
    print_table(summary)

    # Save comparison CSV
    CSV.write(OUTPUT_FILE, summary)
    println("Comparison saved to: ", OUTPUT_FILE)
end

main()
