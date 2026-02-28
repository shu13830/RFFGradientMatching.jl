#!/usr/bin/env julia
# Compare R MAGI vs Julia MAGI posterior results
#
# Usage:
#   julia --project=. experiments/magi/compare.jl

using CSV
using DataFrames
using Statistics
using Printf

results_dir = joinpath(@__DIR__, "results")

# ── Load samples ──────────────────────────────────────────────────────────
r_file  = joinpath(results_dir, "r_magi_theta_samples.csv")
jl_file = joinpath(results_dir, "julia_magi_theta_samples.csv")

if !isfile(r_file)
    error("R MAGI results not found: $r_file\nRun: Rscript experiments/magi/run_r_magi.R")
end
if !isfile(jl_file)
    error("Julia MAGI results not found: $jl_file\nRun: julia --project=. experiments/magi/run_julia_magi.jl")
end

r_samples  = CSV.read(r_file, DataFrame)
jl_samples = CSV.read(jl_file, DataFrame)

θ_true = [2.0, 1.0, 4.0, 1.0]
param_names = ["a", "b", "c", "d"]

# ── Compute statistics ────────────────────────────────────────────────────
function ci_overlap(lo1, hi1, lo2, hi2)
    overlap_lo = max(lo1, lo2)
    overlap_hi = min(hi1, hi2)
    if overlap_lo >= overlap_hi
        return 0.0
    end
    overlap_width = overlap_hi - overlap_lo
    union_width = max(hi1, hi2) - min(lo1, lo2)
    return overlap_width / union_width
end

println()
println("═" ^ 80)
println("R MAGI vs Julia MAGI: θ posterior comparison (LV, N=10, seed=42)")
println("═" ^ 80)
println()

# Header
@printf("%-6s  %5s  │  %14s  │  %14s  │  %6s  │  %7s\n",
    "param", "true", "R (mean±sd)", "Jl (mean±sd)", "Δmean", "CI ovlp")
println("-" ^ 80)

r_means  = Float64[]
jl_means = Float64[]

for (i, name) in enumerate(param_names)
    r_col  = r_samples[!, Symbol(name)]
    jl_col = jl_samples[!, Symbol(name)]

    r_mean  = mean(r_col)
    r_std   = std(r_col)
    r_q025  = quantile(r_col, 0.025)
    r_q975  = quantile(r_col, 0.975)

    jl_mean = mean(jl_col)
    jl_std  = std(jl_col)
    jl_q025 = quantile(jl_col, 0.025)
    jl_q975 = quantile(jl_col, 0.975)

    Δmean = abs(r_mean - jl_mean)
    overlap = ci_overlap(r_q025, r_q975, jl_q025, jl_q975)

    push!(r_means, r_mean)
    push!(jl_means, jl_mean)

    @printf("%-6s  %5.2f  │  %5.3f ± %5.3f  │  %5.3f ± %5.3f  │  %6.3f  │  %6.1f%%\n",
        name, θ_true[i],
        r_mean, r_std,
        jl_mean, jl_std,
        Δmean, overlap * 100)
end

println("-" ^ 80)

# RMSD comparison
r_rmsd  = sqrt(mean((r_means  .- θ_true).^2))
jl_rmsd = sqrt(mean((jl_means .- θ_true).^2))

@printf("RMSD(θ vs true):  R = %.4f,  Julia = %.4f\n", r_rmsd, jl_rmsd)
@printf("n_samples:        R = %d,    Julia = %d\n", nrow(r_samples), nrow(jl_samples))

# ── σ comparison (if available) ───────────────────────────────────────────
r_sigma_file  = joinpath(results_dir, "r_magi_sigma_samples.csv")
jl_sigma_file = joinpath(results_dir, "julia_magi_sigma_samples.csv")

if isfile(r_sigma_file) && isfile(jl_sigma_file)
    r_sigma  = CSV.read(r_sigma_file, DataFrame)
    jl_sigma = CSV.read(jl_sigma_file, DataFrame)

    println()
    println("σ (observation noise) comparison:")
    @printf("%-16s  │  %14s  │  %14s\n", "component", "R (mean±sd)", "Jl (mean±sd)")
    println("-" ^ 55)
    for col in names(r_sigma)
        r_col  = r_sigma[!, col]
        jl_col = jl_sigma[!, col]
        @printf("%-16s  │  %5.3f ± %5.3f  │  %5.3f ± %5.3f\n",
            col, mean(r_col), std(r_col), mean(jl_col), std(jl_col))
    end
end

# ── Trajectory comparison (if available) ──────────────────────────────────
r_traj_file  = joinpath(results_dir, "r_magi_trajectory.csv")
jl_traj_file = joinpath(results_dir, "julia_magi_trajectory.csv")

if isfile(r_traj_file) && isfile(jl_traj_file)
    r_traj  = CSV.read(r_traj_file, DataFrame)
    jl_traj = CSV.read(jl_traj_file, DataFrame)

    println()
    println("Trajectory comparison (observation time points only):")

    # R trajectory has more points due to discretization; match on Julia times
    jl_times = jl_traj.t

    for comp in ["prey", "predator"]
        r_mean_col  = "mean_$(comp)"

        # Find closest R time points to Julia times
        r_means_at_jl = Float64[]
        for t in jl_times
            idx = argmin(abs.(r_traj.t .- t))
            push!(r_means_at_jl, r_traj[idx, r_mean_col])
        end

        jl_means_comp = jl_traj[!, "mean_$(comp)"]
        max_diff = maximum(abs.(r_means_at_jl .- jl_means_comp))
        mean_diff = mean(abs.(r_means_at_jl .- jl_means_comp))

        @printf("  %-10s: mean |Δ| = %.3f, max |Δ| = %.3f\n",
            comp, mean_diff, max_diff)
    end
end

println()
println("═" ^ 80)
println("NOTE: Differences are expected due to:")
println("  - Different GP kernels (R: generalMatern, Julia: Matern52)")
println("  - Both use discretization with midpoints")
println("  - Different HMC tuning (R: nstepsHmc=200, Julia: n_leapfrog=10)")
println("  - MCMC stochasticity")
println("═" ^ 80)
