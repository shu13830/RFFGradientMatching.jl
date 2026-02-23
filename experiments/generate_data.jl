#!/usr/bin/env julia
# Generate common observation data for all methods (RFFGM, GPGM, MAGI).
# Output: CSV files in baselines/data/ with columns [time, x1, x2, ...].
#
# Usage:
#   julia --project=. experiments/generate_data.jl
#   julia --project=. experiments/generate_data.jl --ode LV --N 40 --seed 42

using DifferentialEquations
using Random
using CSV
using DataFrames
using ArgParse
using JSON

include(joinpath(@__DIR__, "..", "src", "ode.jl"))

# ── Experiment configurations ──────────────────────────────────────────────

const ODE_CONFIGS = Dict(
    "LV" => (
        name = "LotkaVolterra",
        f! = lotkavolterrapredatorprey!,
        θ_true = [2.0, 1.0, 4.0, 1.0],
        u0 = [5.0, 3.0],
        tspan = (0.0, 2.0),
        noise_std = 0.5,
        component_names = ["prey", "predator"],
    ),
    "FN" => (
        name = "FitzHughNagumo",
        f! = fitzhughnagumo!,
        θ_true = [3.0, 0.2, 0.2],       # (c, a, b) in Julia parameterization
        u0 = [-1.0, 1.0],
        tspan = (0.0, 2.0),
        noise_std = 0.5,
        component_names = ["V", "R"],
    ),
    "PST" => (
        name = "ProteinSignaling",
        f! = signaltransductioncascade!,
        θ_true = [0.07, 0.6, 0.05, 0.3, 0.017, 0.3],
        u0 = [1.0, 0.0, 1.0, 0.0, 0.0],
        tspan = (0.0, 100.0),
        noise_std = 0.01,
        component_names = ["S", "dS", "R", "Rs", "Rpp"],
    ),
)

const DEFAULT_SEEDS = [42, 123, 456, 789, 1234, 2345, 3456, 4567, 5678, 6789]
const DEFAULT_N_VALUES = [10, 25, 40]

# ── LVC (Lotka-Volterra Competition) closure ODE ─────────────────────────

"""
Create a closure ODE for LV Competition where only off-diagonal alpha
entries appear as parameters. Returns (f!, θ_true, u0, component_names, aux).
"""
function make_lvc_data_config(K::Int; seed_base::Int=42, noise_std::Float64=0.3)
    rng = MersenneTwister(seed_base)
    r_true  = 1.0 .+ 0.5 .* rand(rng, K)
    Ks_true = 10.0 .+ 5.0 .* rand(rng, K)
    diag_alpha = ones(K)
    alpha_offdiag = 0.1 .+ 0.3 .* rand(rng, K * (K - 1))

    function lvc_offdiag!(du, u, p_offdiag, t)
        α = zeros(eltype(p_offdiag), K, K)
        idx = 1
        for i in 1:K, j in 1:K
            if i == j
                α[i, j] = diag_alpha[i]
            else
                α[i, j] = p_offdiag[idx]
                idx += 1
            end
        end
        for i in 1:K
            interaction = sum(α[i, j] * u[j] for j in 1:K)
            du[i] = r_true[i] * u[i] * (1 - interaction / Ks_true[i])
        end
    end

    config = (
        name = "LVC_K$(K)",
        f! = lvc_offdiag!,
        θ_true = alpha_offdiag,
        u0 = 5.0 .* ones(K),
        tspan = (0.0, 5.0),
        noise_std = noise_std,
        component_names = ["x$i" for i in 1:K],
    )

    aux = Dict(
        "K" => K,
        "r_true" => r_true,
        "Ks_true" => Ks_true,
        "diag_alpha" => diag_alpha,
        "alpha_offdiag_true" => alpha_offdiag,
        "param_names" => [
            "alpha_$(i)_$(j)" for i in 1:K for j in 1:K if i != j
        ],
    )

    return config, aux
end

# ── Data generation ────────────────────────────────────────────────────────

function generate_observations(config; N::Int, seed::Int)
    rng = MersenneTwister(seed)
    times = range(config.tspan[1], config.tspan[2], length=N) |> collect

    prob = ODEProblem(config.f!, config.u0, config.tspan, config.θ_true)
    sol = solve(prob, Tsit5(), saveat=times)
    y_clean = Array(sol)  # K × N
    y_obs = y_clean .+ config.noise_std .* randn(rng, size(y_clean))

    df = DataFrame(:time => times)
    for (i, name) in enumerate(config.component_names)
        df[!, Symbol(name)] = y_obs[i, :]
    end

    # Also save clean trajectories for reference
    df_clean = DataFrame(:time => times)
    for (i, name) in enumerate(config.component_names)
        df_clean[!, Symbol(name)] = y_clean[i, :]
    end

    return df, df_clean
end

function save_data(ode_key::String, config; N_values=DEFAULT_N_VALUES, seeds=DEFAULT_SEEDS)
    outdir = joinpath(@__DIR__, "..", "baselines", "data")
    mkpath(outdir)

    for N in N_values, seed in seeds
        df_obs, df_clean = generate_observations(config; N=N, seed=seed)

        fname_obs = joinpath(outdir, "$(lowercase(ode_key))_N$(N)_seed$(seed).csv")
        fname_clean = joinpath(outdir, "$(lowercase(ode_key))_N$(N)_seed$(seed)_clean.csv")
        CSV.write(fname_obs, df_obs)
        CSV.write(fname_clean, df_clean)
        println("  Written: $(basename(fname_obs))")
    end
end

function save_lvc_data(K::Int; N_values=[40], seeds=DEFAULT_SEEDS)
    outdir = joinpath(@__DIR__, "..", "baselines", "data")
    mkpath(outdir)

    config, aux = make_lvc_data_config(K)

    # Save auxiliary parameter file (for MAGI R scripts)
    params_file = joinpath(outdir, "lvc_K$(K)_params.json")
    open(params_file, "w") do io
        JSON.print(io, aux, 2)
    end
    println("  Written: $(basename(params_file))")

    for N in N_values, seed in seeds
        df_obs, df_clean = generate_observations(config; N=N, seed=seed)

        fname_obs = joinpath(outdir, "lvc_K$(K)_N$(N)_seed$(seed).csv")
        fname_clean = joinpath(outdir, "lvc_K$(K)_N$(N)_seed$(seed)_clean.csv")
        CSV.write(fname_obs, df_obs)
        CSV.write(fname_clean, df_clean)
        println("  Written: $(basename(fname_obs))")
    end
end

# ── CLI ────────────────────────────────────────────────────────────────────

function parse_args()
    s = ArgParseSettings(description="Generate common ODE observation data.")
    @add_arg_table! s begin
        "--ode"
            help = "ODE system (LV, FN, PST, LVC, or ALL)"
            default = "ALL"
        "--N"
            help = "Number of observation points (comma-separated)"
            default = "10,25,40"
        "--seed"
            help = "Random seeds (comma-separated)"
            default = join(DEFAULT_SEEDS, ",")
        "--K"
            help = "LVC species counts (comma-separated, for --ode LVC)"
            default = "2,5,10,20"
    end
    return ArgParse.parse_args(s)
end

function main()
    args = parse_args()
    ode_input = args["ode"]
    N_values = parse.(Int, split(args["N"], ","))
    seeds = parse.(Int, split(args["seed"], ","))
    K_values = parse.(Int, split(args["K"], ","))

    # Standard ODEs
    if ode_input == "ALL" || ode_input != "LVC"
        ode_keys = ode_input == "ALL" ? collect(keys(ODE_CONFIGS)) :
                   ode_input == "LVC" ? String[] : [ode_input]
        for key in ode_keys
            config = ODE_CONFIGS[key]
            println("Generating $(config.name) data...")
            save_data(key, config; N_values=N_values, seeds=seeds)
        end
    end

    # LVC (Lotka-Volterra Competition)
    if ode_input == "ALL" || ode_input == "LVC"
        for K in K_values
            println("Generating LVC K=$K data (|θ|=$(K*(K-1)))...")
            save_lvc_data(K; N_values=N_values, seeds=seeds)
        end
    end

    println("Done.")
end

main()
