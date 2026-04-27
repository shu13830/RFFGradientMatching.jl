#!/usr/bin/env julia
# Generate data for R MAGI comparison
# Outputs CSV files to baselines/data/ for each ODE × N × θ × seed

include(joinpath(@__DIR__, "..", "experiments", "pgm2026", "common.jl"))
using Printf

outdir = joinpath(@__DIR__, "data")
mkpath(outdir)

SEEDS = [42, 123, 456, 789, 1234]

for ode_name in ["LV", "PST", "FN"]
    config = ODE_CONFIGS[ode_name]
    for theta_id in [1, 2, 3]
        θ_true = get_θ_true(config, theta_id)
        config_t = ODEConfig(config.name, config.f!, θ_true, config.θ_patterns,
            config.u0, config.tspan, config.noise_std, config.kernel,
            config.param_names, config.component_names)
        for N in [10, 25, 50]
            for seed in SEEDS
                Random.seed!(seed)
                times, y_obs, y_clean, prob = generate_data(config_t; N=N, seed=seed)

                # Save as CSV: time, comp1, comp2, ...
                fname = joinpath(outdir, "$(lowercase(ode_name))_N$(N)_t$(theta_id)_seed$(seed).csv")
                open(fname, "w") do io
                    header = ["time"; config.component_names]
                    println(io, join(header, ","))
                    for i in 1:N
                        vals = [@sprintf("%.8f", times[i]); [@sprintf("%.8f", y_obs[k, i]) for k in 1:size(y_obs, 1)]]
                        println(io, join(vals, ","))
                    end
                end
            end
        end
    end
    println("Generated data for $ode_name")
end

# Also save true θ values
open(joinpath(outdir, "theta_true.csv"), "w") do io
    println(io, "ode,theta_id,param_names,theta_values")
    for ode_name in ["LV", "PST", "FN"]
        config = ODE_CONFIGS[ode_name]
        for theta_id in [1, 2, 3]
            θ = get_θ_true(config, theta_id)
            println(io, "$ode_name,$theta_id,$(join(config.param_names,";")),$(join(θ,";"))")
        end
    end
end
println("Saved theta_true.csv")
println("Done. Files in: $outdir")
