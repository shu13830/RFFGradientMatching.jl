#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp3: Kernel Flexibility -- Non-smooth kernels for gradient matching
#
# Purpose: Demonstrate RFFGM advantage with generalized kernels (Tab.3)
# ODE:     LV (K=2, N=40), FN (K=2, N=40)
# Kernels: RBF, Matern-5/2, Laplace, GenCauchy, ExpPower
# Grid:    {LV, FN} x {5 kernels} x 10 seeds x {RFFGM, GPGM where applicable}
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp3_kernel_flex.jl
#   julia --project=. experiments/pgm2026/run_exp3_kernel_flex.jl \
#       --ode LV --kernel RBF,GenCauchy --seed 42,123 --method RFFGM
# -----------------------------------------------------------------

include(joinpath(@__DIR__, "common.jl"))

# -- CLI -----------------------------------------------------------

function parse_exp3_args()
    s = ArgParseSettings(description="Exp3: Kernel flexibility comparison")
    add_common_args!(s)
    @add_arg_table! s begin
        "--ode"
            help = "ODE systems (comma-separated)"
            default = "LV,FN"
        "--kernel"
            help = "Kernel names (comma-separated, or ALL)"
            default = "ALL"
        "--N"
            help = "Number of observation points"
            arg_type = Int
            default = 40
    end
    args = ArgParse.parse_args(s)
    return (
        ode_keys     = String.(split(args["ode"], ",")),
        kernel_keys  = parse_kernels(args["kernel"]),
        N            = args["N"],
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# -- Main ----------------------------------------------------------

function main()
    args = parse_exp3_args()
    outdir = joinpath(RESULTS_BASE, "exp3")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for ode_key in args.ode_keys
        config = ODE_CONFIGS[ode_key]
        for kname in args.kernel_keys
            kconfig = KERNEL_CONFIGS[kname]
            for seed in args.seeds
                for method in args.methods
                    # Skip methods that don't support this kernel
                    if !(method in kconfig.methods)
                        @info "Skipping $(kname) for $(nameof(method)) (unsupported)"
                        continue
                    end

                    log_run(ode=ode_key, method=nameof(method),
                        N=args.N, seed=seed,
                        extra="kernel=$kname")

                    try
                        result = run_single_experiment(method, config;
                            N=args.N, seed=seed,
                            kernel=kconfig.kernel,
                            n_iterations=args.n_iterations,
                            n_warmup=args.n_warmup)

                        metrics = compute_all_metrics(result, config.θ_true)
                        log_metrics(metrics)

                        save_samples_csv(outdir, result.θ_chain, config.param_names;
                            prefix="$(ode_key)_$(nameof(method))_$(kname)",
                            N=args.N, seed=seed)

                        push!(rows, make_result_row(metrics;
                            ode_key=ode_key,
                            extra_cols=Dict{String,Any}("kernel" => kname)))

                    catch e
                        @warn "Run failed" ode_key kname method=nameof(method) seed exception=e
                        push!(rows, Dict{String,Any}(
                            "ode"         => ode_key,
                            "method"      => string(nameof(method)),
                            "N"           => args.N,
                            "seed"        => seed,
                            "kernel"      => kname,
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
    end

    save_summary_csv(outdir, rows; filename="exp3_summary.csv")
    @info "Exp3 complete."
end

main()
