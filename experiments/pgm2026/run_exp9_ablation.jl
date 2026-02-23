#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp9: L Ablation -- RFF feature count sensitivity
#
# Purpose: Show RFFGM quality vs number of RFF features (Appendix Fig.)
# ODE:     Lotka-Volterra (K=2, N=40)
# L:       {20, 50, 80, 100, 150, 200}
# Output:  RMSD vs L, ESS vs L, RFF approx error ||Phi*Phi'-K||_F / N
#          5 frequency resamples per L (error bars)
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp9_ablation.jl
#   julia --project=. experiments/pgm2026/run_exp9_ablation.jl \
#       --L 20,50,100 --seed 42,123 --n_resamples 3
# -----------------------------------------------------------------

include(joinpath(@__DIR__, "common.jl"))

# -- CLI -----------------------------------------------------------

function parse_exp9_args()
    s = ArgParseSettings(description="Exp9: L (RFF feature count) ablation on LV")
    add_common_args!(s)
    @add_arg_table! s begin
        "--L"
            help = "RFF feature counts (comma-separated)"
            default = "20,50,80,100,150,200"
        "--N"
            help = "Number of observation points"
            arg_type = Int
            default = 40
        "--n_resamples"
            help = "Number of frequency resamples per (L, seed)"
            arg_type = Int
            default = 5
    end
    args = ArgParse.parse_args(s)
    return (
        L_values     = parse.(Int, split(args["L"], ",")),
        N            = args["N"],
        seeds        = parse_seeds(args["seed"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
        n_resamples  = args["n_resamples"],
    )
end

# -- Main ----------------------------------------------------------

function main()
    args = parse_exp9_args()
    config = ODE_CONFIGS["LV"]
    outdir = joinpath(RESULTS_BASE, "exp9")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for L in args.L_values
        for seed in args.seeds
            # Generate data once per seed (fixed observations)
            Random.seed!(seed)
            times, y_obs, y_clean, prob = generate_data(config; N=args.N, seed=seed)

            for resample_idx in 1:args.n_resamples
                # Different RNG seed for each frequency resample
                freq_seed = seed * 1000 + resample_idx

                log_run(ode="LV", method="RFFGM", N=args.N, seed=seed,
                    extra="L=$L resample=$resample_idx/$(args.n_resamples)")

                try
                    Random.seed!(freq_seed)

                    _anneal = min(ANNEAL_LENGTH, max(args.n_warmup - 1, 0))
                    gm = setup_model(RFFGM, config, times, y_obs, prob;
                        n_rff=L,
                        anneal_length=_anneal)

                    # Compute RFF approximation error BEFORE MCMC
                    rff_error = compute_rff_approx_error(gm)

                    bs = create_blocked_sampler(gm)
                    t_start = time()
                    chain, logdens = AbstractMCMC.sample(
                        gm, bs, args.n_iterations;
                        num_burnin=args.n_warmup, anneal=true)
                    wall_time = time() - t_start

                    θ_chain = get_θ(gm, chain[args.n_warmup+1:end])
                    result = (;
                        θ_chain, chain, logdens, gm,
                        wall_time, times, y_obs, y_clean,
                        config, N=args.N, seed, method="RFFGM")

                    metrics = compute_all_metrics(result, config.θ_true)
                    log_metrics(metrics)

                    row = make_result_row(metrics; ode_key="LV",
                        extra_cols=Dict{String,Any}(
                            "L"             => L,
                            "resample_idx"  => resample_idx,
                            "freq_seed"     => freq_seed,
                            "rff_error"     => rff_error))
                    push!(rows, row)

                catch e
                    @warn "Run failed" L seed resample_idx exception=e
                    push!(rows, Dict{String,Any}(
                        "ode"           => "LV",
                        "method"        => "RFFGM",
                        "N"             => args.N,
                        "seed"          => seed,
                        "L"             => L,
                        "resample_idx"  => resample_idx,
                        "freq_seed"     => freq_seed,
                        "rmsd"          => NaN,
                        "ess_mean"      => NaN,
                        "time_sec"      => NaN,
                        "ess_per_sec"   => NaN,
                        "rhat_max"      => NaN,
                        "rff_error"     => NaN,
                    ))
                end
            end
        end
    end

    save_summary_csv(outdir, rows; filename="exp9_summary.csv")
    @info "Exp9 complete."
end

main()
