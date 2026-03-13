#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp1-ELMC: Convergence Comparison with ELMC (Monge metric LMC)
#
# Purpose: Compare RFFGM (L=50,100,200) vs GPGM on Lotka-Volterra
#          using ELMC (Lagrangian MC on Monge patches) for all
#          latent variables jointly.
# Grid:    N ∈ {10, 25} × seeds × {RFFGM(L=50), RFFGM(L=100),
#          RFFGM(L=200), GPGM}
# Settings: burn-in 10000, sampling 10000, joint ELMC
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp1_elmc.jl
#   julia --project=. experiments/pgm2026/run_exp1_elmc.jl \
#       --N 10,25 --seed 42 --L 50,100,200 --alpha 1.0
# -----------------------------------------------------------------

include(joinpath(@__DIR__, "common.jl"))

# ── ELMC sampler (joint, Monge metric) ───────────────────────────

function create_elmc_sampler(gm::RFFGM;
    step_size::Float64=0.001, n_leapfrog::Int=50, α::Float64=1.0)
    block = ELMCBlock(gm, [:W, :θ]; n_leapfrog=n_leapfrog, step_size=step_size, α=α)
    return BlockedSampler([[block]], [1.0])
end

function create_elmc_sampler(gm::GPGM;
    step_size::Float64=0.001, n_leapfrog::Int=50, α::Float64=1.0)
    block = ELMCBlock(gm, [:X, :θ]; n_leapfrog=n_leapfrog, step_size=step_size, α=α)
    return BlockedSampler([[block]], [1.0])
end

# ── Experiment runner (ELMC variant) ─────────────────────────────

function run_elmc_experiment(
    method::Type{<:Union{RFFGM,GPGM}}, config::ODEConfig;
    N::Int, seed::Int,
    kernel=config.kernel,
    n_rff::Int=DEFAULT_N_RFF,
    n_iterations::Int=20_000,
    n_warmup::Int=10_000,
    anneal_length::Int=ANNEAL_LENGTH,
    step_size::Float64=0.001,
    n_leapfrog::Int=50,
    α::Float64=1.0,
)
    Random.seed!(seed)
    times, y_obs, y_clean, prob = generate_data(config; N=N, seed=seed)

    _anneal = min(anneal_length, max(n_warmup - 1, 0))
    gm = setup_model(method, config, times, y_obs, prob;
        kernel=kernel, n_rff=n_rff, anneal_length=_anneal)
    bs = create_elmc_sampler(gm; step_size=step_size, n_leapfrog=n_leapfrog, α=α)

    t_start = time()
    chain, logdens = AbstractMCMC.sample(gm, bs, n_iterations;
        num_burnin=n_warmup, anneal=true)
    wall_time = time() - t_start

    # Extract accept rate from the ELMCBlock
    blk = bs.blocks[1][1]
    n_accept = sum(blk.accept_history)
    n_total = length(blk.accept_history)
    accept_rate = n_total > 0 ? n_accept / n_total : NaN

    θ_chain = get_θ(gm, chain[n_warmup+1:end])

    return (;
        θ_chain, chain, logdens, gm,
        wall_time, times, y_obs, y_clean,
        config, N, seed, method=string(nameof(method)),
        accept_rate,
    )
end

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp1_elmc_args()
    s = ArgParseSettings(description="Exp1-ELMC: Convergence with ELMC Monge metric (LV)")
    add_common_args!(s)
    @add_arg_table! s begin
        "--N"
            help = "Observation points (comma-separated)"
            default = "10,25"
        "--L"
            help = "RFF feature counts (comma-separated)"
            default = "50,100,200"
        "--alpha"
            help = "ELMC embedding parameter α"
            arg_type = Float64
            default = 1.0
        "--step_size"
            help = "ELMC base step size"
            arg_type = Float64
            default = 0.001
        "--n_leapfrog"
            help = "Number of leapfrog steps"
            arg_type = Int
            default = 50
    end
    args = ArgParse.parse_args(s)
    return (
        N_values     = parse.(Int, split(args["N"], ",")),
        L_values     = parse.(Int, split(args["L"], ",")),
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
        α            = args["alpha"],
        step_size    = args["step_size"],
        n_leapfrog   = args["n_leapfrog"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp1_elmc_args()
    config = ODE_CONFIGS["LV"]
    outdir = joinpath(RESULTS_BASE, "exp1_elmc")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for N in args.N_values
        for seed in args.seeds
            # --- RFFGM with multiple L values ---
            if RFFGM in args.methods
                for L in args.L_values
                    label = "RFFGM_L$(L)"
                    @info "=== $label | N=$N | seed=$seed | α=$(args.α) ==="

                    result = run_elmc_experiment(RFFGM, config;
                        N=N, seed=seed, n_rff=L,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup,
                        step_size=args.step_size,
                        n_leapfrog=args.n_leapfrog,
                        α=args.α)

                    metrics = compute_all_metrics(result, config.θ_true)
                    log_metrics(metrics)
                    @info "  accept_rate=$(round(result.accept_rate, digits=3))"

                    logdens_all = get_logdensity(result.gm, result.chain)
                    save_logdens_csv(outdir, logdens_all;
                        prefix=label, N=N, seed=seed)
                    save_samples_csv(outdir, result.θ_chain, config.param_names;
                        prefix=label, N=N, seed=seed)

                    row = make_result_row(metrics; ode_key="LV")
                    row["method"] = label
                    row["n_rff"] = L
                    row["accept_rate"] = result.accept_rate
                    push!(rows, row)
                end
            end

            # --- GPGM (no L parameter) ---
            if GPGM in args.methods
                label = "GPGM"
                @info "=== $label | N=$N | seed=$seed | α=$(args.α) ==="

                result = run_elmc_experiment(GPGM, config;
                    N=N, seed=seed,
                    n_iterations=args.n_iterations,
                    n_warmup=args.n_warmup,
                    step_size=args.step_size,
                    n_leapfrog=args.n_leapfrog,
                    α=args.α)

                metrics = compute_all_metrics(result, config.θ_true)
                log_metrics(metrics)
                @info "  accept_rate=$(round(result.accept_rate, digits=3))"

                logdens_all = get_logdensity(result.gm, result.chain)
                save_logdens_csv(outdir, logdens_all;
                    prefix=label, N=N, seed=seed)
                save_samples_csv(outdir, result.θ_chain, config.param_names;
                    prefix=label, N=N, seed=seed)

                row = make_result_row(metrics; ode_key="LV")
                row["method"] = label
                row["n_rff"] = 0
                row["accept_rate"] = result.accept_rate
                push!(rows, row)
            end
        end
    end

    save_summary_csv(outdir, rows; filename="exp1_elmc_summary.csv")
    @info "Exp1-ELMC complete."
end

main()
