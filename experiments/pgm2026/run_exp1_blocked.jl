#!/usr/bin/env julia
# -----------------------------------------------------------------
# Exp1-Blocked: Convergence Comparison with Block Sampling
#
# Purpose: Compare RFFGM (L=50,100,200) vs GPGM on Lotka-Volterra
#          using blocked HMC (W/X, W+θ/X+θ, θ with probs 0.4, 0.4, 0.2).
# Grid:    N ∈ {10, 25} × seeds × {RFFGM(L=50), RFFGM(L=100),
#          RFFGM(L=200), GPGM}
# Settings: burn-in 10000, sampling 10000, block sampling, n_leapfrog=50
#
# Usage:
#   julia --project=. experiments/pgm2026/run_exp1_blocked.jl
#   julia --project=. experiments/pgm2026/run_exp1_blocked.jl \
#       --N 10,25 --seed 42 --L 50,100,200
# -----------------------------------------------------------------#

include(joinpath(@__DIR__, "common.jl"))

# ── Blocked sampler with configurable n_leapfrog ──────────────────

function create_blocked_sampler_lf(gm::RFFGM;
    step_size_latent=0.05, step_size_joint=0.01, step_size_theta=0.05,
    n_leapfrog=50)

    block_W  = HMCBlock(gm, [:W];     n_leapfrog=n_leapfrog, step_size=step_size_latent, metric=:diag)
    block_Wθ = HMCBlock(gm, [:W, :θ]; n_leapfrog=n_leapfrog, step_size=step_size_joint,  metric=:diag)
    block_θ  = HMCBlock(gm, [:θ];     n_leapfrog=n_leapfrog, step_size=step_size_theta,  metric=:diag)
    return BlockedSampler([[block_W], [block_Wθ], [block_θ]], [0.4, 0.4, 0.2])
end

function create_blocked_sampler_lf(gm::GPGM;
    step_size_latent=0.05, step_size_joint=0.01, step_size_theta=0.05,
    n_leapfrog=50)

    block_X  = HMCBlock(gm, [:X];     n_leapfrog=n_leapfrog, step_size=step_size_latent, metric=:diag)
    block_Xθ = HMCBlock(gm, [:X, :θ]; n_leapfrog=n_leapfrog, step_size=step_size_joint,  metric=:diag)
    block_θ  = HMCBlock(gm, [:θ];     n_leapfrog=n_leapfrog, step_size=step_size_theta,  metric=:diag)
    return BlockedSampler([[block_X], [block_Xθ], [block_θ]], [0.4, 0.4, 0.2])
end

# ── Experiment runner (blocked with custom n_leapfrog) ────────────

function run_blocked_experiment(
    method::Type{<:Union{RFFGM,GPGM}}, config::ODEConfig;
    N::Int, seed::Int,
    kernel=config.kernel,
    n_rff::Int=DEFAULT_N_RFF,
    n_iterations::Int=20_000,
    n_warmup::Int=10_000,
    anneal_length::Int=ANNEAL_LENGTH,
)
    Random.seed!(seed)
    times, y_obs, y_clean, prob = generate_data(config; N=N, seed=seed)

    _anneal = min(anneal_length, max(n_warmup - 1, 0))
    gm = setup_model(method, config, times, y_obs, prob;
        kernel=kernel, n_rff=n_rff, anneal_length=_anneal)
    bs = create_blocked_sampler_lf(gm)

    t_start = time()
    chain, logdens = AbstractMCMC.sample(gm, bs, n_iterations;
        num_burnin=n_warmup, anneal=true)
    wall_time = time() - t_start

    θ_chain = get_θ(gm, chain[n_warmup+1:end])

    return (;
        θ_chain, chain, logdens, gm,
        wall_time, times, y_obs, y_clean,
        config, N, seed, method=string(nameof(method)),
    )
end

# ── CLI ───────────────────────────────────────────────────────────

function parse_exp1_blocked_args()
    s = ArgParseSettings(description="Exp1-Blocked: Convergence with block HMC (LV)")
    add_common_args!(s)
    @add_arg_table! s begin
        "--N"
            help = "Observation points (comma-separated)"
            default = "10,25"
        "--L"
            help = "RFF feature counts (comma-separated)"
            default = "50,100,200"
    end
    args = ArgParse.parse_args(s)
    return (
        N_values     = parse.(Int, split(args["N"], ",")),
        L_values     = parse.(Int, split(args["L"], ",")),
        seeds        = parse_seeds(args["seed"]),
        methods      = parse_methods(args["method"]),
        n_iterations = args["n_iterations"],
        n_warmup     = args["n_warmup"],
    )
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    args = parse_exp1_blocked_args()
    config = ODE_CONFIGS["LV"]
    outdir = joinpath(RESULTS_BASE, "exp1_blocked")
    mkpath(outdir)

    rows = Dict{String,Any}[]

    for N in args.N_values
        for seed in args.seeds
            # --- RFFGM with multiple L values ---
            if RFFGM in args.methods
                for L in args.L_values
                    label = "RFFGM_L$(L)"
                    @info "=== $label | N=$N | seed=$seed ==="

                    result = run_blocked_experiment(RFFGM, config;
                        N=N, seed=seed, n_rff=L,
                        n_iterations=args.n_iterations,
                        n_warmup=args.n_warmup)

                    metrics = compute_all_metrics(result, config.θ_true)
                    log_metrics(metrics)

                    logdens_all = get_logdensity(result.gm, result.chain)
                    save_logdens_csv(outdir, logdens_all;
                        prefix=label, N=N, seed=seed)
                    save_samples_csv(outdir, result.θ_chain, config.param_names;
                        prefix=label, N=N, seed=seed)

                    row = make_result_row(metrics; ode_key="LV")
                    row["method"] = label
                    row["n_rff"] = L
                    push!(rows, row)
                end
            end

            # --- GPGM (no L parameter) ---
            if GPGM in args.methods
                label = "GPGM"
                @info "=== $label | N=$N | seed=$seed ==="

                result = run_blocked_experiment(GPGM, config;
                    N=N, seed=seed,
                    n_iterations=args.n_iterations,
                    n_warmup=args.n_warmup)

                metrics = compute_all_metrics(result, config.θ_true)
                log_metrics(metrics)

                logdens_all = get_logdensity(result.gm, result.chain)
                save_logdens_csv(outdir, logdens_all;
                    prefix=label, N=N, seed=seed)
                save_samples_csv(outdir, result.θ_chain, config.param_names;
                    prefix=label, N=N, seed=seed)

                row = make_result_row(metrics; ode_key="LV")
                row["method"] = label
                row["n_rff"] = 0
                push!(rows, row)
            end
        end
    end

    save_summary_csv(outdir, rows; filename="exp1_blocked_summary.csv")
    @info "Exp1-Blocked complete."
end

main()
