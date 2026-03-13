#!/usr/bin/env julia
# Grid search: ELMC step_size × α for GPGM (D=24, fast)
# Also runs α=0 (Euclidean HMC equivalent) as baseline.

include(joinpath(@__DIR__, "common.jl"))
using RFFGradientMatching: BlockedSamplerState
using Printf

function run_short_elmc(gm, vars, label;
    step_size, α, n_leapfrog=10, n_iter=500, n_warmup=200)

    param_dict = pack_param_dict(gm)
    gm_copy = deepcopy(gm)
    gm_copy.anneal_iter[1] = gm_copy.anneal_length
    gm_copy.β[1] = 1.0

    blk = ELMCBlock(gm_copy, vars; n_leapfrog=n_leapfrog, step_size=step_size, α=α)
    state = BlockedSamplerState(pack_param_dict(gm_copy))
    bs = BlockedSampler([[blk]], [1.0])

    # Run
    t0 = time()
    try
        chain, logdens = AbstractMCMC.sample(gm_copy, bs, n_iter;
            num_burnin=n_warmup, anneal=false)
        wall = time() - t0

        # Accept rate
        n_acc = sum(blk.accept_history)
        n_tot = length(blk.accept_history)
        acc_rate = n_tot > 0 ? n_acc / n_tot : NaN

        # θ posterior stats
        θ_chain = get_θ(gm_copy, chain[n_warmup+1:end])
        n_samples = size(θ_chain, 1)
        θ_mean = n_samples > 0 ? vec(mean(θ_chain, dims=1)) : fill(NaN, 4)
        θ_std = n_samples > 0 ? vec(std(θ_chain, dims=1)) : fill(NaN, 4)

        # RMSD from true
        θ_true = [0.67, 1.33, 1.0, 1.0]
        rmsd = sqrt(mean((θ_mean .- θ_true).^2))

        # Logdens stats
        ld_post = logdens[n_warmup+1:end]
        ld_mean = mean(ld_post)
        ld_std = std(ld_post)

        return (; acc_rate, rmsd, θ_mean, θ_std, ld_mean, ld_std, wall, ok=true)
    catch e
        wall = time() - t0
        return (; acc_rate=NaN, rmsd=NaN, θ_mean=fill(NaN,4), θ_std=fill(NaN,4),
                  ld_mean=NaN, ld_std=NaN, wall, ok=false)
    end
end

function main()
    Random.seed!(42)
    config = ODE_CONFIGS["LV"]
    times, y_obs, _, prob = generate_data(config; N=10, seed=42)

    # GPGM only
    gm_gp = setup_model(GPGM, config, times, y_obs, prob; anneal_length=1)
    vars = [:X, :θ]
    D = length(pack_param_vec(gm_gp, vars))
    println("GPGM D=$D, θ_true=[0.67, 1.33, 1.0, 1.0]")
    println()

    αs = [0.0, 0.1, 0.5, 1.0, 5.0]
    εs = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]

    @printf("%-6s  %-8s  %-7s  %-10s  %-40s  %-6s\n",
            "α", "ε", "accept", "RMSD", "θ_mean", "time")
    println("-"^90)

    for α in αs
        for ε in εs
            r = run_short_elmc(deepcopy(gm_gp), vars, "GPGM";
                step_size=ε, α=α, n_leapfrog=10, n_iter=500, n_warmup=200)
            θ_str = if r.ok
                @sprintf("[%.3f, %.3f, %.3f, %.3f]", r.θ_mean...)
            else
                "ERROR"
            end
            @printf("%-6.1f  %-8.4f  %-7.3f  %-10.4f  %-40s  %-6.1f\n",
                    α, ε,
                    isnan(r.acc_rate) ? -1.0 : r.acc_rate,
                    isnan(r.rmsd) ? -1.0 : r.rmsd,
                    θ_str,
                    r.wall)
        end
        println()
    end
end

main()
