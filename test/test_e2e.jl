@testset "T7: End-to-End" begin
    @testset "LV RFFGM pipeline smoke test" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        θ_true = [2.0, 1.0, 4.0, 1.0]
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), θ_true)
        sol = solve(prob, Tsit5(), saveat=times)
        y_obs = Array(sol) .+ 0.1 .* randn(size(sol))

        k = with_lengthscale(SqExponentialKernel(), 1.0)
        anneal_len = 10
        gm = RFFGM(times, y_obs, prob, "LV";
            k=k, state_noise_std=0.1, obs_noise_std=0.1, n_rff=50,
            anneal_length=anneal_len)

        set_priortransform_on_θ!(gm, fill(Normal(0.0, 1.0), 4), fill(log, 4))

        # Build sampler
        block_W = HMCBlock(gm, [:W]; n_leapfrog=3, step_size=0.01)
        block_θ = HMCBlock(gm, [:θ]; n_leapfrog=3, step_size=0.01)
        bs = BlockedSampler([[block_W], [block_θ]], [0.5, 0.5])

        # Run short chain (anneal_length=10, so num_burnin must be >= 10)
        num_samples = 20
        num_burnin = anneal_len
        chain, logdens = AbstractMCMC.sample(gm, bs, num_samples;
            num_burnin=num_burnin, anneal=true)
        @test length(chain) == num_samples
        @test length(logdens) == num_samples
        @test all(isfinite, logdens)

        # Check θ posterior samples are finite
        θ_final = get_θ(gm)
        @test all(isfinite, θ_final)

        # Extract θ from chain
        θ_chain = get_θ(gm, chain)
        @test size(θ_chain) == (num_samples, 4)
        @test all(isfinite, θ_chain)
    end
end
