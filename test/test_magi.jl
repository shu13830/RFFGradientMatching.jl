@testset "T9: MAGI" begin
    @testset "MAGI construction" begin
        gm = create_test_magi()
        @test gm isa MAGI
        @test gm isa RFFGradientMatching.AbstractGM
        @test length(gm.gp) == 2
        @test gm.odegrad.γ == 1e-3
        @test gm.β[1] == 1.0  # no annealing by default (anneal_length=1)
        @test gm.anneal_length == 1
        @test length(gm.odegrad.θ) == 4
    end

    @testset "MAGI construction with annealing" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        θ_true = [2.0, 1.0, 4.0, 1.0]
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), θ_true)
        sol = solve(prob, Tsit5(), saveat=times)
        y_obs = Array(sol) .+ 0.1 .* randn(size(sol))
        k = with_lengthscale(SqExponentialKernel(), 1.0)

        gm = MAGI(times, y_obs, prob, "LV";
            k=k, state_noise_std=0.1, obs_noise_std=0.1, anneal_length=100)
        @test gm.anneal_length == 100
        @test gm.β[1] == gm.β_schedule[1]
        @test length(gm.β_schedule) == 100
    end

    @testset "MAGI logpdf" begin
        gm = create_test_magi()

        # Scalar output
        lpd = ulogpdf(gm)
        @test isfinite(lpd)

        # Decomposed output
        lpd_dict = ulogpdf(gm, pack_param_dict(gm); merge_output=false)
        @test haskey(lpd_dict, :logpdf_x)
        @test haskey(lpd_dict, :logpdf_y)
        @test haskey(lpd_dict, :logpdf_θ)
        @test haskey(lpd_dict, :logpdf_σ)
        @test haskey(lpd_dict, :logpdf_ϕ)
        @test haskey(lpd_dict, :ulogpdf_e)
        @test haskey(lpd_dict, :logpdf_γ)  # MAGI now uses odegrad.γ (unified with GPGM/RFFGM)
        @test lpd ≈ sum(values(lpd_dict))

        # With sample_target
        lpd_st = ulogpdf(gm, [:X, :θ, :σ])
        @test isfinite(lpd_st)

        # From param_vec
        param_vec = pack_param_vec(gm, [:X, :θ, :σ])
        lpd_vec = ulogpdf(param_vec, gm, [:X, :θ, :σ])
        @test isfinite(lpd_vec)
    end

    @testset "MAGI individual logpdf components" begin
        gm = create_test_magi()
        X = get_X(gm)
        θ = get_θ(gm)
        σ = get_σ(gm)
        ϕ = get_ϕ(gm)

        @test isfinite(logpdf_x(gm, X))
        @test isfinite(logpdf_x(gm, X, ϕ))
        @test isfinite(logpdf_y(gm))
        @test isfinite(logpdf_θ(gm, θ))
        @test isfinite(logpdf_σ(gm, σ))
        @test isfinite(logpdf_ϕ(gm, ϕ))
        @test isfinite(ulogpdf_e(gm, X, θ))
        @test isfinite(ulogpdf_e(gm))
    end

    @testset "MAGI gradient" begin
        gm = create_test_magi()

        # Test gradient with different sample targets
        for sample_target in [[:X, :θ, :σ], [:X], [:θ], [:σ], [:X, :θ]]
            param_vec = pack_param_vec(gm, sample_target)
            grad = ∇ulogpdf(param_vec, gm, sample_target)
            @test length(grad) == length(param_vec)
            @test all(isfinite, grad)
        end
    end

    @testset "MAGI gradient validation (ForwardDiff)" begin
        gm = create_test_magi()

        @test validate_∇tx_logpdf_x(gm)
        @test validate_∇tx_logpdf_y(gm)
        @test validate_∇tx_ulogpdf_e(gm)
        @test validate_∇tθ_logpdf_θ(gm)
        @test validate_∇tθ_ulogpdf_e(gm)
        @test validate_∇tσ_logpdf_σ(gm)
        @test validate_∇tσ_logpdf_y(gm)

        # Overall gradient validation
        @test validate_∇ulogpdf(gm, [:X, :θ, :σ])
        @test validate_∇ulogpdf(gm, [:X, :θ])
        @test validate_∇ulogpdf(gm, [:X])
        @test validate_∇ulogpdf(gm, [:θ])
        @test validate_∇ulogpdf(gm, [:σ])
    end

    @testset "MAGI param pack/unpack" begin
        gm = create_test_magi()

        # pack_param_dict should include :X (not :W)
        pd = pack_param_dict(gm)
        @test haskey(pd, :X)
        @test !haskey(pd, :W)
        @test haskey(pd, :θ)
        @test haskey(pd, :γ)
        @test haskey(pd, :σ)
        @test haskey(pd, :ϕ)

        # Round-trip: dict → vec → dict
        sample_target = [:X, :θ, :σ]
        pv = pack_param_vec(gm, sample_target)
        pd2 = pack_param_dict_from_vec(gm, pv, sample_target)
        pv2 = pack_param_vec_from_dict(gm, pd2, sample_target)
        @test pv ≈ pv2
    end

    @testset "MAGI sampler smoke test" begin
        Random.seed!(42)
        gm = create_test_magi()

        block_X = HMCBlock(gm, [:X]; n_leapfrog=3, step_size=0.01)
        block_θ = HMCBlock(gm, [:θ]; n_leapfrog=3, step_size=0.01)
        bs = BlockedSampler([[block_X], [block_θ]], [0.5, 0.5])

        chain, logdens = AbstractMCMC.sample(gm, bs, 20;
            num_burnin=0, anneal=false)
        @test length(chain) == 20
        @test all(isfinite, logdens)

        # Extract parameters from chain
        θ_chain = get_θ(gm, chain)
        @test size(θ_chain) == (20, 4)
        @test all(isfinite, θ_chain)

        σ_chain = get_σ(gm, chain)
        @test size(σ_chain) == (20, 2)

        X_chain = get_X(gm, chain)
        @test size(X_chain, 1) == 20
    end
end
