@testset "T8: GeneralizedRandomFourierFeatures integration" begin
    @testset "GeneralizedCauchy kernel evaluation" begin
        using GeneralizedRandomFourierFeatures
        k = GeneralizedCauchyKernel(1.5, 1.5)
        @test k(0.0, 0.0) > 0
        @test k(0.0, 1.0) >= 0
    end

    @testset "build_rff_basis with SqExponentialKernel" begin
        using Random
        Random.seed!(42)
        k = 1.0 * with_lengthscale(SqExponentialKernel(), 1.0)
        h = build_rff_basis(k, 1, 100)
        @test h isa GeneralizedRandomFourierFeatures.RFFBasis
        @test size(h.ω) == (1, 100)
        @test h.inner_weights ≈ 1.0
    end

    @testset "build_rff_basis with GeneralizedCauchyKernel" begin
        using Random, GeneralizedRandomFourierFeatures
        Random.seed!(42)
        k = 1.0 * with_lengthscale(GeneralizedCauchyKernel(1.5, 1.5), 0.5)
        h = build_rff_basis(k, 1, 100)
        @test h isa GeneralizedRandomFourierFeatures.RFFBasis
        @test size(h.ω) == (1, 100)
        @test h.inner_weights ≈ 0.5  # lengthscale
    end

    @testset "build_rff_basis with ExponentialPowerKernel" begin
        using Random, GeneralizedRandomFourierFeatures
        Random.seed!(42)
        k = 1.0 * with_lengthscale(ExponentialPowerKernel(γ=1.0), 1.0)
        h = build_rff_basis(k, 1, 100)
        @test h isa GeneralizedRandomFourierFeatures.RFFBasis
        @test size(h.ω) == (1, 100)
    end

    @testset "build_rff_basis with Matern52Kernel (via GenRFF)" begin
        using Random
        Random.seed!(42)
        k = 1.0 * with_lengthscale(Matern52Kernel(), 1.0)
        h = build_rff_basis(k, 1, 100)
        @test h isa GeneralizedRandomFourierFeatures.RFFBasis
        @test size(h.ω) == (1, 100)
    end

    @testset "RFFGM with GeneralizedCauchyKernel" begin
        using Random, GeneralizedRandomFourierFeatures
        Random.seed!(42)
        import RFFGradientMatching: lotkavolterrapredatorprey!
        times = collect(range(0.0, 2.0, length=10))
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), [2.0, 1.0, 4.0, 1.0])
        sol = solve(prob, Tsit5(), saveat=times)
        y_obs = Array(sol) .+ 0.1 .* randn(size(Array(sol)))
        k = 1.0 * with_lengthscale(GeneralizedCauchyKernel(1.5, 1.5), 1.0)
        gm = RFFGM(times, y_obs, prob, "LV";
            k=k, state_noise_std=0.1, obs_noise_std=0.5, n_rff=50)
        @test gm isa RFFGM
        @test length(gm.gp) == 2
    end

    @testset "rff_approx_error" begin
        using Random
        Random.seed!(42)
        k = 1.0 * with_lengthscale(SqExponentialKernel(), 1.0)
        h = build_rff_basis(k, 1, 200)
        t = collect(range(0.0, 2.0, length=20))
        err = rff_approx_error(h, k, t)
        @test err >= 0.0
        @test err < 1.0
    end
end
