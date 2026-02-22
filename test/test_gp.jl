@testset "T3-GP: GP" begin
    @testset "GP construction" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        gp = RFFGradientMatching.GP(times, randn(10), times, randn(10), 0.1, 0.5; k=k)
        @test length(gp.u) == 10
        @test length(gp.z) == 10
        @test size(gp.K) == (10, 10)
    end

    @testset "kernel matrices are symmetric" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        gp = RFFGradientMatching.GP(times, randn(10), times, randn(10), 0.1, 0.5; k=k)
        @test gp.K ≈ gp.K'
        @test gp.K″ ≈ gp.K″'
    end

    @testset "dfdt_mean and dfdt_cov" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        gp = RFFGradientMatching.GP(times, randn(10), times, randn(10), 0.1, 0.5; k=k)
        dm = dfdt_mean(gp)
        dc = dfdt_cov(gp)
        @test length(dm) == 10
        @test size(dc) == (10, 10)
    end
end
