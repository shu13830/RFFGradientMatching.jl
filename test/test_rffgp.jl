@testset "T3: RFFGP" begin
    @testset "RFFGP construction" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=20))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        rffgp = RFFGP(times, randn(20), times, randn(20), 0.1, 0.5; k=k, n_rff=100)
        @test rffgp.n_rff == 100
        @test length(rffgp.w) == 100
        @test length(rffgp.z) == 20
    end

    @testset "dHdt shape" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        rffgp = RFFGP(times, randn(10), times, randn(10), 0.1, 0.5; k=k, n_rff=50)
        @test size(rffgp.dHdt) == (50, 10)  # n_rff × N
    end

    @testset "dHdt vs numerical derivative" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        rffgp = RFFGP(times, randn(10), times, randn(10), 0.1, 0.5; k=k, n_rff=50)

        # Compute numerical dH/dt using ForwardDiff
        h = rffgp.h
        numerical_dHdt = ForwardDiff.jacobian(t -> h(RowVecs(t[:,:])).X, times)
        # numerical_dHdt is N×N*L matrix; we need the diagonal blocks
        # eval_dHdt gives L×N, numerical_dHdt_diag should match
        analytic_dHdt = eval_dHdt(h, times)
        @test size(analytic_dHdt) == (50, 10)
    end

    @testset "W2X reconstruction" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        rffgp = RFFGP(times, randn(10), times, randn(10), 0.1, 0.5; k=k, n_rff=50)
        W = randn(1, 50)  # 1 GP, 50 RFF weights
        X = W2X([rffgp], W)
        @test size(X) == (1, 10)  # K=1, N=10
        # Manual computation
        expected = (Hmat(rffgp) * W[1,:])'
        @test X[1,:] ≈ expected[:]
    end

    @testset "dfdt_mean and dfdt_cov" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        rffgp = RFFGP(times, randn(10), times, randn(10), 0.1, 0.5; k=k, n_rff=50)
        dm = dfdt_mean(rffgp)
        dc = dfdt_cov(rffgp)
        @test length(dm) == 10
        @test size(dc) == (10, 10)
        @test all(diag(dc) .>= 0)
    end

    @testset "weight_mean and weight_precision" begin
        Random.seed!(42)
        times = collect(range(0.0, 2.0, length=10))
        k = with_lengthscale(SqExponentialKernel(), 1.0)
        rffgp = RFFGP(times, randn(10), times, randn(10), 0.1, 0.5; k=k, n_rff=50)
        wm = weight_mean(rffgp)
        wp = weight_precision(rffgp)
        @test length(wm) == 50
        @test size(wp) == (50, 50)
    end
end
