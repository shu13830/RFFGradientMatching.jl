@testset "T2: ODEGrad" begin
    @testset "LV ODE evaluation" begin
        θ = [2.0, 1.0, 4.0, 1.0]
        x = [5.0, 3.0]
        du = zeros(2)
        lotkavolterrapredatorprey!(du, x, θ, 0.0)
        # ẋ₁ = a*x - b*x*y = 2*5 - 1*5*3 = -5
        # ẋ₂ = -c*y + d*x*y = -4*3 + 1*5*3 = 3
        @test du ≈ [-5.0, 3.0]
    end

    @testset "FitzHugh-Nagumo ODE evaluation" begin
        θ = [0.2, 0.2, 3.0]
        u = [1.0, 0.0]
        du = zeros(2)
        fitzhughnagumo!(du, u, θ, 0.0)
        # dV = θ1*(V - V³/3 + R) = 0.2*(1 - 1/3 + 0) = 0.2*2/3
        # dR = (1/θ1)*(V - θ2 + θ3*R) = 5*(1 - 0.2 + 0) = 4.0
        @test du[1] ≈ 0.2 * (1.0 - 1.0/3.0 + 0.0) atol=1e-10
        @test du[2] ≈ (1.0/0.2) * (1.0 - 0.2 + 3.0*0.0) atol=1e-10
    end

    @testset "ODEGradFuns construction" begin
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), [2.0, 1.0, 4.0, 1.0])
        ogf = ODEGradFuns(prob, "LV")
        @test ogf.probname == "LV"
        x = [5.0, 3.0]
        θ = [2.0, 1.0, 4.0, 1.0]
        dx = ogf.ẋ(x, θ)
        @test dx ≈ [-5.0, 3.0]
    end

    @testset "LV Jacobian dẋdx" begin
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), [2.0, 1.0, 4.0, 1.0])
        ogf = ODEGradFuns(prob, "LV")
        x = [5.0, 3.0]
        θ = [2.0, 1.0, 4.0, 1.0]
        J = ogf.dẋdx(x, θ)
        # ∂f₁/∂x₁ = a - b*y = 2 - 3 = -1,  ∂f₁/∂x₂ = -b*x = -5
        # ∂f₂/∂x₁ = d*y = 3,                  ∂f₂/∂x₂ = -c + d*x = -4 + 5 = 1
        @test J ≈ [-1.0 -5.0; 3.0 1.0]
    end

    @testset "LV Jacobian dẋdθ" begin
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), [2.0, 1.0, 4.0, 1.0])
        ogf = ODEGradFuns(prob, "LV")
        x = [5.0, 3.0]
        θ = [2.0, 1.0, 4.0, 1.0]
        Jθ = ogf.dẋdθ(x, θ)
        # ∂f₁/∂a = x = 5,  ∂f₁/∂b = -xy = -15,  ∂f₁/∂c = 0,  ∂f₁/∂d = 0
        # ∂f₂/∂a = 0,       ∂f₂/∂b = 0,           ∂f₂/∂c = -y = -3, ∂f₂/∂d = xy = 15
        @test Jθ ≈ [5.0 -15.0 0.0 0.0; 0.0 0.0 -3.0 15.0]
    end

    @testset "ODEGrad construction and accessors" begin
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), [2.0, 1.0, 4.0, 1.0])
        y_obs = randn(2, 10)
        og = ODEGrad(y_obs, prob, "LV")
        @test n_state_types(og) == 2
        @test n_times(og) == 10
        @test length(og.θ) == 4
    end

    @testset "eval_ẋ on ODEGrad" begin
        prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), [2.0, 1.0, 4.0, 1.0])
        y_obs = [5.0 4.0; 3.0 2.0]  # K=2, N=2
        og = ODEGrad(y_obs, prob, "LV")
        og.θ = [2.0, 1.0, 4.0, 1.0]
        ẋ = eval_ẋ(og, og.X, og.θ)
        @test size(ẋ) == (2, 2)  # K × N
    end
end
