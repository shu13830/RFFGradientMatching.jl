using DifferentialEquations
using KernelFunctions
using Distributions
using Random
using LinearAlgebra
using ForwardDiff
using AbstractMCMC

"""Create a small LV RFFGM for testing."""
function create_test_rffgm(; N=10, n_rff=50, seed=42)
    Random.seed!(seed)
    times = collect(range(0.0, 2.0, length=N))
    θ_true = [2.0, 1.0, 4.0, 1.0]
    prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), θ_true)
    sol = solve(prob, Tsit5(), saveat=times)
    y_obs = Array(sol) .+ 0.1 .* randn(size(sol))

    k = with_lengthscale(SqExponentialKernel(), 1.0)
    gm = RFFGM(times, y_obs, prob, "LV";
        k=k, state_noise_std=0.1, obs_noise_std=0.1, n_rff=n_rff)
    set_priortransform_on_θ!(gm, fill(Normal(0.0, 1.0), 4), fill(log, 4))
    return gm
end

"""Create a small LV GPGM for testing."""
function create_test_gpgm(; N=10, seed=42)
    Random.seed!(seed)
    times = collect(range(0.0, 2.0, length=N))
    θ_true = [2.0, 1.0, 4.0, 1.0]
    prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), θ_true)
    sol = solve(prob, Tsit5(), saveat=times)
    y_obs = Array(sol) .+ 0.1 .* randn(size(sol))

    k = with_lengthscale(SqExponentialKernel(), 1.0)
    gm = GPGM(times, y_obs, prob, "LV";
        k=k, state_noise_std=0.1, obs_noise_std=0.1)
    set_priortransform_on_θ!(gm, fill(Normal(0.0, 1.0), 4), fill(log, 4))
    return gm
end
