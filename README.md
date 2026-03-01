# RFFGradientMatching.jl

[![Build Status](https://github.com/shu13830/RFFGradientMatching.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shu13830/RFFGradientMatching.jl/actions/workflows/CI.yml?query=branch%3Amain)

![Julia](https://img.shields.io/badge/julia-v1.10%2B-blue)

**RFFGradientMatching.jl** is a Julia package for Bayesian inference of parameters in ordinary differential equation (ODE) models using gradient matching. It supports three methods under a unified `AbstractGM` API:

* **GPGM** — Gaussian Process Gradient Matching (function space)
* **RFFGM** — Random Fourier Features Gradient Matching (weight space)
* **MAGI** — Manifold-constrained GP Inference (Julia re-implementation of R CRAN `magi`)

Leverages DifferentialEquations.jl, KernelFunctions.jl, AbstractGPs.jl, GeneralizedRandomFourierFeatures.jl, BayesianLinearRegressors.jl, and MCMC samplers from AdvancedHMC.jl, AbstractMCMC.jl, and AdvancedMH.jl.

## Installation

```julia
using Pkg
Pkg.develop(url="https://github.com/shu13830/RFFGradientMatching.jl")
```

## Quick Start

```julia
using DifferentialEquations, KernelFunctions, Distributions
using AbstractMCMC, Random
using RFFGradientMatching

# 1. Simulate ODE (Lotka-Volterra) and add noise
Random.seed!(42)
θ_true = [2.0, 1.0, 4.0, 1.0]
prob = ODEProblem(lotkavolterrapredatorprey!, [5.0, 3.0], (0.0, 2.0), θ_true)
times = collect(range(0.0, 2.0, length=20))
sol = solve(prob, Tsit5(), saveat=times)
y_obs = Array(sol) .+ 0.5 .* randn(size(sol))

# 2. Initialize RFFGM
gm = RFFGM(times, y_obs, prob, "LV";
    k=SqExponentialKernel(), state_noise_std=1e-3, obs_noise_std=0.5, n_rff=100)

# 3. Set priors and transforms on ODE parameters
set_priortransform_on_θ!(gm, fill(Normal(0, 1), 4), fill(log, 4))

# 4. Optimize hyperparameters and latent states
optimize_ϕ_and_σ!(gm)
optimize_u!(gm)

# 5. Define sampling blocks
block_W  = HMCBlock(gm, [:W];    n_leapfrog=10, step_size=0.05)
block_Wθ = HMCBlock(gm, [:W,:θ]; n_leapfrog=10, step_size=0.01)
block_θ  = HMCBlock(gm, [:θ];    n_leapfrog=10, step_size=0.05)
bs = BlockedSampler([[block_W], [block_Wθ], [block_θ]], [0.4, 0.4, 0.2])

# 6. Run MCMC
chain, logdens = AbstractMCMC.sample(gm, bs, 1000; num_burnin=500, anneal=true)

# 7. Inspect results
θ_samples = get_θ(gm, chain)  # n_samples × n_θ matrix
```

## API Reference

### Key Types

* `AbstractGM` — abstract base type for gradient matching models
* `RFFGM <: AbstractGM` — weight-space (RFF) gradient matching
* `GPGM <: AbstractGM` — function-space (GP) gradient matching
* `MAGI <: AbstractGM` — manifold-constrained GP inference
* `ODEGrad` / `ODEGradFuns` — ODE right-hand side and Jacobians (via ForwardDiff)
* `GP` / `RFFGP` — function-space and weight-space GP models
* `BlockedSampler` — block MCMC sampler with probabilistic block selection
* `HMCBlock`, `NUTSBlock`, `HMCDABlock`, `ESSBlock`, `GESSBlock`, `RWMHBlock`, `StaticMHBlock` — sampling block types

### Core Functions

* `optimize_ϕ_and_σ!(gm)` — optimize kernel hyperparameters and observation noise
* `optimize_u!(gm)` — optimize latent states (inducing point values) via MAP
* `set_priortransform_on_θ!(gm, priors, transforms)` — set priors and transforms on ODE parameters
* `set_priortransform_on_σ!`, `set_priortransform_on_ϕ!`, `set_priortransform_on_γ!` — set priors on other parameters
* `AbstractMCMC.sample(gm, sampler, n; num_burnin, anneal)` — run blocked MCMC (returns `(chain, logdens)`)
* `build_rff_basis(k, input_dims, n_rff)` — build unified RFF basis for standard and generalized kernels
* `rff_approx_error(h, k, t)` — compute RFF kernel approximation error

### Accessors

* `get_θ(gm)`, `get_θ(gm, chain)` — ODE parameters (current / from chain)
* `get_W(gm)`, `get_W(gm, chain)` — RFF weights (RFFGM)
* `get_X(gm)`, `get_X(gm, chain)` — latent states (GPGM/MAGI)
* `get_σ(gm)`, `get_σ(gm, chain)` — observation noise
* `get_γ(gm)`, `get_γ(gm, chain)` — mismatch parameter
* `get_ϕ(gm)`, `get_ϕ(gm, chain)` — kernel hyperparameters
* `get_transformed_θ`, `get_transformed_σ`, etc. — parameters in unconstrained space
* `pack_param_vec`, `pack_param_dict`, `update_model_with_vec!`, `update_model_with_dict!` — parameter pack/unpack utilities

## License

Released under the MIT License. See `LICENSE` for details.
