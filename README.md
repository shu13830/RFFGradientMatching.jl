# RFFGradientMatching.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shu13830.github.io/RFFGradientMatching.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shu13830.github.io/RFFGradientMatching.jl/dev/)
[![Build Status](https://github.com/shu13830/RFFGradientMatching.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shu13830/RFFGradientMatching.jl/actions/workflows/CI.yml?query=branch%3Amain)


![Stable](https://img.shields.io/badge/status-stable-green) ![Julia](https://img.shields.io/badge/julia-v1.9%2B-blue)

**RFFGradientMatching** is a Julia package for Bayesian inference of parameters in ordinary differential equation (ODE) models using gradient matching. It supports both:

* **Gaussian Process Gradient Matching** (GPGM)
* **Random Fourier Features Gradient Matching** (RFGM)

Leverages DifferentialEquations.jl, KernelFunctions.jl, AbstractGPs.jl, RandomFourierFeatures.jl, and a suite of MCMC samplers from AdvancedHMC.jl, AbstractMCMC.jl, and AdvancedMH.jl.

## Installation

```julia
] add RFFGradientMatching
```

## Usage

### 1. Command‑line Script for ODE Parameter Estimation

RFFGradientMatching provides an example CLI script (`estimate_ode.jl`) that uses ArgParse and a `config.yaml` to specify problem settings:

```bash
julia estimate_ode.jl --ode PredatorPrey
```

Inside the script:

```julia
using ArgParse, YAML, DifferentialEquations, KernelFunctions
using AbstractMCMC, MCMCChains, Plots
using RFFGradientMatching

# Load config
parsed = parse_args(s)  # --ode name
cfg = YAML.load_file("config.yaml")[parsed["ode"]]

# Unpack settings
tspan = tuple(cfg["tspan"]...)
obs_time = cfg["obs_time"] |> float
u0, p_true = cfg["initial_conditions"], cfg["true_parameters"]
noise_std, state_noise_std = cfg["noise_std"], cfg["state_noise_std"]
kern = cfg["gp_kernel"] == "RBF" ? SqExponentialKernel() : Matern52Kernel()

# 1. Simulate ODE and add noise
prob = ODEProblem(RFFGradientMatching.lotkavolterrapredatorprey!, u0, tspan, p_true)
sol  = solve(prob, Tsit5(), saveat=obs_time)
y_obs = Array(sol) .+ noise_std * randn(size(sol))

# 2. Initialize RFFGM
gm = RFFGM(obs_time, y_obs, prob, parsed["ode"]; k=kern,
            state_noise_std=state_noise_std,
            obs_noise_std=noise_std)

# 3. Set priors and transforms
set_priortransform_on_θ!(gm,
    fill(Normal(0,1), length(p_true)),
    fill(log, length(p_true)))

# 4. Optimize hyperparameters and latent states
optimize_ϕ_and_σ!(gm)
optimize_u!(gm)

# 5. Define sampling blocks
block_W   = HMCBlock(gm, [:W];   n_leapfrog=10, step_size=0.05)
block_Wθ  = HMCBlock(gm, [:W,:θ]; n_leapfrog=10, step_size=0.01)
block_θ   = HMCBlock(gm, [:θ];   n_leapfrog=10, step_size=0.05)
bs = BlockedSampler([[block_W], [block_Wθ], [block_θ]], [0.4,0.4,0.2])

# 6. Run MCMC
chain = AbstractMCMC.sample(gm, bs, 1000; num_burnin=500)

# 7. Inspect and plot results
theta_samples = get_θ(gm)
plot(theta_samples, label="θ trace")
```

Be sure to include a `config.yaml` alongside the script:

```yaml
PredatorPrey:
  true_parameters: [1.5, 1.0, 3.0, 1.0]
  initial_conditions: [1.0, 1.0]
  tspan: [0.0, 10.0]
  obs_time: [0.0,0.2,...,10.0]
  noise_std: 0.05
  state_noise_std: 1e-4
  gp_kernel: RBF
  mcmc_iterations: 1000
  burn_in: 500
```

### 2. Interactive API Example

```julia
using DifferentialEquations, KernelFunctions, RFFGradientMatching

# Define and simulate ODE
prob = ODEProblem(lotkavolterrapredatorprey!, [1.0,1.0], (0.0,10.0), [1.5,1.0,3.0,1.0])
ts, y = range(0,10, length=50), Array(solve(prob, Tsit5(), saveat=ts))

gm = GPGM(ts, y, prob, "LV"; k=Matern52Kernel(), state_noise_std=0.1, obs_noise_std=0.05)
set_priortransform_on_θ!(gm, fill(Gamma(2,2),4), fill(log,4))
optimize_ϕ_and_σ!(gm); optimize_u!(gm)
chain, _ = AGM(ts, y, prob, "LV"; k=Matern52Kernel())
samples, logdens = sample(gm, bs, 2000; num_burnin=500)
```

## API Reference

### Key Types

* `ODEGrad` – wraps an ODEProblem and gradient utilities
* `GP` / `RFFGP` – GP and RFF approximations with gradient functions
* `GPGM` / `RFGM` – gradient matching models
* `HMCBlock`, `NUTSBlock`, `StaticMHBlock`, `ESSBlock`, etc. – for constructing `BlockedSampler`

### Core Functions

* `optimize_ϕ_and_σ!(gm)`, `optimize_u!(gm)` – initialize hyperparameters and latent states
* `set_priortransform_on_θ!(gm, priors, transforms)` – apply priors/transforms on ODE parameters
* `sample(gm, sampler, n_samples; num_burnin)` – run MCMC
* Accessors: `get_θ`, `get_γ`, `get_σ`, `get_ϕ`, `plot` methods for diagnostics

## License

Released under the MIT License. See `LICENSE` for details.
