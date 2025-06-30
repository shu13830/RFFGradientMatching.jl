#=
# ***ODE Parameter Estimation with RFFGM***
---

=#

using ArgParse
using YAML

using Random
using LinearAlgebra
using Distributions

using DifferentialEquations
using KernelFunctions
using AbstractMCMC
using MCMCChains

using Plots

using Revise
using RFFGradientMatching

# ### Load the configuration
## Execute argparse only when running the script directly
if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings()
    @add_arg_table s begin
        "--ode"
            help = "ODE model name"
            arg_type = String
    end
    ## Parse the command line arguments
    parsed_args = parse_args(s)
    ode_name = get(parsed_args, "ode", "PredatorPrey")
    config = YAML.load_file("$(dirname(@__FILE__))/config.yaml")[ode_name]
else
    ode_name = "PredatorPrey"  # Edit here to change ODE when to use notebook
    config = YAML.load_file("./config.yaml")[ode_name]
end

true_params       = config["true_parameters"]
u0                = config["initial_conditions"]
tspan             = (config["tspan"][1], config["tspan"][2])
obs_time          = config["obs_time"] .|> float
noise_std         = config["noise_std"]
mcmc_iterations   = config["mcmc_iterations"]
burn_in           = config["burn_in"]
state_noise_std   = config["state_noise_std"]
kernel_type       = config["gp_kernel"]

## Select the ODE
ode_func = 
    ode_name == "PredatorPrey" ? RFFGradientMatching.lotkavolterrapredatorprey! :
    ode_name == "Competition" ? RFFGradientMatching.lotkavolterracompetition! :
    ode_name == "SIR" ? RFFGradientMatching.sir! :
    ode_name == "FitzHughNagumo" ? RFFGradientMatching.fitzhughnagumo! :
    ode_name == "Lorenz96" ? RFFGradientMatching.lorenz96! :
    ode_name == "SignalTransduction" ? RFFGradientMatching.signaltransductioncascade! :
    ode_name == "PIF" ? RFFGradientMatching.pif4and5! : nothing

# ## ***1. ODE Simulation and Data Generation***
# Define the ODE problem and solve it numerically
## Set the true parameters and initial conditions
prob = ODEProblem(ode_func, u0, tspan, true_params)
sol = solve(prob, Tsit5(), saveat=0.001)
true_states = Array(solve(prob, Tsit5(), saveat=obs_time))

# Generate observations by adding noise to the true states
y_obs = true_states .+ noise_std * randn(size(true_states))

# plot the true ODE trajectory and the noisy observations
# plot(sol.t, hcat(sol.u...)', c=[:blue :red], labels=["prey" "predator"])
#     scatter!(obs_time, y_obs', c=[:blue :red], labels=false, title=ode_name, size=(500,200))
plot(sol.t, hcat(sol.u...)')
scatter!(obs_time, y_obs', labels=false, title=ode_name, size=(500,200))

# ## ***2. Construct and Initialize the GPGM Model***
## Set the GP kernel (here we use an RBF kernel: SqExponentialKernel)
if kernel_type == "RBF"
    α² = 1.0
    𝓁 = 1.0
    k = α² * KernelFunctions.with_lengthscale(KernelFunctions.RBFKernel(), 𝓁)
elseif kernel_type == "Matern52"
    α² = 1.0
    𝓁 = 1.0
    k = α² * KernelFunctions.with_lengthscale(KernelFunctions.Matern52Kernel(), 𝓁)
elseif kernel_type == "Sigmoid"
    α² = 1.0
    a = 0.0
    b = 1.0
    k = α² * SigmoidKernel(b, a)
else
    error("Invalid kernel type: $kernel_type")
end

## Create an instance of the RFFGM model
gm = RFFGM(obs_time, y_obs, prob, ode_name; k=k, state_noise_std=1e-4, obs_noise_std=noise_std)

## Set priors and transforms on ODE parameters (θ)
priors = [Normal(0., 1.) for _ in 1:length(prob.p)]
transforms = [log for _ in 1:length(prob.p)]
# transforms = [Bijectors.Logit(0, 10) for _ in 1:length(prob.p)]
set_priortransform_on_θ!(gm, priors, transforms)


## Optimize the kernel hyperparameters (ϕ), std of observation noise (σ) and the latent states (u)
optimize_ϕ_and_σ!(gm)
optimize_u!(gm)

# ## ***3. MCMC Sampling Using a Blocked Sampler***
# Here, we prepare HMC blocks to sample the weight variables (W) and the ODE parameters (θ) separately
## Block 1: Sampling the weight variable W using HMC
block_W = HMCBlock(gm, [:W]; n_leapfrog=10, step_size=0.05, metric=:diag)

## Block 2: Jointly Sampling the weight variable W and ODE parameters θ using HMC
block_Wθ = HMCBlock(gm, [:W, :θ]; n_leapfrog=10, step_size=0.01, metric=:diag)

## Block 3: Sampling ODE parameters θ using HMC
block_θ = HMCBlock(gm, [:θ]; n_leapfrog=10, step_size=0.05, metric=:diag)

## Group the blocks and set their selection probabilities
blocks = [[block_W], [block_Wθ], [block_θ]]
probs = [0.4, 0.4, 0.2]

## Create an instance of the BlockedSampler
bs = BlockedSampler(blocks, probs)

# Create the initial parameter dictionary from the model state
init_params = pack_param_dict(gm)

# Run MCMC sampling using the blocked sampler
n_samples = 1000  # number of samples (example: 1000 iterations)
num_burnin = 500  # number of burn-in iterations
chain = AbstractMCMC.sample(gm, bs, n_samples; num_burnin=num_burnin);

# After sampling, check the final sample of the ODE parameters (θ)
get_θ(gm)

# ## ***4. Plot the Results***
# For example, plot the trace of the ODE parameters θ
θ_trace = gm.odegrad.θ
plot(θ_trace, label="θ", title="Trace Plot of ODE Parameter θ", xlabel="MCMC Iteration", ylabel="θ Value")
