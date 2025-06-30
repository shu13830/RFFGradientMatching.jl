using ArgParse
using DifferentialEquations
using Random
using Plots
using KernelFunctions
using YAML

using RandomFeatureGradientMatching

# Set up the argument parser settings
s = ArgParseSettings()
@add_arg_table s begin
    "--ode"
        help = "ODE model name"
        arg_type = String
end
# Parse the command line arguments
parsed_args = parse_args(s)
ode_name = get(parsed_args, "ode", "LotkaVolterra")

println("This file is: ", dirname(@__FILE__))
config = YAML.load_file("$(dirname(@__FILE__))/config.yaml")[ode_name]
true_params       = config["true_parameters"]
u0                = config["initial_conditions"]
tspan             = (config["tspan"][1], config["tspan"][2])
num_obs           = config["num_obs"]
noise_std         = config["noise_std"]
mcmc_iterations   = config["mcmc_iterations"]
burn_in           = config["burn_in"]
state_noise_std   = config["state_noise_std"]
kernel_type       = config["gp_kernel"]

# ==== 1. ODE Simulation and Data Generation ====
# Select the ODE
ode_func = 
    ode_name == "PredatorPrey" ? RandomFeatureGradientMatching.lotkavolterrapredatorprey! :
    ode_name == "Competition" ? RandomFeatureGradientMatching.lotkavolterracompetition! :
    ode_name == "SIR" ? RandomFeatureGradientMatching.sir! :
    ode_name == "FitzHughNagumo" ? RandomFeatureGradientMatching.fitzhughnagumo! :
    ode_name == "Lorenz96" ? RandomFeatureGradientMatching.lorenz96! :
    ode_name == "SignalTransduction" ? RandomFeatureGradientMatching.signaltransductioncascade! :
    ode_name == "PIF" ? RandomFeatureGradientMatching.pif4and5! : nothing

# Set the true parameters and initial conditions
true_params = [2.0, 1.0, 4.0, 1.0]      # [α, β, γ, δ]
u0 = [5.0, 3.0]
tspan = (0.0, 2.0)
num_obs = 20
noise_std = 0.1

# Define the ODE problem and solve it numerically
prob = ODEProblem(ode_func, u0, tspan, true_params)
t_obs = collect(range(tspan[1], tspan[2], length=num_obs))
sol = solve(prob, Tsit5(), saveat=t_obs)
true_states = Array(sol)
# Generate observations by adding noise to the true states
y_obs = true_states .+ noise_std * randn(size(true_states))


# ==== 2. Construct and Initialize the GPGM Model ====
# Set the GP kernel (here we use an RBF kernel: SqExponentialKernel)
k = SqExponentialKernel()

# Create an instance of the GPGM model
gm = GPGM(t_obs, y_obs, prob, ode_name; k=k, state_noise_std=1e-4, obs_noise_std=noise_std)

# Optimize the GP hyperparameters (ϕ) and the latent states (u)
optimize_ϕ_and_σ!(gm)
optimize_u!(gm)

# ==== 3. MCMC Sampling Using a Blocked Sampler ====
# Here, we prepare HMC blocks to sample the state variables (X) and the ODE parameters (θ) separately

# Block 1: Sampling the state variable X using HMC
block_X = HMCBlock(gm, [:X]; n_leapfrog=10, step_size=0.05, metric=:diag, n_iter=1)
# Block 2: Sampling the ODE parameters θ using HMC
block_θ = HMCBlock(gm, [:θ]; n_leapfrog=10, step_size=0.01, metric=:diag, n_iter=1)

# Group the blocks and set their selection probabilities
blocks = [[block_X], [block_θ]]
probs = [0.5, 0.5]  # each block is selected with 50% probability

# Create an instance of the BlockedSampler
bs = BlockedSampler(blocks, probs)

# Create the initial parameter dictionary from the model state
init_params = pack_param_dict(gm)

# Run MCMC sampling using the blocked sampler
n_samples = 1000  # number of samples (example: 1000 iterations)
chain = AbstractMCMC.sample(gm, bs, n_samples; init_params=init_params)

# After sampling, update the model with the final sample from the chain
update_model_with_dict!(gm, chain[end])

# ==== 4. Plot the Results ====
# For example, plot the trace of the ODE parameters θ
θ_trace = gm.odegrad.θ
plot(θ_trace, label="θ", title="Trace Plot of ODE Parameter θ", xlabel="MCMC Iteration", ylabel="θ Value")
