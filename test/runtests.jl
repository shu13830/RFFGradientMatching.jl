using RFFGradientMatching
using Test
using Random
using ForwardDiff
using LinearAlgebra
using DifferentialEquations
using KernelFunctions
using Distributions
using AbstractMCMC

# Import internal functions needed for testing
import RFFGradientMatching:
    # PriorTransformation helpers
    calc_tvar, calc_var, rand_tvar, rand_var,
    # ODE functions
    lotkavolterrapredatorprey!, fitzhughnagumo!,
    # ODEGrad
    ODEGradFuns, ODEGrad, n_times, n_state_types, eval_ẋ,
    # RFFGP/GP helpers
    eval_dHdt, Hmat, W2X, dfdt_mean, dfdt_cov,
    weight_mean, weight_precision, only_params, reconstruct_kernel,
    # Likelihood
    ulogpdf, logpdf_x, logpdf_y, ulogpdf_e, logpdf_θ, logpdf_γ, logpdf_σ, logpdf_ϕ,
    # Gradient
    ∇ulogpdf,
    # Utils
    pack_param_dict, pack_param_vec, pack_param_dict_from_vec,
    pack_param_vec_from_dict, update_model_with_dict!, update_model_with_vec!,
    get_standardized_Y, calc_standardized_Y, calc_destandardized_X,
    get_destandardized_X, calc_X, calc_transformed_X, calc_θ, calc_γ, calc_σ, calc_ϕ,
    # Validation
    validate_∇tx_logpdf_x, validate_∇tx_logpdf_y, validate_∇tx_ulogpdf_e,
    validate_∇w_logpdf_x, validate_∇w_logpdf_y, validate_∇w_ulogpdf_e,
    validate_∇tθ_logpdf_θ, validate_∇tθ_ulogpdf_e,
    validate_∇tγ_logpdf_γ, validate_∇tγ_ulogpdf_e,
    validate_∇tσ_logpdf_σ, validate_∇tσ_logpdf_y,
    validate_∇tϕ_logpdf_ϕ, validate_∇tϕ_logpdf_x, validate_∇tϕ_logpdf_y, validate_∇tϕ_ulogpdf_e,
    validate_∇y_logpdf_y, validate_∇ulogpdf,
    # AbstractGM
    AbstractGM,
    # Sampler
    BlockedSamplerState

include("test_helpers.jl")

@testset "RFFGradientMatching.jl" begin
    include("test_priortransformation.jl")
    include("test_odegrad.jl")
    include("test_rffgp.jl")
    include("test_gp.jl")
    include("test_likelihood.jl")
    include("test_gradient.jl")
    include("test_sampler.jl")
    include("test_e2e.jl")
    include("test_magi.jl")
    include("test_generalized_rff.jl")
end
