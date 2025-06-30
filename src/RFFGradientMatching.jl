module RFFGradientMatching

# Write your package code here.
using AbstractGPs
using AdvancedHMC
using AbstractMCMC
using AdvancedMH
using ArrayInterface
using BayesianLinearRegressors
using Bijectors
using DifferentialEquations
using Distributions
using EllipticalSliceSampling
using FillArrays
using ForwardDiff
using GPLikelihoods
using HypergeometricFunctions
using KernelFunctions
using LinearAlgebra
using LogDensityProblems
using MCMCChains
using MCMCDiagnosticTools
using MCMCTempering
using Optim
using Plots
using ProgressMeter
using Random
using RandomFourierFeatures
using SpecialFunctions
using StatsBase
using StatsFuns
using StatsPlots
using Turing
using UnPack
using UnicodePlots

import GPLikelihoods: AbstractLink
import RandomFourierFeatures: RFFBasis, BasisFunctionRegressor, BayesianLinearRegressor

include("priortransformation.jl")
include("ode.jl")
include("odegrad.jl")
include("kernel/sigmoid.jl")
include("models/gp.jl")
include("models/rffgp.jl")
include("gradmatch.jl")

include("likelihood/gradmatch_logpdf.jl")
include("likelihood/gradmatch_gradlogpdf.jl")

include("sampler/gess/abstractmcmc.jl")
include("sampler/compwisemh/abstractmcmc.jl")
include("sampler/blockedsampler/abstractmcmc.jl")
include("sampler/blockedsampler/blocks.jl")
include("sampler/blockedsampler/plot.jl")
include("sampler/blockedsampler/hmc_utils.jl")

include("models/agm.jl")
include("models/fgpgm.jl")
include("models/vgm.jl")

include("utils.jl")
include("validate.jl")
include("optimize.jl")
include("initialize.jl")

export PriorTransformation

export RFFGM, GPGM, RFFGP, GP
export ODEGrad, ODEGradFuns
export set_priortransform_on_θ!, set_priortransform_on_σ!, set_priortransform_on_ϕ!, set_priortransform_on_γ!
export get_θ, get_σ, get_X, get_W, get_γ, get_ϕ
export get_destandardized_X, get_y_std
export get_transformed_X, get_transformed_θ, get_transformed_σ, get_transformed_γ,  get_transformed_ϕ

export BlockedSampler
export HMCBlock, NUTSBlock, HMCDABlock, RWMHBlock, StaticMHBlock, ESSBlock, GESSBlock
export pack_param_dict, pack_param_dict_from_vec, update_param_dict_from_vec!,
    pack_param_vec_from_dict, pack_param_vec, 
    update_model_with_dict!, update_model_with_vec!

export FGPGM, AGM, VGM
export optimize_ϕ_and_σ!, optimize_u!
export initialize_vars!

export SigmoidKernel

end
