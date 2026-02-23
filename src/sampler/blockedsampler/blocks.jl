"""Sample Block for Hamiltonian Monte Carlo"""
mutable struct HMCBlock <: AbstractSampleBlock
    vars::Vector{Symbol}
    h::AdvancedHMC.Hamiltonian
    sampler::AdvancedHMC.HMCSampler
    type::Symbol
    D::Int
    n::Int
    accept_counter::Int
    reject_counter::Int
end

"""Sample Block for Metropolis-Hastings"""
struct MHBlock <: AbstractSampleBlock
    vars::Vector{Symbol}
    model::AdvancedMH.DensityModel
    sampler::MetropolisHastings
    D::Int
    n::Int
end

struct ComponentWiseMHBlock <: AbstractSampleBlock
    vars::Vector{Symbol}
    model::AdvancedMH.DensityModel
    sampler::ComponentWiseMH
    D::Int
    n::Int
end

"""Sample Block for Elliptical Slice Sampling"""
struct ESSBlock <: AbstractSampleBlock
    vars::Vector{Symbol}
    model::EllipticalSliceSampling.ESSModel
    sampler::EllipticalSliceSampling.ESS
    D::Int
    n::Int
end

"""Sample Block for Generalized Elliptical Slice Sampling"""
struct GESSBlock <: AbstractSampleBlock
    vars::Vector{Symbol}
    model::EllipticalSliceSampling.ESSModel
    sampler::GESS
    D::Int
    n::Int
end

Base.show(io::IO, blk::HMCBlock) = print(
    io, "HMCBlock(
    D=$(blk.D),
    n=$(blk.n),
    vars=$(blk.vars),
    metric=$(blk.h.metric),
    kinetic=$(blk.h.kinetic),
    adaptor=$(blk.sampler.adaptor),
    integrator=$(blk.sampler.κ.τ.integrator),
    termination_criterion=$(blk.sampler.κ.τ.termination_criterion),
)")

Base.show(io::IO, blk::MHBlock) = print(
    io, "MHBlock(
    D=$(blk.D), 
    n=$(blk.n),
    vars=$(blk.vars), 
    proposal=$(typeof(blk.sampler.proposal))
)")

Base.show(io::IO, blk::ESSBlock) = print(
    io, "ESSBlock(
    D=$(blk.D), 
    n=$(blk.n),
    vars=$(blk.vars), 
    prior=$(typeof(blk.model.prior))
)")

Base.show(io::IO, blk::GESSBlock) = print(
    io, "GESSBlock(
    D=$(blk.D), 
    n=$(blk.n),
    vars=$(blk.vars), 
    prior=$(typeof(blk.sampler.prior))
)")

function HMCBlock(mod::AbstractGM,
    vars::Vector{Symbol}; 
    n_leapfrog::Int,
    step_size::Float64=0.1,
    metric::Symbol=:diag,  # {:diag, :dense, :unit}
)
    ℓπ = (p) -> ulogpdf(p, mod, vars)
    ∂ℓπ∂θ = (p) -> (ℓπ(p), ∇ulogpdf(p, mod, vars))
    D = length(pack_param_vec(mod, vars))
    metric = init_metric(metric, D)
    hamiltonian = AdvancedHMC.Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    integrator = Leapfrog(step_size)
    adaptor = AdvancedHMC.NoAdaptation()
    kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{EndPointTS}(integrator, AdvancedHMC.FixedNSteps(n_leapfrog)))
    sampler = AdvancedHMC.HMCSampler(kernel, metric, adaptor)
    HMCBlock(vars, hamiltonian, sampler, :HMC, D, 1, 0, 0)
end

function NUTSBlock(mod::AbstractGM,
    vars::Vector{Symbol};
    target_accept::Float64=0.65,
    metric::Symbol=:diag,  # {:diag, :dense, :unit}
    integrator::Union{Nothing,TIntegrator}=nothing,
    adoptor::Union{Nothing,TAdoptor}=nothing
) where {
    TAdoptor<:AdvancedHMC.Adaptation.AbstractAdaptor,
    TIntegrator<:AdvancedHMC.AbstractIntegrator
}
    ℓπ = (p) -> ulogpdf(p, mod, vars)
    ∂ℓπ∂θ = (p) -> (ℓπ(p), ∇ulogpdf(p, mod, vars))
    D = length(pack_param_vec(mod, vars))
    metric = init_metric(metric, D)
    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    initial_ϵ = find_good_stepsize(hamiltonian, randn(D)) / 2
    integrator = isnothing(integrator) ? Leapfrog(initial_ϵ) : integrator
    adaptor = isnothing(adoptor) ?
        StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(target_accept, integrator)) : adoptor
    kernel =  HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    sampler = HMCSampler(kernel, metric, adaptor)
    HMCBlock(vars, hamiltonian, sampler, :NUTS, D, 1, 0, 0)
end

function HMCDABlock(mod::AbstractGM,
    vars::Vector{Symbol};
    λ::Float64,  # taregt trajectory length
    target_accept::Float64=0.65,
    metric::Symbol=:diag,  # {:diag, :dense, :unit}
    integrator::Union{Nothing,TIntegrator}=nothing
) where {
    TIntegrator<:AdvancedHMC.AbstractIntegrator
}
    ℓπ = (p) -> ulogpdf(p, mod, vars)
    ∂ℓπ∂θ = (p) -> (ℓπ(p), ∇ulogpdf(p, mod, vars))
    D = length(pack_param_vec(mod, vars))
    metric = init_metric(metric, D)
    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    initial_ϵ = find_good_stepsize(hamiltonian, randn(D))
    integrator = isnothing(integrator) ? Leapfrog(initial_ϵ) : integrator
    adaptor = StepSizeAdaptor(target_accept, initial_ϵ)
    kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedIntegrationTime(λ)))
    sampler = HMCSampler(kernel, metric, adaptor)
    HMCBlock(vars, hamiltonian, sampler, :HMCDA, D, 1, 0, 0)
end

function init_metric(metric::Symbol, D::Int)
    if metric == :diag
        return DiagEuclideanMetric(D)
    elseif metric == :dense
        return DenseEuclideanMetric(D)
    elseif metric == :unit
        return UnitEuclideanMetric(D)
    else
        error("metric should be one of {:diag, :dense, :unit}")
    end
    return metric
end

function RWMHBlock(mod::AbstractGM,
    vars::Vector{Symbol};
    σ::Float64, 
    n_iter::Int=1
)
    ℓπ = (p) -> ulogpdf(p, mod, vars)

    # MH components
    D = length(pack_param_vec(mod, vars))
    model = DensityModel(ℓπ)
    if D > 1
        proposal = MvNormal(zeros(D), σ^2 * I)
    else
        proposal = Normal(0, σ)
    end
    sampler = RWMH(proposal)
    MHBlock(vars, model, sampler, D, n_iter)
end

function StaticMHBlock(mod::AbstractGM,
    vars::Vector{Symbol};
    qdist::Union{Distribution,Vector{<:Distribution}},
    n_iter::Int=1
)
    ℓπ = (p) -> ulogpdf(p, mod, vars)

    # MH components
    D = length(pack_param_vec(mod, vars))
    model = DensityModel(ℓπ)
    if D > 1
        if qdist isa Distribution
            qdist = [qdist for _ in 1:D]
        else
            if length(qdist) != D
                qdist = fill(qdist[1], D)
                @warn "qdist should be a vector of length D. Using the first element as the proposal distribution for all parameters."
            end
        end
        proposal = StaticProposal(qdist)
    else
        if qdist isa Distribution
            proposal = StaticProposal(qdist)
        else
            if length(qdist) != 1
                qdist = fill(qdist[1], 1)
                @warn "qdist should be a vector of length 1. Using the first element as the proposal distribution."
            end
            proposal = StaticProposal(qdist[1])
        end
    end
    sampler = MetropolisHastings(proposal)
    MHBlock(vars, model, sampler, D, n_iter)
end

function ComponentWiseMHBlock(mod::AbstractGM,
    vars::Vector{Symbol};
    σ::Union{Float64,Vector{Float64}},
    n_iter::Int=1
)
    ℓπ = (p) -> ulogpdf(p, mod, vars)

    # MH components
    D = length(pack_param_vec(mod, vars))
    model = DensityModel(ℓπ)
    if σ isa Float64
        proposal = [Normal(0, σ) for _ in 1:D]
    else
        if length(σ) != D
            σ = fill(σ[1], D)
            @warn "σ should be a vector of length D. Using the first element as the proposal standard deviation for all parameters."
        end
        proposal = [Normal(0, σ[i]) for i in 1:D]
    end
    sampler = ComponentWiseMH(proposal)
    ComponentWiseMHBlock(vars, model, sampler, D, n_iter)
end

function ESSBlock(mod::AbstractGM,
    vars::Vector{Symbol};
    prior::MvNormal,
    n_iter::Int=1
)
    D = length(pack_param_vec(mod, vars))
    @assert length(prior.μ) == D "dimension of prior should match the number of parameters $D"
    # The residual error of the approximation to the target density
    # R(p) = π(p)/prior(p)
    # log residual error is used as the log likelihood
    # in generalized ESS proposed by Nishihara et al. 2014
    loglikelihood = (p) -> ulogpdf(p, mod, vars) - logpdf(prior, p)

    # ESS components
    model = EllipticalSliceSampling.ESSModel(prior, loglikelihood)
    sampler = EllipticalSliceSampling.ESS()
    ESSBlock(vars, model, sampler, D, n_iter)
end

function GESSBlock(mod::AbstractGM,
    vars::Vector{Symbol};
    prior::Union{MvTDist, Distributions.GenericMvTDist},
    n_iter::Int=1
)
    D = length(pack_param_vec(mod, vars))
    @assert length(prior.μ) == D "dimension of prior should match the number of parameters $D"
    # The residual error of the approximation to the target density
    # R(p) = π(p)/prior(p)
    # log residual error is used as the log likelihood
    # in generalized ESS proposed by Nishihara et al. 2014
    loglikelihood = (p) -> ulogpdf(p, mod, vars) - logpdf(prior, p)

    # ESS components
    model = EllipticalSliceSampling.ESSModel(MvNormal(prior.μ, prior.Σ), loglikelihood)
    sampler = GESS(prior)
    GESSBlock(vars, model, sampler, D, n_iter)
end
