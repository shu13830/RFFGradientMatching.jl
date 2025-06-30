# elliptical slice sampler
struct GESS <: AbstractMCMC.AbstractSampler
    prior::Union{MvTDist, Distributions.GenericMvTDist}
end

# state of the elliptical slice sampler
struct GESSState{S,L,C}
    "Sample of the elliptical slice sampler."
    sample::S
    "Log-likelihood of the sample."
    loglikelihood::L
    "Cache used for in-place sampling."
    cache::C
end

function GESSState(sample, loglikelihood)
    # create cache since it was not provided (initial sampling step)
    cache = ArrayInterface.ismutable(sample) ? similar(sample) : nothing
    return GESSState(sample, loglikelihood, cache)
end

# first step of the elliptical slice sampler
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::EllipticalSliceSampling.ESSModel,
    sampler::GESS;
    init_params=nothing,
    kwargs...,
)
    # initial sample from the Gaussian prior
    f = init_params === nothing ? rand(rng, model.prior) : init_params

    # compute log-likelihood of the initial sample
    loglikelihood = Distributions.loglikelihood(model, f)

    return f, GESSState(f, loglikelihood)
end

# subsequent steps of the elliptical slice sampler
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::EllipticalSliceSampling.ESSModel,
    sampler::GESS,
    state::GESSState;
    kwargs...,
)
    # obtain the prior
    prior = EllipticalSliceSampling.prior(model)
    s = sample_scale_parameter(state.sample, sampler.prior)
    scaled_prior = MvNormal(prior.μ, s * prior.Σ)

    # sample from scaled Gaussian prior
    cache = state.cache
    if cache === nothing
        ν = Random.rand(rng, scaled_prior)
    else
        Random.rand!(rng, scaled_prior, cache)
        ν = cache
    end

    # sample log-likelihood threshold
    loglikelihood = state.loglikelihood
    threshold = loglikelihood - Random.randexp(rng)

    # sample initial angle
    θ = 2 * π * rand(rng)
    θmin = θ - 2 * π
    θmax = θ

    # compute the proposal
    f = state.sample
    fnext = EllipticalSliceSampling.proposal(scaled_prior, f, ν, θ)

    # compute the log-likelihood of the proposal
    loglikelihood = Distributions.loglikelihood(model, fnext)

    # stop if the log-likelihood threshold is reached
    while loglikelihood < threshold
        # shrink the bracket
        if θ < zero(θ)
            θmin = θ
        else
            θmax = θ
        end

        # sample angle
        θ = θmin + rand(rng) * (θmax - θmin)

        # recompute the proposal
        if ArrayInterface.ismutable(fnext)
            EllipticalSliceSampling.proposal!(fnext, scaled_prior, f, ν, θ)
        else
            fnext = EllipticalSliceSampling.proposal(scaled_prior, f, ν, θ)
        end

        # compute the log-likelihood of the proposal
        loglikelihood = Distributions.loglikelihood(model, fnext)
    end

    return fnext, GESSState(fnext, loglikelihood, cache)
end

function sample_scale_parameter(p::AbstractVector{<:Real}, t::Union{MvTDist, Distributions.GenericMvTDist})
    D = length(p)
    ν = t.df
    μ = t.μ
    Σ = t.Σ
    α′ = (D + ν) / 2
    β′ = 0.5 * (ν + (p - μ)' * inv(Σ) * (p - μ))
    s = rand(InverseGamma(α′, β′))
    return s
end
