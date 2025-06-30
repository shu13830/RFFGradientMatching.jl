struct ComponentWiseMH{P} <: AdvancedMH.MHSampler
    proposals::Vector{P}
end

struct ComponentWiseMHTransition{P}
    params::Vector{P}
    lp::Float64
    accepted::Bool
end

function ComponentWiseMHTransition(model::AdvancedMH.DensityModel, params::Vector{<:Real}, accepted::Bool)
    lp = model.logdensity(params)
    return ComponentWiseMHTransition(params, lp, accepted)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, 
    model::AdvancedMH.DensityModel, 
    sampler::ComponentWiseMH; 
    initial_params=nothing, 
    kwargs...
)
    state = isnothing(initial_params) ?
        [rand(rng, sampler.proposals[i]) for i in 1:length(sampler.proposals)] :
        copy(initial_params)
    return ComponentWiseMHTransition(model, state, false), ComponentWiseMHTransition(model, state, false)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AdvancedMH.DensityModel,
    sampler::ComponentWiseMH,
    t_prev::ComponentWiseMHTransition{<:Real};
    kwargs...
)
    new_state = copy(t_prev.params)
    current_lp = t_prev.lp
    accepted_any = false
    for d in 1:length(new_state)
        cand = copy(new_state)
        cand[d] += rand(rng, sampler.proposals[d])
        candidate_lp =  model.logdensity(cand)
        if log(rand(rng)) < candidate_lp - current_lp
            new_state[d] = cand[d]
            current_lp = candidate_lp
            accepted_any = true
        end
    end
    return ComponentWiseMHTransition(model, new_state, accepted_any), ComponentWiseMHTransition(model, new_state, accepted_any)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::AdvancedMH.DensityModel,
    sampler::ComponentWiseMH,
    initial_state,
    n_iter::Int
)
    t, state = AbstractMCMC.step(rng, model, sampler; initial_params=initial_state)
    chain = [t]
    for i in 1:n_iter
        t, state = AbstractMCMC.step(rng, model, sampler, state)
        push!(chain, t)
    end
    return chain
end
