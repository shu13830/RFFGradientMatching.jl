# SampleBlock for block sampling
abstract type AbstractSampleBlock end

struct BlockedSampler <: AbstractMCMC.AbstractSampler
    blocks::Vector{Vector{<:Any}}
    probs::AbstractVector{<:Real}
    global_sample_target::Vector{Symbol}

    function BlockedSampler(blocks::Vector{<:Vector}, probs::Vector{Float64})
        @assert length(blocks) == length(probs)  "number of blocks should match number of probs"
        @assert isprobvec(probs)  "probs should be a probability vector"
        global_sample_target = Symbol[]
        for block in blocks
            @assert all([blk isa AbstractSampleBlock for blk in block]) "blocks should be a vector of AbstractSampleBlock"
            append!(global_sample_target, vcat([blk.vars for blk in block]...))
        end
        global_sample_target = unique(global_sample_target)
        new(blocks, probs, global_sample_target)
    end
end

mutable struct BlockedSamplerState
    params::Dict{Symbol, Any}  # Variable names: values
end

function choose_sampleblock(blocks::BlockedSampler)
    dist = Categorical(blocks.probs)
    idx = rand(dist)
    return blocks.blocks[idx]
end

function AbstractMCMC.step(rng::AbstractRNG,
    model::Union{GPGM, RFFGM},
    sampler::BlockedSampler,
    state::BlockedSamplerState,
    burnin::Bool
)
    block = choose_sampleblock(sampler)
    new_param_dict = copy(state.params)

    # Update all sample sub-blocks within the selected block
    for blk in block
        sample_target = blk.vars
        param_vec = pack_param_vec_from_dict(model, new_param_dict, sample_target)
        if blk isa HMCBlock
            new_param_vecs, stats = AbstractMCMC.sample(rng, blk.h, blk.sampler.κ, param_vec, blk.n+1, verbose=false)
            if stats[end].is_accept
                blk.accept_counter += 1
                blk.reject_counter = 0
            else
                blk.reject_counter += 1
                blk.accept_counter = 0
            end
            adjust_ϵ_heuristically!(burnin, blk)
            new_param_vec = new_param_vecs[end]
        elseif blk isa ESSBlock || blk isa GESSBlock
            new_param_vecs = EllipticalSliceSampling.sample(rng, blk.model, blk.sampler, blk.n, init_params=param_vec)
            new_param_vec = new_param_vecs[end]
        elseif blk isa MHBlock
            mh_transitions = AbstractMCMC.sample(rng, blk.model, blk.sampler, blk.n, init_params=param_vec)
            new_param_vec = mh_transitions[end].params
        elseif blk isa ComponentWiseMHBlock
            cwmh_transitions, _ = AbstractMCMC.sample(rng, blk.model, blk.sampler, param_vec, blk.n)
            new_param_vec = cwmh_transitions[end].params
        else
            error("Unsupported block type: $(typeof(blk))")
        end
        update_param_dict_from_vec!(new_param_dict, model, new_param_vec, sample_target)
    end

    return BlockedSamplerState(new_param_dict)
end

AbstractMCMC.step(
    model::Union{GPGM, RFFGM},
    sampler::BlockedSampler,
    state::BlockedSamplerState,
    burnin::Bool
) = AbstractMCMC.step(Random.default_rng(), model, sampler, state, burnin)

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::Union{GPGM, RFFGM},
    sampler::BlockedSampler,
    num_samples::Int;
    init_params::Dict{Symbol, Any},
    num_burnin::Int=0,
    chain_type::Type=Vector,
    anneal::Bool=true,
)
    if anneal
        @assert num_burnin >= get_anneal_length(model) "num_burnin should be greater than the anneal length in gradient matching"
    end

    state = BlockedSamplerState(init_params)
    chain = chain_type()
    # progress = Progress(num_samples, desc="Sampling")
    logdens_history = Float64[]
    for i in 1:num_samples
        burnin = i <= num_burnin
        state = AbstractMCMC.step(rng, model, sampler, state, burnin)
        update_model_with_dict!(model, sampler.global_sample_target, state.params)
        logdens_dict = ulogpdf(model, state.params; merge_output=false)
        state_dict = copy(state.params)
        state_dict[:logdens] = logdens_dict
        push!(chain, state_dict)
        push!(logdens_history, logdens_dict |> values |> sum)
        compiled_output(i, num_samples, logdens_history)
        # next!(progress)

        if anneal
            anneal_gm_callback(rng, model, sampler, state, i)
        end
    end
    return chain, logdens_history
end

function AbstractMCMC.sample(
    model::Union{GPGM, RFFGM},
    sampler::BlockedSampler,
    num_samples::Int;
    init_params::Union{Nothing,Dict{Symbol, Any}}=nothing,
    num_burnin::Int=0,
    chain_type::Type=Vector,
    anneal::Bool=true,
)
    if isnothing(init_params)
        init_params = pack_param_dict(model)
    end
    AbstractMCMC.sample(
        Random.default_rng(), model, sampler, num_samples;
        init_params=init_params, num_burnin=num_burnin, chain_type=chain_type, anneal=anneal)
end

function anneal_gm_callback(rng, model::Union{RFFGM,GPGM}, sampler, sample, iteration)
    # update the inverse temperature β
    if model.anneal_iter[1] < model.anneal_length
        @info "Annealing Gradient Matching term: iteration $(model.anneal_iter[1]) / $(model.anneal_length)"
        model.anneal_iter[1] += 1
        anneal_iter = model.anneal_iter[1]
        model.β[1] = model.β_schedule[anneal_iter]
    else
        @assert model.anneal_iter[1] == model.anneal_length "Anneal iter should be equal to anneal length"
        model.β[1] = 1.
    end
end

function get_anneal_length(model::Union{RFFGM,GPGM})
    return model.anneal_length
end
