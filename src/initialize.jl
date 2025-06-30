# Helper functions for initialization of parameters using BlockedSampler

"""
    filter_sampler(sampler::BlockedSampler, pred::Function)

Return a new `BlockedSampler` that retains only the sample blocks for which
`pred(block)` returns `true`. Blocks that become empty are discarded. Returns
`nothing` if no blocks remain after filtering.
"""
function filter_sampler(sampler::BlockedSampler, pred::Function)
    new_blocks = Vector{Vector{Any}}()
    new_probs = Float64[]
    for (blkset, pr) in zip(sampler.blocks, sampler.probs)
        sub = [blk for blk in blkset if pred(blk)]
        if !isempty(sub)
            push!(new_blocks, sub)
            push!(new_probs, pr)
        end
    end
    if isempty(new_blocks)
        return nothing
    end
    new_probs ./= sum(new_probs)
    return BlockedSampler(new_blocks, new_probs)
end

exclude_var(sampler::BlockedSampler, var::Symbol) =
    filter_sampler(sampler, blk -> !(var in blk.vars))

only_var(sampler::BlockedSampler, var::Symbol) =
    filter_sampler(sampler, blk -> length(blk.vars) == 1 && blk.vars[1] == var)

function only_var_set(sampler::BlockedSampler, vars::Vector{Symbol})
    filter_sampler(sampler, blk -> sort(blk.vars) == sort(vars))
end

function _get_gm(model::Union{RFFGM,GPGM})
    model
end

"""
    initialize_vars!(rng::AbstractRNG, model, sampler::BlockedSampler;
        num_samples_nonθ::Int=1000, num_samples_θ::Int=1000)

Run a two-stage initialization for BlockedSampler-based inference. First, the
gradient matching term is disabled (`β=0`) and all blocks not containing `:θ`
are sampled for `num_samples_nonθ` iterations to roughly adapt other variables.
Next, with `β=1`, only blocks that update `θ` are sampled for
`num_samples_θ` iterations. After initialization, `β` is reset to zero and the
returned parameter dictionary can be used as a starting point for annealed
sampling.
"""
function initialize_vars!(rng::AbstractRNG, model,
    sampler::BlockedSampler; num_samples_nonθ::Int=1000, num_samples_θ::Int=1000)

    gm = _get_gm(model)
    is_rffgm = gm isa RFFGM
    is_gpgm = gm isa GPGM

    # Check existence of blocks for θ and X/W for phase 2
    if is_gpgm
        samp_θ = only_var_set(sampler, [:X, :θ])
        if samp_θ === nothing
            error("No block found that contains both :X and :θ")
        end
    elseif is_rffgm
        samp_θ = only_var_set(sampler, [:W, :θ])
        if samp_θ === nothing
            error("No block found that contains both :W and :θ")
        end
    else
        error("Unsupported model type for initialization: $(typeof(model))")
    end

    # phase 1: β = 0, sample blocks excluding θ
    gm.β[1] = 0.0
    gm.anneal_iter[1] = 1
    samp_noθ = exclude_var(sampler, :θ)
    params = pack_param_dict(model)
    if samp_noθ !== nothing && num_samples_nonθ > 0
        AbstractMCMC.sample(rng, model, samp_noθ, num_samples_nonθ;
            init_params=params, num_burnin=num_samples_nonθ, anneal=false)
        params = pack_param_dict(model)
    end

    # phase 2: sample θ and X/W with annealing
    # gm.β[1] = 0.0
    # gm.anneal_iter[1] = 1
    # if num_samples_θ < gm.anneal_length
    #     @warn "num_samples_θ is less than the anneal length, which may lead to suboptimal initialization."
    #     num_samples_θ = gm.anneal_length
    #     @warn "num_samples_θ is set to gm.anneal_length ($gm.anneal_length) for annealing"
    # end
    # if samp_θ !== nothing && num_samples_θ > 0
    #     AbstractMCMC.sample(rng, model, samp_θ, num_samples_θ;
    #         init_params=params, num_burnin=num_samples_θ, anneal=true)
    #     params = pack_param_dict(model)
    # end

    # reset β for annealing start
    gm.β[1] = 0.0
    gm.anneal_iter[1] = 1

    return params
end

initialize_vars!(model, sampler::BlockedSampler; kwargs...) =
    initialize_vars!(Random.default_rng(), model, sampler; kwargs...)

