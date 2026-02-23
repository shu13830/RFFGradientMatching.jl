# Utility functions used in the sampling process ---
# --- pack Dict style states ---
function pack_param_dict(gm::AbstractGM)
    param_dict = Dict{Symbol,Any}()
    if gm isa RFFGM
        param_dict[:W] = get_W(gm)
    else  # GPGM or MAGI
        param_dict[:X] = get_transformed_X(gm)
    end
    param_dict[:θ] = get_transformed_θ(gm)
    param_dict[:γ] = get_transformed_γ(gm)
    param_dict[:σ] = get_transformed_σ(gm)
    param_dict[:ϕ] = get_transformed_ϕ(gm)
    return param_dict
end

function update_param_dict_from_vec!(
    param_dict::Dict{Symbol,Any}, mod::AbstractGM,
    param_vec::AbstractVector{<:Real},
    sample_target::Vector{Symbol}
)
    # new_param_dict includes `nothing` for the keys not in sample_target
    new_param_dict = pack_param_dict_from_vec(mod, param_vec, sample_target)
    # update only the keys in sample_target
    for key in sample_target
        param_dict[key] = new_param_dict[key]
    end
    # ensure param_dict does not include `nothing` as values
    for key in keys(param_dict)
        @assert param_dict[key] != nothing
    end
end


function pack_param_vec_from_dict(gm::AbstractGM,
    param_dict::Dict{Symbol,Any}, sample_target::Vector{Symbol}
)
    param_vec = []
    
    if gm isa RFFGM
        if :W in sample_target && haskey(param_dict, :W)
            append!(param_vec, flatten_param(param_dict[:W] |> transpose))
        end
    else  # GPGM or MAGI
        if :X in sample_target && haskey(param_dict, :X)
            append!(param_vec, flatten_param(param_dict[:X] |> transpose))
        end
    end
    
    for key in [:θ, :γ, :σ, :ϕ]
        if key in sample_target && haskey(param_dict, key)
            append!(param_vec, flatten_param(param_dict[key]))
        end
    end
    
    return copy(convert(Vector{Float64}, param_vec))
end

function flatten_param(param)
    if isa(param, Number)
        return [float(param)]
    else
        return vec(param)
    end
end

pack_param_vec(gm::AbstractGM, sample_target::Vector{Symbol}) =
    pack_param_vec_from_dict(gm, pack_param_dict(gm), sample_target)

function pack_param_dict_from_vec(gm::AbstractGM, param_vec::AbstractVector{<:Real}, sample_target::Vector{Symbol})
    idx = 1
    param_dict = Dict{Symbol,Any}()
    param_dict[:Y] = get_Y(gm)

    if gm isa RFFGM
        if :W in sample_target
            n_W = sum([gpk.n_rff for gpk in gm.gp])
            W = reshape(param_vec[idx:idx + n_W - 1], :, length(gm.gp))' |> Matrix
            param_dict[:W] = W
            idx += n_W
        else
            param_dict[:W] = get_W(gm)
        end
    else  # GPGM or MAGI
        if :X in sample_target
            n_X = length(get_X(gm))
            transformed_X = reshape(param_vec[idx:idx + n_X - 1], :, length(gm.gp))' |> Matrix
            param_dict[:X] = transformed_X
            idx += n_X
        else
            param_dict[:X] = get_transformed_X(gm)
        end
    end

    if :θ in sample_target
        n_θ = length(gm.odegrad.θ)
        transformed_θ = param_vec[idx:idx + n_θ - 1]
        param_dict[:θ] = transformed_θ
        idx += n_θ
    else
        param_dict[:θ] = get_transformed_θ(gm)
    end

    if :γ in sample_target
        transformed_γ = param_vec[idx]
        param_dict[:γ] = transformed_γ
        idx += 1
    else
        param_dict[:γ] = get_transformed_γ(gm)
    end

    if :σ in sample_target
        n_σ = length(gm.gp)
        transformed_σ = param_vec[idx:idx + n_σ - 1]
        param_dict[:σ] = transformed_σ
        idx += n_σ
    else
        param_dict[:σ] = get_transformed_σ(gm)
    end

    if :ϕ in sample_target
        n_ϕ = get_transformed_ϕ(gm) |> length
        transformed_ϕ = reshape(param_vec[idx:idx + n_ϕ - 1], :, length(gm.gp))
        param_dict[:ϕ] = transformed_ϕ
        idx += n_ϕ
    else
        param_dict[:ϕ] = get_transformed_ϕ(gm)
    end
    return param_dict
end

function update_model_with_dict!(
    gm::AbstractGM,
    sample_target::Vector{Symbol},
    new_param_dict::Dict{Symbol,Any}
)
    if haskey(new_param_dict, :Y) && :Y in sample_target
        update_Y!(gm, new_param_dict[:Y])
    end
    if haskey(new_param_dict, :X) && :X in sample_target
        update_X!(gm, new_param_dict[:X])
    end
    if haskey(new_param_dict, :W) && :W in sample_target
        update_W!(gm, new_param_dict[:W])
    end
    if haskey(new_param_dict, :θ) && :θ in sample_target
        update_θ!(gm, new_param_dict[:θ])
    end
    if haskey(new_param_dict, :γ) && :γ in sample_target
        update_γ!(gm, new_param_dict[:γ])
    end
    if haskey(new_param_dict, :σ) && :σ in sample_target
        update_σ!(gm, new_param_dict[:σ])
    end
    if haskey(new_param_dict, :ϕ) && :ϕ in sample_target
        update_ϕ!(gm, new_param_dict[:ϕ])
    end
end

function update_model_with_vec!(gm::AbstractGM,
    sample_target::Vector{Symbol}, new_param_vec::AbstractVector{<:Real}
)
    param_dict = pack_param_dict_from_vec(gm, new_param_vec, sample_target)
    update_model_with_dict!(gm, sample_target, param_dict)
end

function update_Y!(gm::AbstractGM, Y::AbstractMatrix{<:Real})
    for (gpk, yk) in zip(gm.gp, eachrow(Y))
        update_y!(gpk, yk[:])
    end
end

function update_X!(gm::Union{GPGM,MAGI}, transformed_X::AbstractMatrix{<:Real})
    X = calc_X(gm.gp, transformed_X)
    gm.odegrad.X = X
    for (gpk, xk) in zip(gm.gp, eachrow(X))
        update_u!(gpk, xk[:])
    end
end

function update_W!(gm::RFFGM, W::AbstractMatrix{<:Real})
    X = W2X(gm.gp, W)
    for (gpk, xk, wk) in zip(gm.gp, eachrow(X), eachrow(W))
        update_u!(gpk, xk[:])
        gpk.w[:] = wk[:]
    end
end

function update_θ!(gm::AbstractGM, transformed_θ::AbstractVector{<:Real})
    gm.odegrad.θ[:] = calc_θ(gm.odegrad, transformed_θ)
end

function update_γ!(gm::AbstractGM, transformed_γ::T) where {T<:Real}
    gm.odegrad.γ = calc_var(gm.odegrad.tγ, transformed_γ)
end

function update_σ!(gm::AbstractGM, transformed_σ::AbstractVector{<:Real})
    for (gpk, σk) in zip(gm.gp, calc_σ(gm.gp, transformed_σ))
        gpk.σ = σk
    end
end

function update_ϕ!(gm::AbstractGM, transformed_ϕ::AbstractMatrix{<:Real})
    new_ϕ = calc_ϕ(gm.gp, transformed_ϕ)
    for (k, (gpk, ϕk)) in enumerate(zip(gm.gp, eachcol(new_ϕ)))
        gm.gp[k] = reconstruct_gp(gpk; ϕ=ϕk[:])
    end
end

# --- extract parameters from a chain ---

function get_θ(gm::AbstractGM, chain::AbstractVector)
    θ_samples = [calc_θ(gm.odegrad, c[:θ]) for c in chain]
    return reduce(hcat, θ_samples)'
end

function get_γ(gm::AbstractGM, chain::AbstractVector)
    return [calc_var(gm.odegrad.tγ, c[:γ]) for c in chain]
end

function get_σ(gm::AbstractGM, chain::AbstractVector)
    σ_samples = [calc_σ(gm.gp, c[:σ]) for c in chain]
    return reduce(hcat, σ_samples)'
end

function get_ϕ(gm::AbstractGM, chain::AbstractVector)
    ϕ_samples = [calc_ϕ(gm.gp, c[:ϕ]) for c in chain]
    n_samples = length(ϕ_samples)
    n_params, n_gp = size(ϕ_samples[1])
    arr = Array{Float64}(undef, n_samples, n_params, n_gp)
    for (i, ϕi) in enumerate(ϕ_samples)
        arr[i, :, :] = ϕi
    end
    return arr
end

function get_X(gm::Union{GPGM,MAGI}, chain::AbstractVector)
    X_samples = [calc_X(gm.gp, c[:X]) for c in chain]
    n_samples = length(X_samples)
    n_gp, n_t = size(X_samples[1])
    arr = Array{Float64}(undef, n_samples, n_gp, n_t)
    for (i, Xi) in enumerate(X_samples)
        arr[i, :, :] = Xi
    end
    return arr
end

function get_W(gm::RFFGM, chain::AbstractVector)
    W_samples = [c[:W] for c in chain]
    n_samples = length(W_samples)
    n_gp, n_rff = size(W_samples[1])
    arr = Array{Float64}(undef, n_samples, n_gp, n_rff)
    for (i, Wi) in enumerate(W_samples)
        arr[i, :, :] = Wi
    end
    return arr
end

function get_logdensity(gm::AbstractGM, chain::AbstractVector)
    logdensities = [c[:logdens] |> values |> sum for c in chain]
    return logdensities
end
function get_logdensity(gm::AbstractGM, chain::AbstractVector, symbol::Symbol)
    logdensities = [c[:logdens][symbol] for c in chain]
    return logdensities
end
