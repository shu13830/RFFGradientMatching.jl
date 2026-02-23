# MAGI log-posterior density
# MAGI = GPGM with fixed γ (no gradient matching noise parameter)
# ODE constraint uses γ_jitter for numerical stability

# --- MAGI-specific logpdf dispatches ---
logpdf_x(gm::MAGI, X::AbstractMatrix{<:Real}, ϕ::AbstractMatrix{<:Real}) = logpdf_x(reconstruct_gp(gm.gp, ϕ=ϕ), X)
logpdf_x(gm::MAGI, X::AbstractMatrix{<:Real}) = logpdf_x(gm.gp, X)
logpdf_x(gm::MAGI) = logpdf_x(gm, get_X(gm))

logpdf_y(gm::MAGI, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, ϕ::AbstractMatrix{<:Real}) =
    logpdf_y(reconstruct_gp(gm.gp; ϕ=ϕ), Y_std, X, σ)
logpdf_y(gm::MAGI, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) =
    logpdf_y(gm.gp, Y_std, X, σ)
logpdf_y(gm::MAGI, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) = logpdf_y(gm.gp, get_standardized_Y(gm), X, σ)
logpdf_y(gm::MAGI, X::AbstractMatrix{<:Real}) = logpdf_y(gm, get_standardized_Y(gm), X, get_σ(gm))
logpdf_y(gm::MAGI) = logpdf_y(gm, get_standardized_Y(gm), get_X(gm), get_σ(gm))

logpdf_θ(gm::MAGI, θ::AbstractVector{<:Real}) = logpdf_θ(gm.odegrad, θ)
logpdf_θ(gm::MAGI) = logpdf_θ(gm, get_θ(gm))

logpdf_σ(gm::MAGI, σ::AbstractVector{<:Real}) = logpdf_σ(gm.gp, σ)
logpdf_σ(gm::MAGI) = logpdf_σ(gm, get_σ(gm))

logpdf_ϕ(gm::MAGI, ϕ::AbstractMatrix{<:Real}) = logpdf_ϕ(gm.gp, ϕ)
logpdf_ϕ(gm::MAGI) = logpdf_ϕ(gm, get_ϕ(gm))

# MAGI ODE constraint: ulogpdf_e with fixed γ_jitter
ulogpdf_e(gm::MAGI, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}) =
    ulogpdf_e(gm.odegrad, gm.gp, X, θ, gm.γ_jitter)
ulogpdf_e(gm::MAGI) = ulogpdf_e(gm, get_X(gm), get_θ(gm))

# --- Main ulogpdf ---
function ulogpdf(gm::MAGI, param_dict::Dict{Symbol,Any};
    sample_target::Union{Nothing, Vector{Symbol}}=nothing,
    merge_output::Bool=true
)
    transformed_X = param_dict[:X]
    transformed_θ = param_dict[:θ]
    transformed_σ = param_dict[:σ]
    transformed_ϕ = param_dict[:ϕ]

    if isnothing(sample_target)
        Y_std = get_standardized_Y(gm)
        X = get_X(gm)
        θ = get_θ(gm)
        σ = get_σ(gm)
        ϕ = get_ϕ(gm)
    else
        Y_std = get_standardized_Y(gm)
        X = :X in sample_target ? calc_X(gm.gp, transformed_X) : get_X(gm)
        θ = :θ in sample_target ? calc_θ(gm.odegrad, transformed_θ) : get_θ(gm)
        σ = :σ in sample_target ? calc_σ(gm.gp, transformed_σ) : get_σ(gm)
        ϕ = :ϕ in sample_target ? calc_ϕ(gm.gp, transformed_ϕ) : get_ϕ(gm)
    end

    if isnothing(sample_target)
        lx = logpdf_x(gm, X, ϕ)
        ly = logpdf_y(gm, Y_std, X, σ, ϕ)
    else
        lx = :ϕ in sample_target ? logpdf_x(gm, X, ϕ) : logpdf_x(gm, X)
        ly = :ϕ in sample_target ? logpdf_y(gm, Y_std, X, σ, ϕ) : logpdf_y(gm, Y_std, X, σ)
    end
    lθ = logpdf_θ(gm, θ)
    lσ = logpdf_σ(gm, σ)
    lϕ = logpdf_ϕ(gm, ϕ)
    # MAGI: ODE constraint with fixed γ_jitter, weighted by β
    le = gm.β[1] * ulogpdf_e(gm.odegrad, gm.gp, X, θ, gm.γ_jitter)

    if merge_output
        return lx + ly + lθ + lσ + lϕ + le
    else
        return Dict(
            :logpdf_x => lx,
            :logpdf_y => ly,
            :logpdf_θ => lθ,
            :logpdf_σ => lσ,
            :logpdf_ϕ => lϕ,
            :ulogpdf_e => le
        )
    end
end

ulogpdf(gm::MAGI) = ulogpdf(gm, pack_param_dict(gm))
function ulogpdf(gm::MAGI, sample_target::Vector{Symbol})
    ulogpdf(gm, pack_param_dict(gm); sample_target=sample_target)
end
function ulogpdf(param_vec::AbstractVector{<:Real}, gm::MAGI, sample_target::Vector{Symbol})
    return ulogpdf(gm, pack_param_dict_from_vec(gm, param_vec, sample_target); sample_target=sample_target)
end
