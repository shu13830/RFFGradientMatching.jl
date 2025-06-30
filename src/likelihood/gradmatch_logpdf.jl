function ulogpdf(gm::RFFGM, param_dict::Dict{Symbol,Any}; 
    sample_target::Union{Nothing, Vector{Symbol}}=nothing, 
    merge_output::Bool=true
)
    transformed_θ = param_dict[:θ]
    transformed_γ = param_dict[:γ]
    transformed_σ = param_dict[:σ]
    transformed_ϕ = param_dict[:ϕ]

    if isnothing(sample_target)
        Y_std = get_standardized_Y(gm)
        W = get_W(gm)
        θ = get_θ(gm)
        γ = get_γ(gm)
        σ = get_σ(gm)
        ϕ = get_ϕ(gm)
    else
        Y_std = get_standardized_Y(gm)
        W = :W in sample_target ? param_dict[:W] : get_W(gm)
        θ = :θ in sample_target ? calc_θ(gm.odegrad, transformed_θ) : get_θ(gm)
        γ = :γ in sample_target ? calc_var(gm.odegrad.tγ, transformed_γ) : get_γ(gm)
        σ = :σ in sample_target ? calc_σ(gm.gp, transformed_σ) : get_σ(gm)
        ϕ = :ϕ in sample_target ? calc_ϕ(gm.gp, transformed_ϕ) : get_ϕ(gm)
    end

    lx = logpdf_x(gm, W)
    ly = logpdf_y(gm, Y_std, W, σ)
    lθ = logpdf_θ(gm, θ)
    lγ = logpdf_γ(gm, γ)
    lσ = logpdf_σ(gm, σ)
    lϕ = logpdf_ϕ(gm, ϕ)
    le = gm.β[1] * ulogpdf_e(gm, W, θ, γ)  # weight by the inverse temperature

    if merge_output
        return lx + ly + lθ + lγ + lσ + lϕ + le
    else
        return Dict(
            :logpdf_x => lx,
            :logpdf_y => ly,
            :logpdf_θ => lθ,
            :logpdf_γ => lγ,
            :logpdf_σ => lσ,
            :logpdf_ϕ => lϕ,
            :ulogpdf_e => le
        )
    end
end

function ulogpdf(gm::GPGM, param_dict::Dict{Symbol,Any};
    sample_target::Union{Nothing, Vector{Symbol}}=nothing,
    merge_output::Bool=true
)
    transformed_X = param_dict[:X]
    transformed_θ = param_dict[:θ]
    transformed_γ = param_dict[:γ]
    transformed_σ = param_dict[:σ]
    transformed_ϕ = param_dict[:ϕ]

    if isnothing(sample_target)
        Y_std = get_standardized_Y(gm)
        X = get_X(gm)
        θ = get_θ(gm)
        γ = get_γ(gm)
        σ = get_σ(gm)
        ϕ = get_ϕ(gm)
    else
        Y_std = get_standardized_Y(gm)
        X = :X in sample_target ? calc_X(gm.gp, transformed_X) : get_X(gm)
        θ = :θ in sample_target ? calc_θ(gm.odegrad, transformed_θ) : get_θ(gm)
        γ = :γ in sample_target ? calc_var(gm.odegrad.tγ, transformed_γ) : get_γ(gm)
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
    lγ = logpdf_γ(gm, γ)
    lσ = logpdf_σ(gm, σ)
    lϕ = logpdf_ϕ(gm, ϕ)
    le = gm.β[1] * ulogpdf_e(gm, X, θ, γ)  # weight by the inverse temperature

    if merge_output
        return lx + ly + lθ + lγ + lσ + lϕ + le
    else
        return Dict(
            :logpdf_x => lx,
            :logpdf_y => ly,
            :logpdf_θ => lθ,
            :logpdf_γ => lγ,
            :logpdf_σ => lσ,
            :logpdf_ϕ => lϕ,
            :ulogpdf_e => le
        )
    end
end

ulogpdf(gm::Union{RFFGM,GPGM}) = ulogpdf(gm, pack_param_dict(gm))
function ulogpdf(gm::Union{RFFGM,GPGM}, sample_target::Vector{Symbol})
    ulogpdf(gm, pack_param_dict(gm); sample_target=sample_target)
end
function ulogpdf(param_vec::AbstractVector{<:Real}, gm::Union{RFFGM,GPGM}, sample_target::Vector{Symbol})
    return ulogpdf(gm, pack_param_dict_from_vec(gm, param_vec, sample_target); sample_target=sample_target)
end

# --- x ---
function logpdf_x(gp::Union{Vector{GP},Vector{RFFGP}}, X::AbstractMatrix{<:Real})
    lpd = 0.0
    for (k, xk) in enumerate(eachrow(X))
        lpd += logpdf(gp[k].fz, xk)
    end
    return lpd
end
logpdf_x(gm::GPGM, X::AbstractMatrix{<:Real}, ϕ::AbstractMatrix{<:Real}) = logpdf_x(reconstruct_gp(gm.gp, ϕ=ϕ), X)
logpdf_x(gm::GPGM, X::AbstractMatrix{<:Real}) = logpdf_x(gm.gp, X)
logpdf_x(gm::RFFGM, W::AbstractMatrix{<:Real}) = logpdf_x(gm.gp, W2X(gm.gp, W))
logpdf_x(gm::GPGM) = logpdf_x(gm, get_X(gm))
logpdf_x(gm::RFFGM) = logpdf_x(gm, get_W(gm))

# --- y ---
function logpdf_y(gp::Vector{GP}, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    lpd = 0.0
    for (gpk, ystdk, xk, σk) in zip(gp, eachrow(Y_std), eachrow(X), σ)
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, xk, σk)
        lpd += logpdf(MvNormal(y_mean, y_cov), ystdk)
    end
    return lpd
end
logpdf_y(gm::GPGM, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, ϕ::AbstractMatrix{<:Real}) =
    logpdf_y(reconstruct_gp(gm.gp; ϕ=ϕ), Y_std, X, σ)
logpdf_y(gm::GPGM, X::AbstractMatrix{<:Real}, ϕ::AbstractMatrix{<:Real}) =
    logpdf_y(reconstruct_gp(gm.gp; ϕ=ϕ), get_standardized_Y(gm), X, get_σ(gm))
logpdf_y(gm::GPGM, σ::AbstractVector{<:Real}, ϕ::AbstractMatrix{<:Real}) =    
    logpdf_y(reconstruct_gp(gm.gp; ϕ=ϕ), get_standardized_Y(gm), get_X(gm), σ)
logpdf_y(gm::GPGM, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) =
    logpdf_y(gm.gp, Y_std, X, σ)
logpdf_y(gm::GPGM, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) = logpdf_y(gm.gp, get_standardized_Y(gm), X, σ)
logpdf_y(gm::GPGM, X::AbstractMatrix{<:Real}) = logpdf_y(gm, get_standardized_Y(gm), X, get_σ(gm))
logpdf_y(gm::GPGM, σ::AbstractVector{<:Real}) = logpdf_y(gm, get_standardized_Y(gm), get_X(gm), σ)
logpdf_y(gm::GPGM) = logpdf_y(gm, get_standardized_Y(gm), get_X(gm), get_σ(gm))

function logpdf_y(gp::Vector{RFFGP}, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    lpd = 0.0
    for (gpk, ystdk, wk, σk) in zip(gp, eachrow(Y_std), eachrow(W), σ)
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, wk, σk)
        lpd += logpdf(MvNormal(y_mean, y_cov), ystdk)
    end
    return lpd
end
logpdf_y(gm::RFFGM, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) = logpdf_y(gm.gp, Y_std, W, σ)
logpdf_y(gm::RFFGM, W::AbstractMatrix{<:Real}) = logpdf_y(gm, get_standardized_Y(gm), W, get_σ(gm))
logpdf_y(gm::RFFGM, σ::AbstractVector{<:Real}) = logpdf_y(gm, get_standardized_Y(gm), get_W(gm), σ)
logpdf_y(gm::RFFGM) = logpdf_y(gm, get_standardized_Y(gm), get_W(gm), get_σ(gm))

# --- θ ---
function logpdf_θ(odegrad::ODEGrad, θ::AbstractVector{<:Real})
    @unpack tθ = odegrad
    @assert length(tθ) == length(θ)
    lpd = 0.0
    for (tθi, θi) in zip(tθ,θ)
        lpd += logpdf(tθi.prior, calc_tvar(tθi, θi))
    end
    return lpd
end
logpdf_θ(gm::Union{RFFGM,GPGM}, θ::AbstractVector{<:Real}) = logpdf_θ(gm.odegrad, θ)
logpdf_θ(gm::Union{RFFGM,GPGM}) = logpdf_θ(gm, get_θ(gm))


# --- e ---
"unnormalized logpdf of the gradient error term"
function ulogpdf_e(
    ẋode::Matrix{T1},
    ẋgp_mean::Vector{Vector{T2}},
    ẋgp_cov::Vector{Matrix{T3}},
    γ::T4
) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}
    K, N = size(ẋode)
    lpd = 0.0
    for (k, fk) in enumerate(eachrow(ẋode))
        e = fk - ẋgp_mean[k]  # gradient error
        e_cov = Hermitian(ẋgp_cov[k] + γ^2 * LinearAlgebra.I)
        lpd += logpdf(MvNormal(zeros(N), e_cov), e)
    end
    return lpd
end

function ulogpdf_e(
    odegrad::ODEGrad, 
    gp::Vector{GP}, 
    X::AbstractMatrix{<:Real}, 
    θ::AbstractVector{<:Real}, 
    γ::T
) where {T<:Real}
    X_destandardized = calc_destandardized_X(gp, X)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ get_y_std(gp)  # K x N
    return ulogpdf_e(ẋode, dfdt_mean(gp, X), dfdt_cov(gp), γ)
end
ulogpdf_e(gm::GPGM, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T, ϕ::AbstractMatrix{<:Real}) where {T<:Real} =
    ulogpdf_e(gm.odegrad, reconstruct_gp(gm.gp; ϕ=ϕ), X, θ, γ)
ulogpdf_e(gm::GPGM, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ)
ulogpdf_e(gm::GPGM) = ulogpdf_e(gm, get_X(gm), get_θ(gm), get_γ(gm))

function ulogpdf_e(
    odegrad::ODEGrad, 
    gp::Vector{RFFGP}, 
    W::AbstractMatrix{<:Real}, 
    θ::AbstractVector{<:Real}, 
    γ::T
) where {T<:Real}
    X = W2X(gp, W)
    X_destandardized = calc_destandardized_X(gp, X)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ get_y_std(gp)  # K x N
    return ulogpdf_e(ẋode, dfdt_mean(gp, W), dfdt_cov(gp), γ)
end
ulogpdf_e(gm::RFFGM, W::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T, ϕ::AbstractMatrix{<:Real}) where {T<:Real} =
    ulogpdf_e(gm.odegrad, reconstruct_gp(gm.gp; ϕ=ϕ), W, θ, γ)
ulogpdf_e(gm::RFFGM, W::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ulogpdf_e(gm.odegrad, gm.gp, W, θ, γ)
ulogpdf_e(gm::RFFGM) = ulogpdf_e(gm, get_W(gm), get_θ(gm), get_γ(gm))

# --- γ ---
logpdf_γ(odegrad::ODEGrad, γ::T) where {T<:Real} = logpdf(odegrad.tγ.prior, calc_tvar(odegrad.tγ, γ))
logpdf_γ(gm::Union{RFFGM,GPGM}, γ::T) where {T<:Real} = logpdf_γ(gm.odegrad, γ)
logpdf_γ(gm::Union{RFFGM,GPGM}) = logpdf_γ(gm, get_γ(gm))

# --- σ ---
function logpdf_σ(gp::Union{Vector{RFFGP},Vector{GP}}, σ::AbstractVector{<:Real})
    lpd = 0.0
    for (gpk, σk) in zip(gp, σ)
        lpd += logpdf(gpk.tσ.prior, calc_tvar(gpk.tσ, σk))
    end
    return lpd
end
logpdf_σ(gm::Union{RFFGM,GPGM}, σ::AbstractVector{<:Real}) = logpdf_σ(gm.gp, σ)
logpdf_σ(gm::Union{RFFGM,GPGM}) = logpdf_σ(gm, get_σ(gm))


# --- ϕ ---
function logpdf_ϕ(gp::Union{Vector{RFFGP},Vector{GP}}, ϕ::AbstractMatrix{<:Real})
    lpd = 0.0
    for (gpk, ϕk) in zip(gp, eachcol(ϕ))
        for (tϕki, ϕki) in zip(gpk.tϕ, ϕk)
            lpd += logpdf(tϕki.prior, calc_tvar(tϕki, ϕki))
        end
    end
    return lpd
end
logpdf_ϕ(gm::Union{RFFGM,GPGM}, ϕ::AbstractMatrix{<:Real}) = logpdf_ϕ(gm.gp, ϕ)
logpdf_ϕ(gm::Union{RFFGM,GPGM}) = logpdf_ϕ(gm, get_ϕ(gm))

