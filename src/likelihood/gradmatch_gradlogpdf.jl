function ∇ulogpdf(gm::RFFGM, param_dict::Dict{Symbol,Any}, sample_target::Vector{Symbol})

    if :Y in sample_target && haskey(param_dict, :Y)
        Y_std = calc_standardized_Y(gm.gp, param_dict[:Y])
    else
        Y_std = get_standardized_Y(gm)
    end
    if :W in sample_target
        W = param_dict[:W]
    else
        W = get_W(gm)
    end
    if :θ in sample_target
        transformed_θ = param_dict[:θ]
    else
        transformed_θ = get_transformed_θ(gm)
    end
    if :γ in sample_target
        transformed_γ = param_dict[:γ]
    else
        transformed_γ = get_transformed_γ(gm)
    end
    if :σ in sample_target
        transformed_σ = param_dict[:σ]
    else
        transformed_σ = get_transformed_σ(gm)
    end
    if :ϕ in sample_target
        transformed_ϕ = param_dict[:ϕ]
        ϕ = calc_ϕ(gm.gp, transformed_ϕ)
        gp = reconstruct_gp(gm.gp; ϕ=ϕ)
    else
        transformed_ϕ = get_transformed_ϕ(gm)
        gp = gm.gp
    end

    X = W2X(gp, W)
    θ = calc_θ(gm.odegrad, transformed_θ)
    γ = calc_γ(gm.odegrad, transformed_γ)
    σ = calc_σ(gp, transformed_σ)

    # compute gradients
    grad = AbstractVector{<:Real}[]
    if :Y in sample_target
        gy_dy = ∇y_logpdf_y(gp, Y_std, W, σ)
        push!(grad, gy_dy)
    end
    if :W in sample_target
        gy_dw = ∇w_logpdf_y(gp, Y_std, W, σ)
        gx_dw = ∇w_logpdf_x(gp, X)
        if gm.β[1] == 0.0
            ge_dw = zeros(length(gx_dw))
        else
            ge_dw = gm.β[1] * ∇w_ulogpdf_e(gm.odegrad, gp, X, W, θ, γ)  # NOTE: weighted by the inverse temperature
        end
        @assert length(gx_dw) == length(gy_dw) == length(ge_dw)
        push!(grad, gx_dw .+ gy_dw .+ ge_dw)
    end
    if :θ in sample_target
        gθ_dtθ = ∇tθ_logpdf_θ(gm.odegrad, θ, transformed_θ)
        if gm.β[1] == 0.0
            ge_dtθ = zeros(length(gθ_dtθ))
        else
            ge_dtθ = gm.β[1] * ∇tθ_ulogpdf_e(gm.odegrad, gp, X, W, θ, γ)  # NOTE: weighted by the inverse temperature
        end
        @assert length(gθ_dtθ) == length(ge_dtθ)
        push!(grad, gθ_dtθ .+ ge_dtθ)
    end
    if :γ in sample_target
        gγ_dtγ = ∇tγ_logpdf_γ(gm.odegrad, γ, transformed_γ)
        if gm.β[1] == 0.0
            ge_dtγ = zeros(length(gγ_dtγ))
        else
            ge_dtγ = gm.β[1] * ∇tγ_ulogpdf_e(gm.odegrad, gp, X, W, θ, γ)  # NOTE: weighted by the inverse temperature
        end
        @assert length(gγ_dtγ) == length(ge_dtγ)
        push!(grad, gγ_dtγ .+ ge_dtγ)
    end
    if :σ in sample_target
        gσ_dtσ = ∇tσ_logpdf_σ(gp, σ, transformed_σ)
        gy_dtσ = ∇tσ_logpdf_y(gp, Y_std, W, σ)
        @assert length(gy_dtσ) == length(gσ_dtσ)
        push!(grad, gy_dtσ .+ gσ_dtσ)
    end
    if :ϕ in sample_target
        gx_dtϕ = ∇tϕ_logpdf_x(gp, X, ϕ)  # TODO
        gy_dtϕ = ∇tϕ_logpdf_y(gp, Y_std, X, σ, ϕ)  # TODO
        gϕ_dtϕ = ∇tϕ_logpdf_ϕ(gp, ϕ)  # TODO
        if gm.β[1] == 0.0
            ge_dtϕ = zeros(length(gx_dtϕ))
        else
            ge_dtϕ = gm.β[1] * ∇tϕ_ulogpdf_e(gm.odegrad, gp, X, W, θ, γ, ϕ)  # TODO
        end
        @assert length(gx_dtϕ) == length(gy_dtϕ) == length(gϕ_dtϕ) == length(ge_dtϕ)
        push!(grad, gx_dtϕ .+ gy_dtϕ.+ gϕ_dtϕ.+ ge_dtϕ)
    end

    if length(grad) > 0
        grad = reduce(vcat, grad)
    end

    return grad
end

function ∇ulogpdf(gm::GPGM, param_dict::Dict{Symbol,Any}, sample_target::Vector{Symbol})
    
    if :Y in sample_target && haskey(param_dict, :Y)
        Y_std = calc_standardized_Y(gm.gp, param_dict[:Y])
    else
        Y_std = get_standardized_Y(gm)
    end
    if :X in sample_target
        transformed_X = param_dict[:X]
    else
        transformed_X = get_transformed_X(gm)
    end
    if :θ in sample_target
        transformed_θ = param_dict[:θ]
    else
        transformed_θ = get_transformed_θ(gm)
    end
    if :γ in sample_target
        transformed_γ = param_dict[:γ]
    else
        transformed_γ = get_transformed_γ(gm)
    end
    if :σ in sample_target
        transformed_σ = param_dict[:σ]
    else
        transformed_σ = get_transformed_σ(gm)
    end
    if :ϕ in sample_target
        transformed_ϕ = param_dict[:ϕ]
        ϕ = calc_ϕ(gm.gp, transformed_ϕ)
        gp = reconstruct_gp(gm.gp; ϕ=ϕ)
    else
        transformed_ϕ = get_transformed_ϕ(gm)
        gp = gm.gp
    end

    X = calc_X(gm.gp, transformed_X)
    θ = calc_var(gm.odegrad.tθ, transformed_θ)
    γ = calc_var(gm.odegrad.tγ, transformed_γ)
    σ = [calc_var(gpk.tσ, transformed_σk) for (gpk, transformed_σk) in zip(gp, transformed_σ)]

    # compute gradients
    grad = AbstractVector{<:Real}[]
    if :Y in sample_target
        gy_dy = ∇y_logpdf_y(gp, Y_std, X, σ)
        push!(grad, gy_dy)
    end
    if :X in sample_target
        gy_dx = ∇tx_logpdf_y(gp, Y_std, X, σ)
        gx_dx = ∇tx_logpdf_x(gp, X)
        if gm.β[1] == 0.0
            ge_dx = zeros(length(gx_dx))
        else
            ge_dx = gm.β[1] * ∇tx_ulogpdf_e(gm.odegrad, gp, X, θ, γ)  # NOTE: weighted by the inverse temperature
        end
        @assert length(gx_dx) == length(gy_dx) == length(ge_dx)
        push!(grad, gx_dx .+ gy_dx .+ ge_dx)
    end
    if :θ in sample_target
        gθ_dtθ = ∇tθ_logpdf_θ(gm.odegrad, θ, transformed_θ)
        if gm.β[1] == 0.0
            ge_dtθ = zeros(length(gθ_dtθ))
        else
            ge_dtθ = gm.β[1] * ∇tθ_ulogpdf_e(gm.odegrad, gp, X, θ, γ)  # NOTE: weighted by the inverse temperature
        end
        @assert length(gθ_dtθ) == length(ge_dtθ)
        push!(grad, gθ_dtθ .+ ge_dtθ)
    end
    if :γ in sample_target
        gγ_dtγ = ∇tγ_logpdf_γ(gm.odegrad, γ, transformed_γ)
        if gm.β[1] == 0.0
            ge_dtγ = zeros(length(gγ_dtγ))
        else
            ge_dtγ = gm.β[1] * ∇tγ_ulogpdf_e(gm.odegrad, gp, X, θ, γ)  # NOTE: weighted by the inverse temperature
        end
        @assert length(gγ_dtγ) == length(ge_dtγ)
        push!(grad, gγ_dtγ .+ ge_dtγ)
    end
    if :σ in sample_target
        gσ_dtσ = ∇tσ_logpdf_σ(gp, σ, transformed_σ)
        gy_dtσ = ∇tσ_logpdf_y(gp, Y_std, X, σ)
        @assert length(gy_dtσ) == length(gσ_dtσ)
        push!(grad, gy_dtσ .+ gσ_dtσ)
    end
    if :ϕ in sample_target
        gx_dtϕ = ∇tϕ_logpdf_x(gp, X, ϕ)  # TODO
        gy_dtϕ = ∇tϕ_logpdf_y(gp, Y_std, X, σ, ϕ)  # TODO
        gϕ_dtϕ = ∇tϕ_logpdf_ϕ(gp, ϕ)  # TODO
        if gm.β[1] == 0.0
            ge_dtϕ = zeros(length(gx_dtϕ))
        else
            ge_dtϕ = gm.β[1] * ∇tϕ_ulogpdf_e(gm.odegrad, gp, X, θ, γ, ϕ)  # TODO
        end
        @assert length(gx_dtϕ) == length(gy_dtϕ) == length(gϕ_dtϕ) == length(ge_dtϕ)
        push!(grad, gx_dtϕ .+ gy_dtϕ.+ gϕ_dtϕ.+ ge_dtϕ)
    end

    if length(grad) > 0
        grad = reduce(vcat, grad)
    end
    return grad
end

# ∇ulogpdf(gm::Union{RFFGM,GPGM}) = ∇ulogpdf(gm, pack_param_dict(gm))
∇ulogpdf(gm::Union{RFFGM,GPGM}, sample_target::Vector{Symbol}) = ∇ulogpdf(gm, pack_param_dict(gm), sample_target)

function ∇ulogpdf(param_vec::AbstractVector{<:Real}, gm::Union{RFFGM,GPGM}, sample_target::Vector{Symbol})
    return ∇ulogpdf(gm, pack_param_dict_from_vec(gm, param_vec, sample_target), sample_target)
end

# --- y ---
function ∇y_logpdf_y(gp::Vector{GP}, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    grads = []
    for (gpk, ystdk, xk, σk) in zip(gp, eachrow(Y_std), eachrow(X), σ)
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, xk, σk)
        grads_k = gradlogpdf(MvNormal(y_mean, y_cov), ystdk) ./ gpk.y_std
        push!(grads, grads_k)
    end
    return reduce(vcat, grads)
end
∇y_logpdf_y(gm::GPGM, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) = ∇y_logpdf_y(gm.gp, Y_std, X, σ)

function ∇y_logpdf_y(gp::Vector{RFFGP}, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    grads = []
    for (gpk, ystdk, wk, σk) in zip(gp, eachrow(Y_std), eachrow(W), σ)
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, wk, σk)
        grads_k = gradlogpdf(MvNormal(y_mean, y_cov), ystdk) ./ gpk.y_std
        push!(grads, grads_k)
    end
    return reduce(vcat, grads)
end
∇y_logpdf_y(gm::RFFGM, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) = ∇y_logpdf_y(gm.gp, Y_std, W, σ)

# --- x ---
# TESTED
∇tx_logpdf_x(gp::Vector{GP}, X::AbstractMatrix{<:Real}) = 
    reduce(vcat, [gpk.L' * gradlogpdf(gpk.fz, xk) for (gpk, xk) in zip(gp, eachrow(X))])
∇tx_logpdf_x(gm::GPGM, X::AbstractMatrix{<:Real}) = ∇tx_logpdf_x(gm.gp, X)

# TESTED
function ∇tx_logpdf_y(gp::Vector{GP}, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    grads = []
    for (gpk, ystdk, xk, σk) in zip(gp, eachrow(Y_std), eachrow(X), σ)
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, xk, σk)
        if gpk.z == gpk.x  # in case where inducing points are the same as training points
            grads_k = gpk.L' * (- gradlogpdf(MvNormal(y_mean, y_cov), ystdk))
        else  # in case where inducing points are different from training points
            grads_k = gpk.L' * gpk.KᵀK⁻¹' * (- gradlogpdf(MvNormal(y_mean, y_cov), ystdk))
        end
        push!(grads, grads_k)
    end
    return reduce(vcat, grads)
end
∇tx_logpdf_y(gm::GPGM, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) = ∇tx_logpdf_y(gm.gp, Y_std, X, σ)

# TESTED
function ∇tx_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{GP},
    X::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    γ::T
) where {T<:Real}
    X_destandardized = calc_destandardized_X(gp, X)
    y_std = get_y_std(gp)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ y_std  # K x N
    ẋgp_mean = dfdt_mean(gp, X)  # K length Vector of N length Vectors
    ẋgp_cov = dfdt_cov(gp)  # K length Vector of N x N Matrices
    ∇ẋgp_e = []
    for (k, ẋode_k) in enumerate(eachrow(ẋode))
        e = ẋgp_mean[k] - ẋode_k  # gradient error
        e_cov = Hermitian(ẋgp_cov[k] + γ^2 * LinearAlgebra.I)
        push!(∇ẋgp_e, gradlogpdf(MvNormal(zeros(length(e)), e_cov), e))  # N length Vector
    end
    ∇ẋgp_e = reduce(vcat, ∇ẋgp_e')  # K x N
    ∇ẋode_e = - ∇ẋgp_e  # K x N
    ∇x_ẋode = eval_dẋdx(odegrad, X_destandardized, θ, y_std)  # K x K x N
    ∇x_e = []
    for (k, gpk) in enumerate(gp)
        ∇xk_ẋgp = gpk.K′ᵀK⁻¹  # N x N
        ∇xk_e = ∇xk_ẋgp' * ∇ẋgp_e[k,:] +
            sum(∇ẋode_e .* ∇x_ẋode[:,k,:], dims=1)[:]  # N length Vector
        push!(∇x_e, gpk.L' * ∇xk_e)  # TODO
    end
    grads = reduce(vcat, ∇x_e)  # KN length Vector
    return grads
end
∇tx_ulogpdf_e(gm::GPGM, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ∇tx_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ)

# --- w ---
# TESTED
∇w_logpdf_x(gp::Vector{RFFGP}, X::AbstractMatrix{<:Real}) =
    reduce(vcat, [gpk.H' * gradlogpdf(gpk.fz, xk) for (gpk, xk) in zip(gp, eachrow(X))])
∇w_logpdf_x(gm::RFFGM, X::AbstractMatrix{<:Real}) = ∇w_logpdf_x(gm.gp, X)

# TESTED
function ∇w_logpdf_y(gp::Vector{RFFGP}, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    grads = []
    for (gpk, ystdk, wk, σk) in zip(gp, eachrow(Y_std), eachrow(W), σ)
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, wk[:], σk)
        if gpk.z == gpk.x  # in case where inducing points are the same as training points
            grads_k = gpk.H' * (- gradlogpdf(MvNormal(y_mean, y_cov), ystdk))
        else  # in case where inducing points are different from training points
            grads_k = gpk.H′' * (- gradlogpdf(MvNormal(y_mean, y_cov), ystdk))
        end
        push!(grads, grads_k)
    end
    return reduce(vcat, grads)
end
∇w_logpdf_y(gm::RFFGM, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) =
    ∇w_logpdf_y(gm.gp, Y_std, W, σ)

# TESTED
function ∇w_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{RFFGP},
    X::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    γ::T
) where {T<:Real}
    X_destandardized = calc_destandardized_X(gp, X)
    y_std = get_y_std(gp)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ y_std  # K x N
    ẋgp_mean = dfdt_mean(gp, W)  # K length Vector of N length Vectors
    ẋgp_cov = dfdt_cov(gp)  # K length Vector of N x N Matrices
    ∇ẋgp_e = []
    for (k, ẋode_k) in enumerate(eachrow(ẋode))
        e = ẋgp_mean[k] - ẋode_k  # gradient error
        e_cov = Hermitian(ẋgp_cov[k] + γ^2 * LinearAlgebra.I)
        push!(∇ẋgp_e, gradlogpdf(MvNormal(zeros(length(e)), e_cov), e))  # N length Vector
    end
    ∇ẋgp_e = reduce(vcat, ∇ẋgp_e')  # K x N
    ∇ẋode_e = - ∇ẋgp_e  # K x N
    ∇x_ẋode = eval_dẋdx(odegrad, X_destandardized, θ, y_std)  # K x K x N
    ∇w_e = []
    for (k, gpk) in enumerate(gp)
        ∇wk_ẋgp = gpk.dHdt  # n(RFF) x N
        ∇wk_e = ∇wk_ẋgp * ∇ẋgp_e[k,:] +
            gpk.H' * sum(∇ẋode_e .* ∇x_ẋode[:,k,:], dims=1)[:]  # N length Vector
        push!(∇w_e, ∇wk_e)
    end
    grads = reduce(vcat, ∇w_e)  # KN length Vector
    return grads
end
∇w_ulogpdf_e(gm::RFFGM, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ∇w_ulogpdf_e(gm.odegrad, gm.gp, X, W, θ, γ)

# --- σ ---
# TESTED
function ∇tσ_logpdf_y(gp::Vector{GP}, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    function gradlogpdf_dσ(dist::Normal, y::T, σ::T) where {T<:Real}
        x = dist.μ
        δ = dist.σ  # δ^2 = σ^2 + constant. Therefore, δ = sqrt(σ^2 + constant)
        grad = ((y - x)^2) * σ / (δ^4) - σ / δ^2
        return grad
    end

    grads = Float64[]
    for (gpk, ystdk, xk, σk) in zip(gp, eachrow(Y_std), eachrow(X), σ)
        grads_k = Float64[]
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, xk, σk)
        y_std = sqrt.(diag(y_cov))
        for (x_ki, σ_ki, y_ki) in zip(y_mean, y_std, ystdk)
            push!(grads_k, gradlogpdf_dσ(Normal(x_ki, σ_ki), y_ki, σk))
        end
        push!(grads, sum(grads_k) / gpk.tσ.dtv_dv(σk))
    end
    return reduce(vcat, grads)
end
∇tσ_logpdf_y(gm::GPGM, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) =
    ∇tσ_logpdf_y(gm.gp, Y_std, X, σ)

function ∇tσ_logpdf_y(gp::Vector{RFFGP}, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    function gradlogpdf_dσ(dist::Normal, y::T) where {T<:Real}
        x = dist.μ
        σ = dist.σ
        grad = (y - x)^2 / σ^3 - 1 / σ
        return grad
    end

    grads = Float64[]
    for (gpk, ystdk, wk, σk) in zip(gp, eachrow(Y_std), eachrow(W), σ)
        y_mean, y_cov = calc_y_mean_and_diagcov(gpk, wk[:], σk)
        y_std = sqrt.(diag(y_cov))
        grads_k = Float64[]
        for (x_ki, σ_ki, y_ki) in zip(y_mean, y_std, ystdk)
            push!(grads_k, gradlogpdf_dσ(Normal(x_ki, σ_ki), y_ki))
        end
        push!(grads, sum(grads_k) / gpk.tσ.dtv_dv(σk))
    end
    return reduce(vcat, grads)
end
∇tσ_logpdf_y(gm::RFFGM, Y_std::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}) =
    ∇tσ_logpdf_y(gm.gp, Y_std, W, σ)
# TESTED
function ∇tσ_logpdf_σ(gp::Union{Vector{RFFGP},Vector{GP}}, σ::AbstractVector{<:Real}, transformed_σ::AbstractVector{<:Real})
    grads = Float64[]
    for (gpk, σk, transformed_σk) in zip(gp, σ, transformed_σ)
        push!(grads, gradlogpdf(gpk.tσ.prior, transformed_σk))
    end
    return grads
end
∇tσ_logpdf_σ(gm::Union{RFFGM,GPGM}, σ::AbstractVector{<:Real}, transformed_σ::AbstractVector{<:Real}) =
    ∇tσ_logpdf_σ(gm.gp, σ, transformed_σ)

# --- θ ---
# TESTED
function ∇tθ_logpdf_θ(odegrad::ODEGrad, θ::AbstractVector{<:Real}, transformed_θ::AbstractVector{<:Real})
    # assumed to be used whithin HMC
    @unpack tθ = odegrad
    grads = Float64[]
    for (tθi, θi, transformed_θi) in zip(tθ, θ, transformed_θ)
        push!(grads, gradlogpdf(tθi.prior, transformed_θi))
    end
    return grads
end
∇tθ_logpdf_θ(gm::Union{RFFGM,GPGM}, θ::AbstractVector{<:Real}, transformed_θ::AbstractVector{<:Real}) =
    ∇tθ_logpdf_θ(gm.odegrad, θ, transformed_θ)

# TESTED
function ∇tθ_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{GP},
    X::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    γ::T
) where {T<:Real}
    X_destandardized = calc_destandardized_X(gp, X)
    y_std = get_y_std(gp)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ y_std  # K x N
    ẋgp_mean = dfdt_mean(gp, X)  # K length Vector of N length Vectors
    ẋgp_cov = dfdt_cov(gp)  # K length Vector of N x N Matrices
    ∇ẋgp_e = []
    for (k, ẋode_k) in enumerate(eachrow(ẋode))
        e = ẋgp_mean[k] - ẋode_k  # gradient error
        e_cov = Hermitian(ẋgp_cov[k] + γ^2 * LinearAlgebra.I)
        push!(∇ẋgp_e, gradlogpdf(MvNormal(zeros(length(e)), e_cov), e))  # N length Vector
    end
    ∇ẋgp_e = reduce(vcat, ∇ẋgp_e')  # K x N
    ∇ẋode_e = - ∇ẋgp_e  # K x N
    ∇θ_ẋode = eval_dẋdθ(odegrad, X_destandardized, θ, y_std)  # K x n(θ) x N
    ∇tθ_e = []
    for (i, θi) in enumerate(θ)
        ∇θk_e = sum(∇ẋode_e .* ∇θ_ẋode[:,i,:])  # scalar
        ∇tθk_e = ∇θk_e / odegrad.tθ[i].dtv_dv(θi)  # scalar
        push!(∇tθ_e, ∇tθk_e)
    end
    return ∇tθ_e
end
∇tθ_ulogpdf_e(gm::GPGM, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ∇tθ_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ)

# TESTED
function ∇tθ_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{RFFGP},
    X::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    γ::T
) where {T<:Real}
    X_destandardized = calc_destandardized_X(gp, X)
    y_std = get_y_std(gp)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ y_std  # K x N
    ẋgp_mean = dfdt_mean(gp, W)  # K length Vector of N length Vectors
    ẋgp_cov = dfdt_cov(gp)  # K length Vector of N x N Matrices
    ∇ẋgp_e = []
    for (k, ẋode_k) in enumerate(eachrow(ẋode))
        e = ẋgp_mean[k] - ẋode_k  # gradient error
        e_cov = Hermitian(ẋgp_cov[k] + γ^2 * LinearAlgebra.I)
        push!(∇ẋgp_e, gradlogpdf(MvNormal(zeros(length(e)), e_cov), e))  # N length Vector
    end
    ∇ẋgp_e = reduce(vcat, ∇ẋgp_e')  # K x N
    ∇ẋode_e = - ∇ẋgp_e  # K x N
    ∇θ_ẋode = eval_dẋdθ(odegrad, X_destandardized, θ, y_std)  # K x n(θ) x N
    ∇tθ_e = []
    for (i, θi) in enumerate(θ)
        ∇θk_e = sum(∇ẋode_e .* ∇θ_ẋode[:,i,:])  # scalar
        ∇tθk_e = ∇θk_e / odegrad.tθ[i].dtv_dv(θi)  # scalar
        push!(∇tθ_e, ∇tθk_e)
    end
    return ∇tθ_e
end
∇tθ_ulogpdf_e(gm::RFFGM, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ∇tθ_ulogpdf_e(gm.odegrad, gm.gp, X, W, θ, γ)

# --- γ ---
# TESTED
∇tγ_logpdf_γ(odegrad::ODEGrad, γ::T1, transformed_γ::T2) where {T1<:Real,T2<:Real} =
    [gradlogpdf(odegrad.tγ.prior, transformed_γ)]
∇tγ_logpdf_γ(gm::Union{RFFGM,GPGM}, γ::T1, transformed_γ::T2) where {T1<:Real,T2<:Real} =
    ∇tγ_logpdf_γ(gm.odegrad, γ, transformed_γ)

function gradlogpdf_dΣ(dist::MvNormal, y::AbstractVector{<:Real})
    @unpack μ, Σ = dist
    diff = y - μ
    Σ_inv = inv(Σ)
    grad = 0.5 * (Σ_inv * (diff * diff') * Σ_inv - Σ_inv)  # N x N
    return grad
end

# TESTED
function ∇tγ_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{GP},
    X::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    γ::T
) where {T<:Real}
    X_destandardized = calc_destandardized_X(gp, X)
    y_std = get_y_std(gp)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ y_std  # K x N
    ẋgp_mean = dfdt_mean(gp, X)  # K length Vector of N length Vectors
    ẋgp_cov = dfdt_cov(gp)  # K length Vector of N x N Matrices
    ∇tγ_e = []
    for (k, ẋode_k) in enumerate(eachrow(ẋode))
        e = ẋgp_mean[k] - ẋode_k  # gradient error
        e_cov = Hermitian(ẋgp_cov[k] + γ^2 * LinearAlgebra.I)
        ∇Σ_e = gradlogpdf_dΣ(MvNormal(zeros(length(e)), e_cov), e)  # N x N
        push!(∇tγ_e, sum(diag(∇Σ_e) * 2*γ / odegrad.tγ.dtv_dv(γ)))
    end
    return [sum(∇tγ_e)]
end
∇tγ_ulogpdf_e(gm::GPGM, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ∇tγ_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ)

# TESTED
function ∇tγ_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{RFFGP},
    X::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    γ::T
) where {T<:Real}
    X_destandardized = calc_destandardized_X(gp, X)
    y_std = get_y_std(gp)
    ẋode = eval_ẋ(odegrad, X_destandardized, θ) ./ y_std  # K x N
    ẋgp_mean = dfdt_mean(gp, W)  # K length Vector of N length Vectors
    ẋgp_cov = dfdt_cov(gp)  # K length Vector of N x N Matrices
    ∇tγ_e = []
    for (k, ẋode_k) in enumerate(eachrow(ẋode))
        e = ẋgp_mean[k] - ẋode_k  # gradient error
        e_cov = Hermitian(ẋgp_cov[k] + γ^2 * LinearAlgebra.I)
        ∇Σ_e = gradlogpdf_dΣ(MvNormal(zeros(length(e)), e_cov), e)  # N x N
        push!(∇tγ_e, sum(diag(∇Σ_e) * 2*γ / odegrad.tγ.dtv_dv(γ)))
    end
    return [sum(∇tγ_e)]
end
∇tγ_ulogpdf_e(gm::RFFGM, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T) where {T<:Real} =
    ∇tγ_ulogpdf_e(gm.odegrad, gm.gp, X, W, θ, γ)

# --- ϕ ---
# TODO
function ∇tϕ_logpdf_ϕ(gp::Union{Vector{GP},Vector{RFFGP}}, ϕ::AbstractMatrix{<:Real})
    grads = AbstractVector{<:Real}[]
    for (gpk, ϕk) in zip(gp, ϕ)
        grads_k = Float64[]
        for (tϕki, ϕki) in zip(gpk.tϕ, ϕk)
            push!(grads_k, gradlogpdf(tϕki.prior, calc_tvar(tϕki, ϕki)))
        end
        push!(grads, grads_k)
    end
    return reduce(vcat, grads)
end
∇tϕ_logpdf_ϕ(gm::Union{RFFGM,GPGM}, ϕ::AbstractMatrix{<:Real}) = ∇tϕ_logpdf_ϕ(gm.gp, ϕ)

# TODO
function ∇tϕ_logpdf_y(
    gp::Vector{GP}, Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, ϕ::AbstractMatrix{<:Real}
)
    error("Not implemented")
    ∇ϕ_logpdf_y = ∇tx_logpdf_y(gp, Y_std, X, σ) * ∇ϕ_x(gp, X, ϕ)
    ∇t𝓁_ϕ = eval_dϕd𝓁(gp, X, ϕ)
    ∇tα_ϕ = eval_dϕdα(gp, X, ϕ)
    grads = Float64[∇ϕ_logpdf_y * ∇t𝓁_ϕ, ∇ϕ_logpdf_y * ∇tα_ϕ]
    return grads
end
∇tϕ_logpdf_y(gm::GPGM, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, ϕ::AbstractMatrix{<:Real}) =
    ∇tϕ_logpdf_y(gm.gp, X, σ, ϕ)

# TODO
function ∇tϕ_logpdf_y(gp::Vector{RFFGP},
    Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, ϕ::AbstractMatrix{<:Real}
)
    error("Not implemented")
    ∇ϕ_logpdf_y =  ∇w_logpdf_y(gp, Y_std, W, σ) * ∇ϕ_w(gp, X, ϕ)
    ∇t𝓁_ϕ = eval_dϕd𝓁(gp, X, ϕ)
    ∇tα_ϕ = eval_dϕdα(gp, X, ϕ)
    grads = Float64[∇ϕ_logpdf_y * ∇t𝓁_ϕ, ∇ϕ_logpdf_y * ∇tα_ϕ]
    return grads
end
∇tϕ_logpdf_y(gm::RFFGM,
    Y_std::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real}, ϕ::AbstractMatrix{<:Real}) =
    ∇tϕ_logpdf_y(gm.gp, X, σ, ϕ)

# TODO
function ∇tϕ_logpdf_x(gp::Vector{GP}, X::AbstractMatrix{<:Real}, ϕ::AbstractMatrix{<:Real})
    error("Not implemented")
    ∇ϕ_logpdf_x =  ∇tx_logpdf_x(gp, X, σ) * ∇ϕ_x(gp, X, ϕ)
    ∇t𝓁_ϕ = eval_dϕd𝓁(gp, X, ϕ)
    ∇tα_ϕ = eval_dϕdα(gp, X, ϕ)
    grads = Float64[∇ϕ_logpdf_x * ∇t𝓁_ϕ, ∇ϕ_logpdf_x * ∇tα_ϕ]
    return grads
end
∇tϕ_logpdf_x(gm::GPGM, X::AbstractMatrix{<:Real}, ϕ::AbstractMatrix{<:Real}) =
    ∇tϕ_logpdf_x(gm.gp, X, ϕ)

# TODO
function ∇tϕ_logpdf_x(gp::Vector{RFFGP}, X::AbstractMatrix{<:Real}, ϕ::AbstractMatrix{<:Real})
    error("Not implemented")
    ∇ϕ_logpdf_x =  ∇w_logpdf_x(gp, X, σ) * ∇ϕ_w(gp, X, ϕ)
    ∇t𝓁_ϕ = eval_dϕd𝓁(gp, X, ϕ)
    ∇tα_ϕ = eval_dϕdα(gp, X, ϕ)
    grads = Float64[∇ϕ_logpdf_x * ∇t𝓁_ϕ, ∇ϕ_logpdf_x * ∇tα_ϕ]
    return grads
end
∇tϕ_logpdf_x(gm::RFFGM, X::AbstractMatrix{<:Real}, ϕ::AbstractMatrix{<:Real}) =
    ∇tϕ_logpdf_x(gm.gp, X, ϕ)

# TODO
function ∇tϕ_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{GP}, 
    X::AbstractMatrix{<:Real}, 
    θ::AbstractVector{<:Real},
    γ::T,
    ϕ::AbstractMatrix{<:Real}
) where {T<:Real}
    error("Not implemented")
    ∇ϕ_ulogpdf_e =  ∇tx_ulogpdf_e(odegrad, gp, X, θ, γ) * ∇ϕ_x(gp, X, ϕ)
    ∇t𝓁_ϕ = eval_dϕd𝓁(gp, X, ϕ)
    ∇tα_ϕ = eval_dϕdα(gp, X, ϕ)
    grads = Float64[∇ϕ_ulogpdf_e * ∇t𝓁_ϕ, ∇ϕ_ulogpdf_e * ∇tα_ϕ]
    return grads
end
∇tϕ_ulogpdf_e(gm::GPGM, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T, ϕ::AbstractMatrix{<:Real}) where {T<:Real} =
    ∇tϕ_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ, ϕ)

# TODO
function ∇tϕ_ulogpdf_e(
    odegrad::ODEGrad,
    gp::Vector{GP}, 
    X::AbstractMatrix{<:Real}, 
    W::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    γ::T,
    ϕ::AbstractMatrix{<:Real}
) where {T<:Real}
    error("Not implemented")
    ∇ϕ_ulogpdf_e =  ∇w_ulogpdf_e(odegrad, gp, X, W, θ, γ) * ∇ϕ_w(gp, X, ϕ)
    ∇t𝓁_ϕ = eval_dϕd𝓁(gp, X, ϕ)
    ∇tα_ϕ = eval_dϕdα(gp, X, ϕ)
    grads = Float64[∇ϕ_ulogpdf_e * ∇t𝓁_ϕ, ∇ϕ_ulogpdf_e * ∇tα_ϕ]
    return grads
end
∇tϕ_ulogpdf_e(gm::RFFGM, X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, γ::T, ϕ::AbstractMatrix{<:Real}) where {T<:Real} =
    ∇tϕ_ulogpdf_e(gm.odegrad, gm.gp, X, W, θ, γ, ϕ)
