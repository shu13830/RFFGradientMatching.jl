# MAGI gradient of log-posterior density
# Reuses existing gradient functions from gradmatch_gradlogpdf.jl
# γ is stored in odegrad.γ (shared with GPGM/RFFGM)

function ∇ulogpdf(gm::MAGI, param_dict::Dict{Symbol,Any}, sample_target::Vector{Symbol})

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
    θ = calc_θ(gm.odegrad, transformed_θ)
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
            ge_dx = gm.β[1] * ∇tx_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ)
        end
        @assert length(gx_dx) == length(gy_dx) == length(ge_dx)
        push!(grad, gx_dx .+ gy_dx .+ ge_dx)
    end
    if :θ in sample_target
        gθ_dtθ = ∇tθ_logpdf_θ(gm.odegrad, θ, transformed_θ)
        if gm.β[1] == 0.0
            ge_dtθ = zeros(length(gθ_dtθ))
        else
            ge_dtθ = gm.β[1] * ∇tθ_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ)
        end
        @assert length(gθ_dtθ) == length(ge_dtθ)
        push!(grad, gθ_dtθ .+ ge_dtθ)
    end
    if :γ in sample_target
        gγ_dtγ = ∇tγ_logpdf_γ(gm.odegrad, γ, transformed_γ)
        if gm.β[1] == 0.0
            ge_dtγ = zeros(length(gγ_dtγ))
        else
            ge_dtγ = gm.β[1] * ∇tγ_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ)
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
        gx_dtϕ = ∇tϕ_logpdf_x(gp, X, ϕ)
        gy_dtϕ = ∇tϕ_logpdf_y(gp, Y_std, X, σ, ϕ)
        gϕ_dtϕ = ∇tϕ_logpdf_ϕ(gp, ϕ)
        if gm.β[1] == 0.0
            ge_dtϕ = zeros(length(gx_dtϕ))
        else
            ge_dtϕ = gm.β[1] * ∇tϕ_ulogpdf_e(gm.odegrad, gm.gp, X, θ, γ, ϕ)
        end
        @assert length(gx_dtϕ) == length(gy_dtϕ) == length(gϕ_dtϕ) == length(ge_dtϕ)
        push!(grad, gx_dtϕ .+ gy_dtϕ .+ gϕ_dtϕ .+ ge_dtϕ)
    end

    if length(grad) > 0
        grad = reduce(vcat, grad)
    end
    return grad
end

∇ulogpdf(gm::MAGI, sample_target::Vector{Symbol}) = ∇ulogpdf(gm, pack_param_dict(gm), sample_target)

function ∇ulogpdf(param_vec::AbstractVector{<:Real}, gm::MAGI, sample_target::Vector{Symbol})
    return ∇ulogpdf(gm, pack_param_dict_from_vec(gm, param_vec, sample_target), sample_target)
end
