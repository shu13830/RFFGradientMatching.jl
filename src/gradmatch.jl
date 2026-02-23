abstract type AbstractGM end

struct RFFGM <: AbstractGM
    odegrad::ODEGrad
    gp::Vector{RFFGP}

    β::Vector{Float64}  # inverse temperature to weigt the impact of the gradient matching
    β_schedule::Vector{Float64}
    anneal_length::Int
    anneal_iter::Vector{Int}

    function RFFGM(odegrad::ODEGrad, gp::Vector{RFFGP}, anneal_length::Int)
        β_schedule = collect(0:1/anneal_length:1)[2:end]
        β = [β_schedule[1]]
        anneal_iter = [1]
        @assert length(β) == 1
        @assert length(anneal_iter) == 1
        new(odegrad, gp, β, β_schedule, anneal_length, anneal_iter)
    end

    function RFFGM(
        times::Vector{Float64}, obs::Matrix{Float64}, prob::ODEProblem, probname::String;
        k::KernelFunctions.Kernel,
        state_noise_std::Float64,
        obs_noise_std::Float64,
        n_rff::Union{Int, Nothing}=nothing,
        inducing_points::Union{Nothing, Vector{Float64}}=nothing,
        standardize::Bool=false,
        centralize::Bool=false,
        anneal_length::Int=1000
    )
        @assert length(times) == size(obs, 2)
        n_rff = n_rff == nothing ? 100 : n_rff
        n_gps = size(obs, 1)

        odegrad = ODEGrad(obs, prob, probname)
        if isnothing(inducing_points)
            gp = [
                RFFGP(
                    times, deepcopy(obs[i,:]), times, obs[i,:], state_noise_std, obs_noise_std;
                    k=k, n_rff=n_rff, standardize=standardize, centralize=centralize)
                for i in 1:n_gps
            ]
        else
            gp = [
                RFFGP(
                    inducing_points, zeros(length(inducing_points)), times, obs[i,:], state_noise_std, obs_noise_std;
                    k=k, n_rff=n_rff, standardize=standardize, centralize=centralize)
                for i in 1:n_gps
            ]
        end

        RFFGM(odegrad, gp, anneal_length)
    end
end

Base.show(io::IO, gm::RFFGM) = print(io, "Random Feature-based Gradient Matching
    ODE problem: $(gm.odegrad.functions.probname)
    #states in the ODE: $(length(gm.gp))
    #ODE parameters: $(gm.odegrad.θ |> length)
    Regressor: Random Fourier Features-based Gaussian Process
        #RFFs: $(gm.gp[1].n_rff)
")

struct GPGM <: AbstractGM
    odegrad::ODEGrad
    gp::Vector{GP}
    β::Vector{Float64}  # inverse temperature to weigt the impact of the gradient matching
    β_schedule::Vector{Float64}
    anneal_length::Int
    anneal_iter::Vector{Int}

    function GPGM(odegrad::ODEGrad, gp::Vector{GP}, anneal_length::Int)
        β_schedule = collect(0:1/anneal_length:1)[2:end]
        β = [β_schedule[1]]
        anneal_iter = [1]
        @assert length(β) == 1
        @assert length(anneal_iter) == 1
        new(odegrad, gp, β, β_schedule, anneal_length, anneal_iter)
    end

    function GPGM(
        times::Vector{Float64}, obs::Matrix{Float64}, prob::ODEProblem, probname::String;
        k::KernelFunctions.Kernel,
        state_noise_std::Float64,
        obs_noise_std::Float64,
        inducing_points::Union{Nothing, Vector{Float64}}=nothing,
        standardize::Bool=false,
        centralize::Bool=false,
        anneal_length::Int=1000
    )
        @assert length(times) == size(obs, 2)
        n_gps = size(obs, 1)

        odegrad = ODEGrad(obs, prob, probname)
        if isnothing(inducing_points)
            gp = [
                GP(times, deepcopy(obs[i,:]), times, obs[i,:], state_noise_std, obs_noise_std;
                    k=k, standardize=standardize, centralize=centralize)
                for i in 1:n_gps
            ]
        else
            gp = [
                GP(inducing_points, zeros(length(inducing_points)), times, obs[i,:], state_noise_std, obs_noise_std;
                    k=k, standardize=standardize, centralize=centralize)
                for i in 1:n_gps
            ]
        end

        GPGM(odegrad, gp, anneal_length)
    end
end

Base.show(io::IO, gm::GPGM) = print(io, "Gaussian Process-based Gradient Matching
    ODE problem: $(gm.odegrad.functions.probname)
    #states in the ODE: $(length(gm.gp))
    #ODE parameters: $(gm.odegrad.θ |> length)
    Regressor: Gaussian Process
")

struct MAGI <: AbstractGM
    odegrad::ODEGrad
    gp::Vector{GP}
    β::Vector{Float64}
    β_schedule::Vector{Float64}
    anneal_length::Int
    anneal_iter::Vector{Int}
    γ_jitter::Float64  # fixed small jitter for numerical stability (replaces γ)

    function MAGI(odegrad::ODEGrad, gp::Vector{GP}, anneal_length::Int;
                  γ_jitter::Float64=1e-3)
        if anneal_length <= 1
            β_schedule = [1.0]
            β = [1.0]
            anneal_iter = [1]
        else
            β_schedule = collect(0:1/anneal_length:1)[2:end]
            β = [β_schedule[1]]
            anneal_iter = [1]
        end
        new(odegrad, gp, β, β_schedule, anneal_length, anneal_iter, γ_jitter)
    end

    function MAGI(
        times::Vector{Float64}, obs::Matrix{Float64}, prob::ODEProblem, probname::String;
        k::KernelFunctions.Kernel,
        state_noise_std::Float64,
        obs_noise_std::Float64,
        inducing_points::Union{Nothing, Vector{Float64}}=nothing,
        standardize::Bool=false,
        centralize::Bool=false,
        anneal_length::Int=1,
        γ_jitter::Float64=1e-3
    )
        @assert length(times) == size(obs, 2)
        n_gps = size(obs, 1)

        odegrad = ODEGrad(obs, prob, probname)
        if isnothing(inducing_points)
            gp = [
                GP(times, deepcopy(obs[i,:]), times, obs[i,:], state_noise_std, obs_noise_std;
                    k=k, standardize=standardize, centralize=centralize)
                for i in 1:n_gps
            ]
        else
            gp = [
                GP(inducing_points, zeros(length(inducing_points)), times, obs[i,:], state_noise_std, obs_noise_std;
                    k=k, standardize=standardize, centralize=centralize)
                for i in 1:n_gps
            ]
        end

        MAGI(odegrad, gp, anneal_length; γ_jitter=γ_jitter)
    end
end

Base.show(io::IO, gm::MAGI) = print(io, "MAGI (Manifold-constrained GP Inference)
    ODE problem: $(gm.odegrad.functions.probname)
    #states in the ODE: $(length(gm.gp))
    #ODE parameters: $(gm.odegrad.θ |> length)
    γ_jitter: $(gm.γ_jitter)
")

function set_priortransform_on_θ!(
    gm::AbstractGM,
    priors::Vector{<:Distributions.Distribution},
    transforms::Vector{<:Union{Function, Bijectors.Transform}}
)
    @assert length(priors) == length(gm.odegrad.θ)
    @assert length(transforms) == length(gm.odegrad.θ)
    for i in 1:length(gm.odegrad.θ)
        θ = gm.odegrad.θ[i]
        # check if current parameter is in the assumed domain
        try
            transforms[i](θ)
        catch DomainError
            new_θ = inverse(transforms[i])(rand(priors[i]))
            @warn "Parameter $i is out of the assumed domain. Resetting to value newly sampled from the prior."
            gm.odegrad.θ[i] = new_θ
        end
    end
    gm.odegrad.tθ = [PriorTransformation(p, t) for (p, t) in zip(priors, transforms)]
end

function set_priortransform_on_γ!(
    gm::AbstractGM,
    prior::Distributions.Distribution,
    transform::Union{Function, Bijectors.Transform}
)
    gm.odegrad.tγ = PriorTransformation(prior, transform)
end

function set_priortransform_on_σ!(
    gm::AbstractGM,
    priors::Vector{Tprior},
    transforms::Vector{Ttransform}
) where {Tprior<:Distributions.Distribution, Ttransform<:Union{Function, Bijectors.Transform}}
    for (k, gpk) in enumerate(gm.gp)
        gpk.tσ = PriorTransformation(priors[k], transforms[k])
    end
end
function set_priortransform_on_σ!(
    gm::AbstractGM,
    prior::Tprior,
    transform::Union{Function, Bijectors.Transform}
) where {Tprior<:Distributions.Distribution}
    set_priortransform_on_σ!(gm, fill(prior, length(gm.gp)), fill(transform, length(gm.gp)))
end


function set_priortransform_on_ϕ!(
    gm::Union{GPGM,MAGI},
    priors::Vector{Tprior},
    transforms::Vector{Ttransform}
) where {
    Tprior<:Distributions.Distribution, 
    Ttransform<:Union{Function, Bijectors.Transform}
}
    for gp in gm.gp
        ϕ = only_params(gp.k)
        @assert length(priors) == length(ϕ)
        @assert length(transforms) == length(ϕ)
        for (i, ϕ_i) in enumerate(ϕ)
            # check if current parameter is in the assumed domain
            try
                transforms[i](ϕ_i)
            catch DomainError
                @error "Parameter $i is out of the assumed domain."
            end
        end
        gp.tϕ = [PriorTransformation(p, t) for (p, t) in zip(priors, transforms)]
    end
end

get_standardized_Y(gm::AbstractGM) = vcat([gp.y_standardized' for gp in gm.gp]...)
get_Y(gm::AbstractGM) = vcat([gp.y' for gp in gm.gp]...)
get_X(gm::Union{GPGM,MAGI}) = vcat([gp.u' for gp in gm.gp]...)
get_transformed_X(gm::Union{GPGM,MAGI}) = vcat([(gp.L⁻¹ * gp.u)' for gp in gm.gp]...)
get_destandardized_X(gm::Union{GPGM,MAGI}) = gm.odegrad.X
get_X(gm::RFFGM) = W2X(gm.gp, get_W(gm))
get_W(gm::RFFGM) = reduce(vcat, [gpk.w' for gpk in gm.gp])
get_θ(gm::AbstractGM) = gm.odegrad.θ
get_γ(gm::AbstractGM) = gm.odegrad.γ
get_σ(gm::AbstractGM) = [gp.σ for gp in gm.gp]
get_ϕ(gm::AbstractGM) = reduce(hcat, [only_params(gp.k) for gp in gm.gp])  # n(params) × n(gp)
get_y_std(gp::Union{Vector{GP},Vector{RFFGP}}) = [gpk.y_std for gpk in gp]

function calc_standardized_Y(gp::Union{Vector{GP},Vector{RFFGP}}, Y::AbstractMatrix{<:Real})
    vcat([((Y[k,:] .- gpk.y_mean) ./ gpk.y_std)' for (k, gpk) in enumerate(gp)]...)
end

function calc_destandardized_X(gp::Union{Vector{GP},Vector{RFFGP}}, X::AbstractMatrix{<:Real})
    hcat([(X[k,:] .* gpk.y_std) .+ gpk.y_mean for (k, gpk) in enumerate(gp)]...)' |> Matrix
end

get_transformed_θ(gm::AbstractGM) = try
    calc_tvar(gm.odegrad.tθ, gm.odegrad.θ)
catch UndefRefError
    error("Prior transformation on θ is not set.")
end

get_transformed_γ(gm::AbstractGM) = try
    calc_tvar(gm.odegrad.tγ, gm.odegrad.γ)
catch UndefRefError
    error("Prior transformation on γ is not set.")
end

get_transformed_σ(gm::AbstractGM) = try
    [calc_tvar(gp.tσ, gp.σ) for gp in gm.gp]
catch UndefRefError
    error("Prior transformation on σ is not set.")
end

get_transformed_ϕ(gm::AbstractGM) = try
    reduce(hcat, [[calc_tvar(tϕi, ϕi) for (tϕi, ϕi) in zip(gp.tϕ, only_params(gp.k))] for gp in gm.gp])
catch UndefRefError
    error("Prior transformation on ϕ is not set.")
end

"""
    calc_X(gp::Union{Vector{GP},Vector{RFFGP}}, transformed_X::AbstractMatrix{<:Real})

Calculate X from the transformed X.
Transformation: transformed_x = L⁻¹ * x, where L is the Cholesky decomposition of the kernel matrix K.
Here, we calculate x from transformed_x, as x = L * transformed_x.

# Arguments
- `gp::Union{Vector{GP},Vector{RFFGP}}`: Vector of Gaussian Process objects.
- `transformed_X::AbstractMatrix{<:Real}`: Transformed X.
"""
function calc_X(gp::Vector{GP}, transformed_X::AbstractMatrix{<:Real})
    X = vcat([(gpk.L*txk)' for (gpk, txk) in zip(gp, eachrow(transformed_X))]...)
    return X
end

function calc_transformed_X(gp::Vector{GP}, X::AbstractMatrix{<:Real})
    transformed_X = vcat([(gpk.L⁻¹*xk)' for (gpk, xk) in zip(gp, eachrow(X))]...)
    return transformed_X
end

function calc_θ(odegrad::ODEGrad, transformed_θ::AbstractVector{<:Real})
    θ = [calc_var(tθ, transformed_θi) for (tθ, transformed_θi) in zip(odegrad.tθ, transformed_θ)]
    return θ
end
calc_γ(odegrad::ODEGrad, transformed_γ::T) where {T<:Real} = calc_var(odegrad.tγ, transformed_γ)
function calc_σ(gp::Union{Vector{GP},Vector{RFFGP}}, transformed_σ::AbstractVector{<:Real})
    σ = [calc_var(gpk.tσ, transformed_σk) for (gpk, transformed_σk) in zip(gp, transformed_σ)]
    return σ
end
function calc_ϕ(gp::Union{Vector{GP},Vector{RFFGP}}, transformed_ϕ::AbstractMatrix{<:Real})
    ϕ = [[calc_var(tϕki, transformed_ϕki) for (tϕki, transformed_ϕki) in zip(gpk.tϕ, transformed_ϕk)]
        for (gpk, transformed_ϕk) in zip(gp, eachcol(transformed_ϕ))]
    ϕ = reduce(hcat, ϕ)
    return ϕ
end

dfdt_mean(gm::AbstractGM) = [dfdt_mean(gp) for gp in gm.gp]
dfdt_cov(gm::AbstractGM) = [dfdt_cov(gp) for gp in gm.gp]
