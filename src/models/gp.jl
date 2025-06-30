"""
Gaussian Process regression with inducing points.
"""
mutable struct GP
    z::Vector{Float64}               # inducing time points
    u::Vector{Float64}               # latent states at inducing time points
    x::Vector{Float64}               # training time points
    y::Vector{Float64}               # observations at training time points
    y_mean::Float64                  # mean of the observations
    y_std::Float64                   # std of the observations
    y_standardized::Vector{Float64}  # standardized observations
    σᵤ::Float64                      # state noise std
    σ::Float64                       # observation noise std
    tσ::PriorTransformation          # transform for σ
    k::KernelFunctions.Kernel        # kernel
    tϕ::Vector{PriorTransformation}  # transforms for the kernel parameters
    f::AbstractGPs.GP                # f = GP(k)
    fz::AbstractGPs.FiniteGP         # fz = f(z)
    f′::AbstractGPs.PosteriorGP      # f′ = posterior(fz, u)
    f′x::AbstractGPs.FiniteGP        # f′x = f′(x)
    K::Matrix{Float64}               # kernel matrix
    K⁻¹::Matrix{Float64}             # inverse of the kernel matrix
    KᵀK⁻¹::Matrix{Float64}           # K' * inv(K+σᵤ²I)
    K̂::Matrix{Float64}               # kernel matrix for posterior predictive
    L::Matrix{Float64}               # Cholesky decomposition of the kernel matrix
    L⁻¹::Matrix{Float64}             # inverse of the Cholesky decomposition of the kernel matrix
    K′::Matrix{Float64}              # first derivative of the kernel with respect to the input 
    K″::Matrix{Float64}              # second derivative of the kernel with respect to the input
    K′ᵀK⁻¹::Matrix{Float64}          # K′' * inv(K)
    standardize::Bool                # whether to standardize the observations
    centralize::Bool                 # whether to centralize the observations

    function GP(
        z::Vector{Float64},
        u::Vector{Float64},
        x::Vector{Float64},
        y::Vector{Float64},
        σᵤ::Float64,
        σ::Float64;
        k::KernelFunctions.Kernel,
        standardize::Bool=false,
        centralize::Bool=false
    )
        tσ = PriorTransformation(Normal(0, 1), log)
        ϕ = only_params(k)
        tϕ = [PriorTransformation(Normal(0, 1), log) for (i, ϕi) in enumerate(ϕ)]
        # GP prior
        f = AbstractGPs.GP(k)
        fz = f(z, σᵤ^2)
        f′ = AbstractGPs.posterior(fz, u)
        f′x = f′(x, σ^2)
        K = cov(fz)
        K⁻¹ = inv(K)
        KᵀK⁻¹ = KernelFunctions.kernelmatrix(k, x, z) * K⁻¹
        K̂ = KernelFunctions.kernelmatrix(k, x, x) - KᵀK⁻¹ * KernelFunctions.kernelmatrix(k, z, x)
        # NOTE: if x contains same points as z, 
        # diag(K̂) may have negative values due to numerical errors.
        # Such elements are replaced with 1e-10 to make K̂ positive semi-definite.
        K̂[diagind(K̂)] .= max.(diag(K̂), 0) .+ 1e-10 
        
        L = cholesky(K).L
        L⁻¹ = inv(L)
        K′ = eval_dKdt(k, σᵤ, z)
        K″ = eval_d²Kdt²(k, σᵤ, z)
        K′ᵀK⁻¹ = K′' * K⁻¹

        if standardize
            y_mean = StatsBase.mean(y)
            y_std = StatsBase.std(y)
            y_standardized = (y .- y_mean) ./ y_std
        elseif centralize
            y_mean = StatsBase.mean(y)
            y_std = 1.0
            y_standardized = y .- y_mean
        else
            y_mean = 0.0
            y_std = 1.0
            y_standardized = y
        end

        return new(z, u, x, y, y_mean, y_std, y_standardized, σᵤ, σ, tσ, k, 
            tϕ, f, fz, f′, f′x, K, K⁻¹, KᵀK⁻¹, K̂, L, L⁻¹, K′, K″, K′ᵀK⁻¹, standardize, centralize)
    end
end

function reconstruct_kernel(k::KernelFunctions.Kernel, ϕ::AbstractVector{<:Real})
    base_k, inner, outer = params(k)
    new_inner = ϕ[1:end-1]
    new_outer = ϕ[end]
    k_new = new_outer^2 * kernel_inner(base_k, new_inner)
    return k_new
end

kernel_inner(::SqExponentialKernel, inner::AbstractVector{<:Real}) = with_lengthscale(SqExponentialKernel(), inner[1])
kernel_inner(::Matern52Kernel, inner::AbstractVector{<:Real}) = with_lengthscale(Matern52Kernel(), inner[1])
kernel_inner(::SigmoidKernel, inner::AbstractVector{<:Real}) = SigmoidKernel(inner...)

function reconstruct_gp(gp::GP;
    ϕ::Union{Nothing,AbstractVector{<:Real}}=nothing,
    u::Union{Nothing,AbstractVector{<:Real}}=nothing,
    σ::Union{Nothing,Float64}=nothing
)
    _ϕ = isnothing(ϕ) ? only_params(gp.k) : ϕ
    _u = isnothing(u) ? gp.u : u
    _σ = isnothing(σ) ? gp.σ : σ

    if isnothing(ϕ) && isnothing(u) && !isnothing(σ)
        gp.σ = _σ
        gp.f′x = gp.f′(gp.x, gp.σ^2)
    elseif isnothing(ϕ) && !isnothing(u)
        gp.f′ = AbstractGPs.posterior(gp.fz, _u)
        gp.σ = _σ
        gp.f′x = gp.f′(gp.x, gp.σ^2)
    elseif !isnothing(ϕ)
        gp = GP(gp.z, _u, gp.x, gp.y, gp.σᵤ, _σ;
            k=reconstruct_kernel(gp.k, _ϕ), 
            standardize=gp.standardize, 
            centralize=gp.centralize)
    else
        @error "There is no update in the GP."
    end
    return gp
end

function reconstruct_gp(gp::Vector{GP}; ϕ::AbstractMatrix{<:Real})
    gp_reconstructed = GP[]
    for (gpk, ϕk) in zip(gp, eachcol(ϕ))
        push!(gp_reconstructed, reconstruct_gp(gpk, ϕ=ϕk[:]))
    end
    return gp_reconstructed
end

function f_conditional(f::AbstractGPs.GP, z::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    fz = f(z, 1e-6)
    f′ = AbstractGPs.posterior(fz, u)
    return f′
end

function update_u!(gp::GP, u::AbstractVector{<:Real})
    @assert length(u) == length(gp.z)
    gp.u[:] = u
    gp.f′ = AbstractGPs.posterior(gp.fz, u)
end

function update_y!(gp::GP, y::Vector{Float64})
    @assert length(y) == length(gp.x)
    gp.y[:] = y
    if gp.standardize
        gp.y_mean = StatsBase.mean(y)
        gp.y_std = StatsBase.std(y)
    elseif gp.centralize
        gp.y_mean = StatsBase.mean(y)
        gp.y_std = 1.0
    else
        gp.y_mean = 0.0
        gp.y_std = 1.0
    end
    gp.y_standardized = (y .- gp.y_mean) ./ gp.y_std
end

function calc_y_mean_and_diagcov(gp::GP, x::AbstractVector{<:Real}, σ::Real)
    if gp.z == gp.x  # in case where inducing points are the same as training points
        y_mean = x
        y_cov = Diagonal(σ^2 * ones(length(x)))
    else  # in case where inducing points are different from training points
        y_mean = gp.KᵀK⁻¹ * x
        y_cov = Diagonal(gp.K̂) + σ^2 * LinearAlgebra.I
    end
    return y_mean, y_cov
end

# predict the mean and variance of the GP at the given time points
f_predictive(gp::GP, t_test::AbstractVector{<:Real}, sd::Float64) = gp.f′(t_test, sd)

" cross-covariances between the kth state and its derivative"
function eval_dKdt(
    k::Tk, inner::Union{Float64,Tuple{Float64,Float64}}, outer::Float64, σ::Float64, t::AbstractVector{<:Real}
) where {Tk<:Union{SqExponentialKernel, Matern52Kernel, SigmoidKernel}}
    function _dktt′_dt(k::SqExponentialKernel, 𝓁::Float64, α::Float64, σ::Float64)
        _k = α^2 * with_lengthscale(k, 𝓁) + σ^2 * WhiteKernel()
        dkdt = (t, t′) -> ForwardDiff.derivative(t -> _k(t, t′), t)
        dkdt
    end
    function _dktt′_dt(k::Matern52Kernel, 𝓁::Float64, α::Float64, σ::Float64)
        function __dktt′_dt(t::Float64, t′::Float64)
            r = abs(t - t′)
            dk_dr = -α^2 * 5/3/𝓁^2*(t-t′) * (1+√5*r/𝓁) * exp(-√5*r/𝓁)
            dk_dr
        end
        dkdt = (t, t′) -> __dktt′_dt(t, t′)
        dkdt
    end
    function _dktt′_dt(k::SigmoidKernel, inner::Tuple{Float64,Float64}, α::Float64, σ::Float64)
        b, a = inner
        _k = α^2 * SigmoidKernel(b, a) + σ^2 * WhiteKernel()
        dkdt = (t, t′) -> ForwardDiff.derivative(t -> _k(t, t′), t)
        dkdt
    end

    dktt′_dt = _dktt′_dt(k, inner, outer, σ)
    dKdt = [dktt′_dt(t_i, t_j) for t_j in t, t_i in t]  # matrix
    return dKdt
end

eval_dKdt(
    k::KernelFunctions.Kernel, inner::Union{Float64,Tuple{Float64,Float64}}, outer::Float64, σ::Float64, t::AbstractVector{<:Real}
) = error("Only support SqExponentialKernel, Matern52Kernel and SigmoidKernel. Not implemented for kernel:\n$k.")

function eval_dKdt(k::KernelFunctions.Kernel, noise_std::Float64, t::AbstractVector{<:Real})
    _k, inner, outer = params(k)
    return eval_dKdt(_k, inner, outer, noise_std, t)
end

"the auto-covariance for each state derivative"
function eval_d²Kdt²(
    k::Tk, inner::Union{Float64,Tuple{Float64,Float64}}, outer::Float64, σ::Float64, t::AbstractVector{<:Real}
) where {Tk<:Union{SqExponentialKernel, Matern52Kernel, SigmoidKernel}}
    function _d²ktt′_dtdt′(k::SqExponentialKernel, 𝓁::Float64, α::Float64, σ::Float64)
        k = α^2 * with_lengthscale(k, 𝓁) + σ^2 * WhiteKernel()
        d2k_dtdt′ = (t, t′) -> ForwardDiff.derivative(ξ -> ForwardDiff.derivative(η -> k(ξ, η), t′), t)
    end
    function _d²ktt′_dtdt′(k::Matern52Kernel, 𝓁::Float64, α::Float64, σ::Float64)
        function __d²ktt′_dtdt′(t::Float64, t′::Float64)
            r = abs(t - t′)
            d2k_dr2 = α^2 * 5/3/𝓁^2 * (1 + √5*r/𝓁 - 5*r^2/𝓁^2) * exp(-√5*r/𝓁)
            d2k_dr2
        end
        d2k_dtdt′ = (t, t′) -> __d²ktt′_dtdt′(t, t′)
    end
    function _d²ktt′_dtdt′(k::SigmoidKernel, inner::Tuple{Float64,Float64}, α::Float64, σ::Float64)
        b, a = inner
        k = α^2 * SigmoidKernel(b, a) + σ^2 * WhiteKernel()
        d2k_dtdt′ = (t, t′) -> ForwardDiff.derivative(ξ -> ForwardDiff.derivative(η -> k(ξ, η), t′), t)
    end
    
    d²ktt′_dtdt′ = _d²ktt′_dtdt′(k, inner, outer, σ)
    d²Kdt² = [d²ktt′_dtdt′(t_i, t_j) for t_j in t, t_i in t]  # matrix
    return d²Kdt²
end

eval_d²Kdt²(
    k::KernelFunctions.Kernel, inner::Union{Float64,Tuple{Float64,Float64}}, outer::Float64, σ::Float64, x::AbstractVector{<:Real}
) = error("Only support SqExponentialKernel, Matern52Kernel and SigmoidKernel. Not implemented for kernel:\n$k.")

function eval_d²Kdt²(k::KernelFunctions.Kernel, noise_std::Float64, t::AbstractVector{<:Real})
    _k, inner, outer = params(k)
    return eval_d²Kdt²(_k, inner, outer, noise_std, t)
end

eval_dKdα(k::SqExponentialKernel, 𝓁::Float64, α::Float64, σ::Float64, t::AbstractVector{<:Real}) = error("Not implemented")
eval_dKdα(k::Matern52Kernel, 𝓁::Float64, α::Float64, σ::Float64, t::AbstractVector{<:Real}) = error("Not implemented")
eval_dKdα(k::SigmoidKernel, b::Float64, a::Float64, α::Float64, σ::Float64, t::AbstractVector{<:Real}) = error("Not implemented")

eval_dKd𝓁(k::SqExponentialKernel, 𝓁::Float64, α::Float64, σ::Float64, t::AbstractVector{<:Real}) = error("Not implemented")
eval_dKd𝓁(k::Matern52Kernel, 𝓁::Float64, α::Float64, σ::Float64, t::AbstractVector{<:Real}) = error("Not implemented")

eval_dKdb(k::SigmoidKernel, b::Float64, a::Float64, α::Float64, σ::Float64, t::AbstractVector{<:Real}) = error("Not implemented")
eval_dKda(k::SigmoidKernel, b::Float64, a::Float64, α::Float64, σ::Float64, t::AbstractVector{<:Real}) = error("Not implemented")

function params(k::KernelFunctions.ScaledKernel)
    σ² = only(k.σ²)
    _k, inner, outer = params(k.kernel)
    return _k, inner, outer * √σ²
end
function params(k::KernelFunctions.TransformedKernel)
    s = only(k.transform.s)
    _k, inner, outer = params(k.kernel)
    return _k, inner / s, outer
end
params(::SqExponentialKernel) = (SqExponentialKernel(), 1.0, 1.0)
params(::Matern52Kernel) = (Matern52Kernel(), 1.0, 1.0)
params(k::SigmoidKernel) = (SigmoidKernel(), (k.b, k.a), 1.0)
params(::KernelFunctions.Kernel) = error("Not implemented for kernel:\n$k")

only_params(k::KernelFunctions.ScaledKernel) = params(k)[2:end] |> collect
only_params(k::KernelFunctions.TransformedKernel) = params(k)[2:end] |> collect
only_params(k::KernelFunctions.Kernel) = error("Not implemented for kernel:\n$k")
only_params(::SqExponentialKernel) = params(SqExponentialKernel())[2:end] |> collect
only_params(::Matern52Kernel) = params(Matern52Kernel())[2:end] |> collect
only_params(::SigmoidKernel) = params(SigmoidKernel())[2:end] |> collect

cov_inducing(gp::GP) = cov(gp.fz)

# gradient of the GP function
dfdt_mean(gp::GP, x::AbstractVector{<:Real}) = gp.K′ᵀK⁻¹ * x
dfdt_mean(gp::GP) = dfdt_mean(gp, gp.u)
dfdt_mean(gp::Vector{GP}, X::AbstractMatrix{<:Real}) = [dfdt_mean(gp[k], xk[:]) for (k, xk) in enumerate(eachrow(X))]

dfdt_cov(gp::GP) = gp.K″ - gp.K′' * (cov(gp.fz) \ gp.K′)
dfdt_cov(gp::Vector{GP}) = [dfdt_cov(gp[i]) for i in 1:length(gp)]

# logpdf functions
logpdf_f(gp::GP) = logpdf(gp.fz, gp.u)
logpdf_f′(gp::GP, t::AbstractVector{<:Real}, y::AbstractVector{<:Real}, sd::Float64) = logpdf(f_predictive(gp, t, sd), y)  #TODO

# derivatives of the logpdf functions
# gradlogpdf_f(gp::GP) = gradlogpdf(rgp.f′(RowVecs(t[:,:]), sd), y)  # TODO
# gradlogpdf_f(gp::GP) = gradlogpdf(rgp.f′(RowVecs(t[:,:]), sd), y)  # TODO
# gradlogpdf_f′_w(rgp::RFFGP, t::AbstractVector{<:Real}, y::AbstractVector{<:Real}, sd::Float64) = gradlogpdf(rgp.f′(RowVecs(t[:,:]), sd), y)  # TODO

function plot(gp::GP)
    times = gp.z
    diff_t = times[end]-times[1]
    t_test = collect((times[1]-diff_t/20):diff_t/100:(times[end]+diff_t/20))
    Plots.plot(gp.f′(t_test, gp.σ), ribbon_scale=3, label="f(t)", xlabel="t")
    Plots.scatter!(gp.z, gp.u, c=:blue, label="x")
    Plots.scatter!(gp.z, gp.y, c=:red, label="y")
    Plots.title!("GP regression")
end

function plot_graddist(gp::GP)
    grad_mean = dfdt_mean(gp)
    σ_vec = sqrt.(diag(dfdt_cov(gp)))
    upper = grad_mean .+ 3 * σ_vec
    lower = grad_mean .- 3 * σ_vec
    Plots.scatter(gp.z, grad_mean, c=:blue, label="𝔼[df/dt]", ms=1, xlabel="t")
    Plots.bar!(gp.z, upper, fillrange=lower, fillalpha=0.5, label="±3σ", c=:lightblue)
    Plots.hline!([0], c=:black, label="", ls=:dash)
    Plots.title!("Gradient distribution of GP")
end

function plot_gradcov(gp::GP; clims=(-1., 1.))
    grad_cov = dfdt_cov(gp)
    mid = sum(clims) / 2
    Plots.heatmap(
        grad_cov, y_flip=true, clims=clims,
        c=cgrad([:blue, :white, :red], [clims[1], mid, clims[2]]))
    Plots.title!("Cov[df/dt, df/dt] in GP")
end
