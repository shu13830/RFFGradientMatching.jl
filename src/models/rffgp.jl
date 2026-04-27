"""
Random Fourier Features-based Gaussian Process regression with inducing points.
"""
mutable struct RFFGP
    z::Vector{Float64}               # inducing time points
    u::Vector{Float64}               # latent states at inducing time points
    x::Vector{Float64}               # training time points
    y::Vector{Float64}               # observations at training time points
    y_mean::Float64                  # mean of the observations
    y_std::Float64                   # std of the observations
    y_standardized::Vector{Float64}  # standardized observations
    σᵤ::Float64                      # state noise std
    σ::Float64                       # observation noise std
    tσ::PriorTransformation          # prior transform for σ
    k::KernelFunctions.Kernel        # kernel
    n_rff::Int                       # number of random fourier features
    h::RFFBasis                      # random fourier features
    tϕ::Vector{PriorTransformation}  # prior transforms for the kernel parameters
    blr::BayesianLinearRegressor     # Bayesian linear regressor
    f::BasisFunctionRegressor        # f = BasisFunctionRegressor(blr, h)
    fz::AbstractGPs.FiniteGP         # fz = f(z)
    f′::BasisFunctionRegressor       # f′ = posterior(fz, u)
    f′x::AbstractGPs.FiniteGP        # f′x = f′(x)
    H::Matrix{Float64}               # random fourier features at inducing points
    H′::Matrix{Float64}              # random fourier features at training points
    w::Vector{Float64}               # weights of random fourier features
    dHdt::Matrix{Float64}            # gradient of the random fourier features with respect to the input (rows: length(z), cols: n_rff)
    dfdt_cov_cache::Matrix{Float64}  # cached Cov[df/dt] = dHdt Λw⁻¹ dHdt'
    L::LowerTriangular{Float64, Matrix{Float64}}  # Cholesky factor of K = H Λw⁻¹ H' + σᵤ²I
    e_cov_chol::Union{Nothing, Cholesky}  # cached Cholesky of e_cov = dfdt_cov + γ²I
    standardize::Bool                # whether to standardize the observations
    centralize::Bool                 # whether to centralize the observations

    function RFFGP(
        z::Vector{Float64},
        u::Vector{Float64},
        x::Vector{Float64},
        y::Vector{Float64},
        σᵤ::Float64,
        σ::Float64;
        k::KernelFunctions.Kernel,
        n_rff::Int=100,
        standardize::Bool=false,
        centralize::Bool=false
    )

        # Default prior and transform for σ. User can change them by calling `set_priortransform_on_σ!``
        tσ = PriorTransformation(Normal(0, 1), log)
        
        # Build RFF basis (supports both standard and generalized kernels via compat.jl)
        input_dims = 1  # input dimension is only time here
        h = build_rff_basis(k, input_dims, n_rff)
        tϕ = [PriorTransformation(Normal(0, 1), log) for (i, ϕi) in enumerate(only_params(k))]

        # construct an approximated GP with random Fourier features
        prior_mw = FillArrays.Zeros(n_rff)  # prior weight mean
        prior_Λw = Diagonal(FillArrays.Ones(n_rff))  # prior weight precision
        blr = BayesianLinearRegressor(prior_mw, prior_Λw)
        f = BasisFunctionRegressor(blr, h)
        fz = f(RowVecs(z[:,:]), σᵤ^2)
        f′ = BayesianLinearRegressors.posterior(fz, u)
        f′x = f′(RowVecs(x[:,:]), σ^2)
        H = h(RowVecs(z[:,:])).X
        H′ = h(RowVecs(x[:,:])).X
        w = f′.blr.mw
        dHdt = eval_dHdt(h, z)
        dfdt_cov_cache = dHdt * (f′.blr.Λw \ dHdt')

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

        K = H * (prior_Λw \ H') + σᵤ^2 * I
        L = cholesky(Hermitian(K)).L

        return new(z, u, x, y, y_mean, y_std, y_standardized, σᵤ, σ, tσ, k,
            n_rff, h, tϕ, blr, f, fz, f′, f′x, H, H′, w, dHdt, dfdt_cov_cache, L, nothing, standardize, centralize)
    end
end

function reconstruct_gp(gp::RFFGP;
    ϕ::Union{Nothing,Vector{Float64}}=nothing,
    u::Union{Nothing,Vector{Float64}}=nothing,
    σ::Union{Nothing,Float64}=nothing
)
    _ϕ = isnothing(ϕ) ? only_params(gp.k) : ϕ
    _u = isnothing(u) ? gp.u : u
    _σ = isnothing(σ) ? gp.σ : σ

    if isnothing(ϕ) && isnothing(u) && !isnothing(σ)
        gp.σ = _σ
        gp.f′x = gp.f′(RowVecs(gp.x[:,:]), gp.σ^2)
    elseif isnothing(ϕ) && !isnothing(u)
        gp.u = _u
        gp.f′ = BayesianLinearRegressors.posterior(gp.fz, _u)
        gp.σ = _σ
        gp.f′x = gp.f′(RowVecs(gp.x[:,:]), gp.σ^2)
        gp.w = gp.f′.blr.mw
    elseif !isnothing(ϕ)
        gp = RFFGP(gp.z, _u, gp.x, gp.y, gp.σᵤ, _σ;
            k=reconstruct_kernel(gp.k, _ϕ), 
            n_rff=gp.n_rff, 
            standardize=gp.standardize, 
            centralize=gp.centralize)
    else
        @error "There is no update in the GP."
    end
    return gp
end

function reconstruct_gp(gp::Vector{RFFGP}; ϕ::AbstractMatrix{<:Real})
    gp_reconstructed = RFFGP[]
    for (gpk, ϕk) in zip(gp, eachcol(ϕ))
        push!(gp_reconstructed, reconstruct_gp(gpk, ϕ=ϕk[:]))
    end
    return gp_reconstructed
end

function set_σ_prior!(gp::RFFGP, pσ::Distribution)
    gp.pσ = pσ
end

function f_conditional(f::BasisFunctionRegressor, z::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    fz = f(RowVecs(z[:,:]), 1e-6)
    f′ = BayesianLinearRegressors.posterior(fz, u)
    return f′
end

function update_u!(gp::RFFGP, u::Vector{Float64})
    @assert length(u) == length(gp.z)
    gp.u[:] = u
    gp.f′ = BayesianLinearRegressors.posterior(gp.fz, u)
end

function update_y!(gp::RFFGP, y::Vector{Float64})
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

weight_mean(gp::RFFGP) = gp.f′.blr.mw  # == Λw⁻¹/σᵤ² * Φᵀx
weight_mean(gp::Vector{RFFGP}) = [weight_mean(gpk) for gpk in gp]
weight_mean_matrix(gp::Vector{RFFGP}) = reduce(vcat, weight_mean(gp)')
weight_precision(gp::RFFGP) = gp.f′.blr.Λw  # == (α²I + ΦᵀΦ/σᵤ²)
noise_cov(gp::RFFGP) = gp.fz.Σy  # == Σx = σᵤ²I
dHdt(gp::RFFGP) = gp.dHdt  # == ∂H/∂t
Hmat(gp::RFFGP, t::AbstractVector{<:Real}) = gp.h(RowVecs(t[:,:])).X  # == H
Hmat(gp::RFFGP) = Hmat(gp, gp.z)

# predict the mean and variance of the GP at the given time points
f_predictive(gp::RFFGP, t_test::AbstractVector{<:Real}, sd::Float64) = gp.f′(RowVecs(t_test[:,:]), sd)

# gradient of the GP function
# numerical_dHdt = t -> ForwardDiff.jacobian(_t -> h(RowVecs(_t)).X, t)
# eval_dHdt(h, t) ≈ numerical_dHdt(t)
function eval_dHdt(h::RFFBasis, t::AbstractVector{<:Real})
    𝓁 = h.inner_weights
    outer_scaled = h.outer_weights  # √(2 * α^2 / n_rff)
    _t = t ./ 𝓁
    return (outer_scaled * h.ω ./ 𝓁 .* -sin.(h.ω .* _t .+ h.τ')) |> Matrix
end

function eval_dhdα(h::RFFBasis, t::AbstractVector{<:Real}, α::AbstractVector{<:Real})
    𝓁 = h.inner_weights
    _t = t ./ 𝓁
    n_rff = length(h.ω)
    return √(2 / n_rff) .* cos.(h.ω .* _t .+ h.τ')
end  # TODO

function eval_dhd𝓁(h::RFFBasis, t::AbstractVector{<:Real}, 𝓁::AbstractVector{<:Real})
    𝓁 = h.inner_weights
    outer_scaled = h.outer_weights  # √(2 * α^2 / n_rff)
    _t = t ./ 𝓁
    return outer_scaled .* (h.ω .* _t) ./ 𝓁^2 .* sin.(h.ω .* _t .+ h.τ')
end  # TODO

mean(gp::RFFGP, w::AbstractVector{<:Real}, t::AbstractVector{<:Real}) = Hmat(gp, t) * w
mean(gp::RFFGP) = mean(gp, gp.w, gp.z)
mean(gp::Vector{RFFGP}, W::AbstractMatrix{<:Real}) = [mean(gpk, wk[:], gpk.z) for (gpk, wk) in zip(gp, eachrow(W))]
mean(gp::Vector{RFFGP}, t::AbstractVector{<:Real}) = [mean(gpk, gpk.w, t) for gpk in gp]
W2X(gp::Vector{RFFGP}, W::AbstractMatrix{<:Real}) = reduce(vcat, mean(gp, W)')
t2X(gp::Vector{RFFGP}, t::AbstractVector{<:Real}) = reduce(vcat, mean(gp, t)')
Wt2X(gp::Vector{RFFGP}, W::AbstractMatrix{<:Real}, t::AbstractVector{<:Real}) = reduce(vcat, mean(gp, W, t)')

function calc_y_mean_and_diagcov(gp::RFFGP, w::AbstractVector{<:Real}, σ::Real)
    y_mean = gp.H′ * w
    y_cov = Diagonal(σ^2 * ones(length(gp.y_standardized)))
    return y_mean, y_cov
end

dfdt_mean(gp::RFFGP, w::AbstractVector{<:Real}) = gp.dHdt * w
dfdt_mean(gp::RFFGP) = dfdt_mean(gp, gp.w)
dfdt_mean(gp::Vector{RFFGP}, W::AbstractMatrix{<:Real}) = [dfdt_mean(gp[k], wk[:]) for (k, wk) in enumerate(eachrow(W))]

dfdt_cov(gp::RFFGP) = gp.dfdt_cov_cache
dfdt_cov(gp::Vector{RFFGP}) = [dfdt_cov(gpk) for gpk in gp]

# logpdf functions
logpdf_w(gp::RFFGP) = sum(logpdf.(Normal(0, 1), gp.f′.blr.mw))  # TODO
logpdf_f(gp::RFFGP) = logpdf(gp.fz, gp.y)  # TODO
logpdf_f′(gp::RFFGP, t::AbstractVector{<:Real}, y::AbstractVector{<:Real}, sd::Float64) = logpdf(gp.f′(RowVecs(t[:,:]), sd), y)  # TODO

# derivatives of the logpdf functions
∇w_logpdf_f(gp::RFFGP) = gradlogpdf(gp.f′(RowVecs(gp.z[:,:]), gp.σ), gp.y)  # TODO
∇w_logpdf_f′(gp::RFFGP, t::AbstractVector{<:Real}, y::AbstractVector{<:Real}, sd::Float64) =
    gradlogpdf(gp.f′(RowVecs(t[:,:]), sd), y)  # TODO

function plot(gp::RFFGP)
    times = gp.z
    diff_t = times[end]-times[1]
    t_test = collect((times[1]-diff_t/20):diff_t/100:(times[end]+diff_t/20))
    Plots.plot(t_test, gp.f′(RowVecs(t_test[:,:]), gp.σ), ribbon_scale=3, label="f(t)", xlabel="t")
    Plots.scatter!(gp.z, gp.u, c=:blue, label="x")
    Plots.scatter!(gp.z, gp.y, c=:red, label="y")
    Plots.title!("RFF-GP regression")
end

function plot_graddist(gp::RFFGP)
    grad_mean = dfdt_mean(gp)
    σ_vec = sqrt.(diag(dfdt_cov(gp)))
    upper = grad_mean .+ 3 * σ_vec
    lower = grad_mean .- 3 * σ_vec
    Plots.scatter(gp.z, grad_mean, c=:blue, label="𝔼[df/dt]", ms=1, xlabel="t")
    Plots.bar!(gp.z, upper, fillrange=lower, fillalpha=0.5, label="±3σ", c=:lightblue)
    Plots.hline!([0], c=:black, label="", ls=:dash)
    Plots.title!("Gradient distribution of RFF-GP")
end

function plot_gradcov(gp::RFFGP; clims=(-1., 1.))
    grad_cov = dfdt_cov(gp)
    mid = sum(clims) / 2
    Plots.heatmap(
        grad_cov, y_flip=true, clims=clims,
        c=cgrad([:blue, :white, :red], [clims[1], 0, clims[2]]))
    Plots.title!("Cov[df/dt, df/dt] in RFF-GP")
end

function compare_gradcovelms(gp::GP, rgp::RFFGP)
    gradcov_gp = dfdt_cov(gp)
    gradcov_rgp = dfdt_cov(rgp)
    elm_gradcov_gp = vcat(eachcol(gradcov_gp)...)
    elm_gradcov_rgp = vcat(eachcol(gradcov_rgp)...)

    scatter(
        elm_gradcov_gp, elm_gradcov_rgp,
        label=false,
        xlabel="Cov[df/dt]_i,j in GP",
        ylabel="Cov[df/dt]_i,j in RFF-GP",
    )
    plot!(-1:0.1:1, -1:0.1:1, label=false, ls=:dash, c=:black)
end
