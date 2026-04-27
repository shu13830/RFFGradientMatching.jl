struct ODEGradFuns
    prob::ODEProblem
    probname::String
    ẋ::Function     # ODE
    dẋdx::Function  # state derivative
    dẋdθ::Function  # parameter derivative

    function ODEGradFuns(prob::ODEProblem, probname::String)
        ẋ = (u, p) -> begin
            du = Vector{Union{<:Real, <:ForwardDiff.Dual}}(undef, length(u))
            prob.f(du, u, p, 0.0)
            du
        end
        # Use hand-coded in-place Jacobians if available, otherwise ForwardDiff wrapper
        manual_dẋdx!, manual_dẋdθ! = get_manual_jacobians(probname)
        if manual_dẋdx! !== nothing
            dẋdx = manual_dẋdx!
            dẋdθ = manual_dẋdθ!
        else
            dẋdx = (J, x, θ) -> (J .= ForwardDiff.jacobian(_x -> ẋ(_x, θ), collect(x)); J)
            dẋdθ = (J, x, θ) -> (J .= ForwardDiff.jacobian(_θ -> ẋ(collect(x), _θ), θ); J)
        end
        return new(prob, probname, ẋ, dẋdx, dẋdθ)
    end
end

mutable struct ODEGrad
    Y::Matrix{Float64}               # observation data
    X::Matrix{Float64}               # states at discrete time points
    θ::Vector{Float64}               # parameters of ODE
    tθ::Vector{PriorTransformation}  # prior transformation for θ
    functions::ODEGradFuns           # ODE relevant functions for gradient matching
    γ::Float64                       # noise std for gradient matching
    tγ::PriorTransformation          # prior transformation for γ
    bandsize::Union{Nothing, Int}    # band matrix bandwidth (nothing = dense)

    function ODEGrad(
        obs::Matrix{Float64},
        prob::ODEProblem,
        probname::String
    )
        X = copy(obs)  # initialize states with observation value
        tθ = fill(PriorTransformation(Normal(0, 1), identity), length(prob.p))
        # θ = prob.p  # initialize parameters with true value
        θ = [rand_var(tθi) for tθi in tθ]
        functions = ODEGradFuns(prob, probname)
        γ = 0.3
        tγ = PriorTransformation(Normal(0, 1), log)
        new(obs, X, θ, tθ, functions, γ, tγ, nothing)
    end
end

n_times(og::ODEGrad) = size(og.X, 2)
n_state_types(og::ODEGrad) = size(og.X, 1)

function eval_ẋ(og::ODEGrad, X::AbstractMatrix{T}, θ::AbstractVector{T2}) where {T<:Real, T2<:Real}
    K, N = size(X)
    ET = promote_type(T, T2)
    if ET === Float64
        # Fast path: in-place ODE evaluation (no ForwardDiff)
        result = Matrix{Float64}(undef, K, N)
        du = Vector{Float64}(undef, K)
        for i in 1:N
            og.functions.prob.f(du, view(X, :, i), θ, 0.0)
            result[:, i] .= du
        end
        return result
    else
        # ForwardDiff path: use ẋ closure (supports Dual numbers)
        return reduce(hcat, [og.functions.ẋ(X[:, i], θ) for i in 1:N])
    end
end
eval_ẋ(og::ODEGrad) = eval_ẋ(og, og.X, og.θ)

function eval_dẋdx(og::ODEGrad, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, y_std::AbstractVector{<:Real})
    K, N = size(X)
    result = Array{Float64}(undef, K, K, N)
    inv_ystd = 1.0 ./ y_std
    J_buf = Matrix{Float64}(undef, K, K)
    for i in 1:N
        og.functions.dẋdx(J_buf, view(X, :, i), θ)
        @inbounds for c in 1:K, r in 1:K
            result[r, c, i] = J_buf[r, c] * inv_ystd[r] * y_std[c]
        end
    end
    return result
end
eval_dẋdx(og::ODEGrad) = eval_dẋdx(og, og.X, og.θ, vec(StatsBase.std(og.Y, dims=2)))

function eval_dẋdθ(og::ODEGrad, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, y_std::AbstractVector{<:Real})
    K, N = size(X)
    P = length(θ)
    result = Array{Float64}(undef, K, P, N)
    inv_ystd = 1.0 ./ y_std
    J_buf = Matrix{Float64}(undef, K, P)
    for i in 1:N
        og.functions.dẋdθ(J_buf, view(X, :, i), θ)
        @inbounds for c in 1:P, r in 1:K
            result[r, c, i] = J_buf[r, c] * inv_ystd[r]
        end
    end
    return result
end
eval_dẋdθ(og::ODEGrad) = eval_dẋdθ(og, og.X, og.θ, vec(StatsBase.std(og.Y, dims=2)))
