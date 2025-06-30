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
        dẋdx = (x, θ) -> ForwardDiff.jacobian(_x -> ẋ(_x, θ), x)
        dẋdθ = (x, θ) -> ForwardDiff.jacobian(_θ -> ẋ(x, _θ), θ)
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
        new(obs, X, θ, tθ, functions, γ, tγ)
    end
end

n_times(og::ODEGrad) = size(og.X, 2)
n_state_types(og::ODEGrad) = size(og.X, 1)

eval_ẋ(og::ODEGrad, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}) =
    reduce(hcat, [og.functions.ẋ(x_i, θ) for x_i in eachcol(X)])  # K x N
eval_ẋ(og::ODEGrad) = eval_ẋ(og, og.X, og.θ)

eval_dẋdx(og::ODEGrad, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, y_std::AbstractVector{<:Real}) = 
    cat([og.functions.dẋdx(x_i, θ) ./ y_std .* y_std' for x_i in eachcol(X)]..., dims=3)  # K x K x N
eval_dẋdx(og::ODEGrad) = eval_dẋdx(og, og.X, og.θ, vec(StatsBase.std(og.Y, dims=2)))

eval_dẋdθ(og::ODEGrad, X::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real}, y_std::AbstractVector{<:Real}) =
    cat([og.functions.dẋdθ(x_i, θ) ./ y_std for x_i in eachcol(X)]..., dims=3)  # K x n(θ) x N
eval_dẋdθ(og::ODEGrad) = eval_dẋdθ(og, og.X, og.θ, vec(StatsBase.std(og.Y, dims=2)))
