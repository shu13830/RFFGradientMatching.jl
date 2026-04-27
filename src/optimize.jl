default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function optimize_u!(gm::AbstractGM; maxiter::Int=1000, optimizer=default_optimizer)

    options = Optim.Options(; iterations=maxiter, show_trace=false)

    for (k, gp) in enumerate(gm.gp)
        @info "Optimizing states of GP $k"
        # construct loss function to minimize
        loss = (u) -> begin
            _f = AbstractGPs.GP(gp.k)
            _fz = _f(gp.z, gp.σᵤ^2)
            _f′ = AbstractGPs.posterior(_fz, u)
            _f′x = _f′(gp.x, gp.σ^2)
            return - logpdf(_f′x, gp.y_standardized) - logpdf(_fz, u)
        end

        # optimize u (with fallback on failure)
        u_init = copy(gp.u)
        try
            result = Optim.optimize(loss, u_init, optimizer, options, autodiff=:forward)
            u_opt = Optim.minimizer(result)
            gm.gp[k] = reconstruct_gp(gp; u=u_opt)
            @info "Optimized"
        catch e
            @warn "optimize_u! failed for GP $k ($e). Using initial values (ϕ/σ already optimized)."
        end
    end
end

# Kernel-aware parameter ↔ optimization space transforms
# Default: all params positive → log space
_ϕσ_to_opt(::KernelFunctions.Kernel, inner, outer, σ) = log.([inner..., outer, σ])
_opt_to_ϕσ(::KernelFunctions.Kernel, p) = (exp.(p[1:end-1]), exp(p[end]))

# SigmoidKernel: b > 0 → log, a ∈ ℝ → direct, outer > 0 → log, σ > 0 → log
function _ϕσ_to_opt(::SigmoidKernel, inner, outer, σ)
    b, a = inner
    return [log(b), a, log(outer), log(σ)]
end
function _opt_to_ϕσ(::SigmoidKernel, p)
    ϕ = [exp(p[1]), p[2], exp(p[3])]  # b=exp, a=direct, outer=exp
    σ = exp(p[end])
    return (ϕ, σ)
end

# Bounds for SigmoidKernel optimization: log(b) ∈ [-3, 3], a ∈ [-3, 3], log(outer) ∈ [-3, 3], log(σ) ∈ [-10, 3]
_opt_bounds(::KernelFunctions.Kernel, n) = (nothing, nothing)
function _opt_bounds(::SigmoidKernel, n)
    lower = [-3.0, -3.0, -3.0, -10.0]
    upper = [ 3.0,  3.0,  3.0,   3.0]
    return (lower, upper)
end

function optimize_ϕ_and_σ!(gm::AbstractGM; maxiter::Int=1000, optimizer=default_optimizer)

    options = Optim.Options(; iterations=maxiter, show_trace=false)

    gps = gm.gp
    for (k, gp) in enumerate(gps)
        @info "Optimizing hyperparameters of GP $k"
        base_k, inner, outer = params(gp.k)

        # construct loss function to minimize (kernel-aware transform)
        loss = (params) -> begin
            _ϕ, _σ = _opt_to_ϕσ(base_k, params)
            _k = reconstruct_kernel(gp.k, _ϕ)
            _gp = AbstractGPs.GP(_k)
            _gpx = _gp(gp.x, _σ^2)
            try
                return -logpdf(_gpx, gp.y_standardized)
            catch
                return eltype(params)(1e10)
            end
        end

        # optimize ϕ and σ
        param_init = _ϕσ_to_opt(base_k, inner, outer, gp.σ)
        lb, ub = _opt_bounds(base_k, length(param_init))

        if isnothing(lb)
            # Unconstrained optimization (SqExp, Matern52, etc.)
            result = Optim.optimize(loss, param_init, optimizer, options, autodiff=:forward)
        else
            # Box-constrained optimization (SigmoidKernel)
            param_init = clamp.(param_init, lb, ub)
            result = Optim.optimize(loss, lb, ub, param_init, Fminbox(optimizer), options, autodiff=:forward)
        end
        param_opt = Optim.minimizer(result)

        # update hyperparameters (with fallback on Cholesky failure)
        opt_ϕ, opt_σ = _opt_to_ϕσ(base_k, param_opt)
        try
            gps[k] = reconstruct_gp(gp; ϕ=opt_ϕ, σ=opt_σ)
            @info "Optimized GP $k: ϕ=$opt_ϕ, σ=$opt_σ"
        catch e
            @warn "GP $k: optimization produced invalid kernel ($(typeof(e))). Keeping initial hyperparameters."
        end
    end
end
