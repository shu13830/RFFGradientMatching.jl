default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function optimize_u!(gm::Union{RFFGM,GPGM}; maxiter::Int=1000, optimizer=default_optimizer)
    
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

        # optimize u
        u_init = copy(gp.u)
        result = Optim.optimize(loss, u_init, optimizer, options, autodiff=:forward)
        u_opt = Optim.minimizer(result)

        # update u
        gm.gp[k] = reconstruct_gp(gp; u=u_opt)
        @info "Optimized"
    end
end

function optimize_ϕ_and_σ!(gm::Union{RFFGM,GPGM}; maxiter::Int=1000, optimizer=default_optimizer)
    
    options = Optim.Options(; iterations=maxiter, show_trace=false)

    gps = gm.gp
    for (k, gp) in enumerate(gps)
        @info "Optimizing hyperparameters of GP $k"
        # construct loss function to minimize
        loss = (params) -> begin
            _σ = exp(params[end])
            _ϕ = exp.(params[1:end-1])
            _k = reconstruct_kernel(gp.k, _ϕ)
            _gp = AbstractGPs.GP(_k)
            _gpx = _gp(gp.x, _σ^2)
            return -logpdf(_gpx, gp.y_standardized)
        end

        # optimize ϕ and σ
        base_k, inner, outer = params(gp.k)
        param_init = log.([inner..., outer, gp.σ])
        result = Optim.optimize(loss, param_init, optimizer, options, autodiff=:forward)
        param_opt = Optim.minimizer(result)

        # update hyperparameters
        opt_σ = exp(param_opt[end])
        opt_ϕ = exp.(param_opt[1:end-1])
        gps[k] = reconstruct_gp(gp; ϕ=opt_ϕ, σ=opt_σ)
        @info "Optimized GP $k: ϕ=$opt_ϕ, σ=$opt_σ"
    end
end
