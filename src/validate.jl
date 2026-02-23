# ∇x ----- 
# TESTED
function validate_∇tx_logpdf_x(gpgm::Union{GPGM,MAGI}; atol=1e-5)
    function ∇tx_logpdf_x_autodiff(gpgm, transformed_X)
        logpdf_fn = tx -> logpdf_x(gpgm, calc_X(gpgm.gp, tx))
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, transformed_X)')
    end

    transformed_X = get_transformed_X(gpgm)
    X = calc_X(gpgm.gp, transformed_X)
    analytic_grad = ∇tx_logpdf_x(gpgm, X)
    autodiff_grad = ∇tx_logpdf_x_autodiff(gpgm, transformed_X)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇tx_logpdf_y(gpgm::Union{GPGM,MAGI}; atol=1e-5)
    function ∇tx_logpdf_y_autodiff(gpgm, Y_std, transformed_X, σ)
        logpdf_fn = tx -> logpdf_y(gpgm, Y_std, calc_X(gpgm.gp, tx), σ)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, transformed_X)')
    end

    Y_std = get_standardized_Y(gpgm)
    transformed_X = get_transformed_X(gpgm)
    X = calc_X(gpgm.gp, transformed_X)
    transformed_σ = get_transformed_σ(gpgm)
    σ = [calc_var(gpk.tσ, transformed_σk) for (gpk, transformed_σk) in zip(gpgm.gp, transformed_σ)];
    analytic_grad = ∇tx_logpdf_y(gpgm, Y_std, X, σ)
    autodiff_grad = ∇tx_logpdf_y_autodiff(gpgm, Y_std, transformed_X, σ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=1e-5)
    return true
end

# TESTED
function validate_∇tx_ulogpdf_e(gpgm::GPGM; atol=1e-5)
    function ∇tx_ulogpdf_e_autodiff(gpgm, transformed_X, θ, γ)
        logpdf_fn = tx -> ulogpdf_e(gpgm, calc_X(gpgm.gp, tx), θ, γ)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, transformed_X)')
    end

    transformed_X = get_transformed_X(gpgm)
    X = calc_X(gpgm.gp, transformed_X)
    θ = get_θ(gpgm)
    γ = get_γ(gpgm)
    analytic_grad = ∇tx_ulogpdf_e(gpgm, X, θ, γ)
    autodiff_grad = ∇tx_ulogpdf_e_autodiff(gpgm, transformed_X, θ, γ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=1e-5)
    return true
end

function validate_∇tx_ulogpdf_e(magi::MAGI; atol=1e-5)
    function ∇tx_ulogpdf_e_autodiff(magi, transformed_X, θ, γ_jitter)
        logpdf_fn = tx -> ulogpdf_e(magi.odegrad, magi.gp, calc_X(magi.gp, tx), θ, γ_jitter)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, transformed_X)')
    end

    transformed_X = get_transformed_X(magi)
    X = calc_X(magi.gp, transformed_X)
    θ = get_θ(magi)
    analytic_grad = ∇tx_ulogpdf_e(magi.odegrad, magi.gp, X, θ, magi.γ_jitter)
    autodiff_grad = ∇tx_ulogpdf_e_autodiff(magi, transformed_X, θ, magi.γ_jitter)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# ∇w ----- 
# TESTED
function validate_∇w_logpdf_x(rffgm::RFFGM; atol=1e-5)
    function ∇w_logpdf_x_autodiff(rffgm, W)
        logpdf_fn = w -> logpdf_x(rffgm, w)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, W)')
    end
    
    X = get_X(rffgm)
    W = get_W(rffgm)
    analytic_grad = ∇w_logpdf_x(rffgm, X)
    autodiff_grad = ∇w_logpdf_x_autodiff(rffgm, W)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇w_logpdf_y(rffgm::RFFGM; atol=1e-5)
    function ∇w_logpdf_y_autodiff(rffgm, Y_std, W, σ)
        logpdf_fn = w -> logpdf_y(rffgm, Y_std, w, σ)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, W)')
    end

    Y_std = get_standardized_Y(rffgm)
    X = get_X(rffgm)
    W = get_W(rffgm)
    σ = get_σ(rffgm)
    analytic_grad = ∇w_logpdf_y(rffgm, Y_std, W, σ)
    autodiff_grad = ∇w_logpdf_y_autodiff(rffgm, Y_std, W, σ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=1e-5)
    return true
end

# TESTED
function validate_∇w_ulogpdf_e(rffgm::RFFGM; atol=1e-5)
    function ∇w_ulogpdf_e_autodiff(rffgm, W, θ, γ)
        logpdf_fn = w -> ulogpdf_e(rffgm, w, θ, γ)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, W)')
    end

    X = get_X(rffgm)
    W = get_W(rffgm)
    θ = get_θ(rffgm)
    γ = get_γ(rffgm)
    analytic_grad = ∇w_ulogpdf_e(rffgm, X, W, θ, γ)
    autodiff_grad = ∇w_ulogpdf_e_autodiff(rffgm, W, θ, γ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=1e-5)
    return true
end

# ∇θ -----
# TESTED
function validate_∇tθ_logpdf_θ(gm::AbstractGM; atol=1e-5)
    function ∇tθ_logpdf_θ_autodiff(gm, transformed_θ)
        function logpdf_fn(transformed_θ)
            θ = calc_θ(gm.odegrad, transformed_θ)
            return logpdf_θ(gm, θ)
        end
        return ForwardDiff.gradient(logpdf_fn, transformed_θ)
    end

    θ = get_θ(gm)
    transformed_θ = get_transformed_θ(gm)
    analytic_grad = ∇tθ_logpdf_θ(gm, θ, transformed_θ)
    autodiff_grad = ∇tθ_logpdf_θ_autodiff(gm, transformed_θ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇tθ_ulogpdf_e(gpgm::GPGM; atol=1e-5)
    function ∇tθ_ulogpdf_e_autodiff(gpgm, X, transformed_θ, γ)
        logpdf_fn = transformed_θ -> ulogpdf_e(gpgm, X, calc_θ(gpgm.odegrad, transformed_θ), γ)
        return ForwardDiff.gradient(logpdf_fn, transformed_θ)
    end

    X = get_X(gpgm)
    transformed_θ = get_transformed_θ(gpgm)
    transformed_γ = get_transformed_γ(gpgm)
    θ = calc_θ(gpgm.odegrad, transformed_θ)
    γ = calc_γ(gpgm.odegrad, transformed_γ)
    analytic_grad = ∇tθ_ulogpdf_e(gpgm, X, θ, γ)
    autodiff_grad = ∇tθ_ulogpdf_e_autodiff(gpgm, X, transformed_θ, γ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

function validate_∇tθ_ulogpdf_e(magi::MAGI; atol=1e-5)
    function ∇tθ_ulogpdf_e_autodiff(magi, X, transformed_θ, γ_jitter)
        logpdf_fn = tθ -> ulogpdf_e(magi.odegrad, magi.gp, X, calc_θ(magi.odegrad, tθ), γ_jitter)
        return ForwardDiff.gradient(logpdf_fn, transformed_θ)
    end

    X = get_X(magi)
    transformed_θ = get_transformed_θ(magi)
    θ = calc_θ(magi.odegrad, transformed_θ)
    analytic_grad = ∇tθ_ulogpdf_e(magi.odegrad, magi.gp, X, θ, magi.γ_jitter)
    autodiff_grad = ∇tθ_ulogpdf_e_autodiff(magi, X, transformed_θ, magi.γ_jitter)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇tθ_ulogpdf_e(rffgm::RFFGM; atol=1e-5)
    function ∇tθ_ulogpdf_e_autodiff(rffgm, W, transformed_θ, γ)
        logpdf_fn = transformed_θ -> ulogpdf_e(rffgm, W, calc_θ(rffgm.odegrad, transformed_θ), γ)
        return ForwardDiff.gradient(logpdf_fn, transformed_θ)
    end

    X = get_X(rffgm)
    W = get_W(rffgm)
    transformed_θ = get_transformed_θ(rffgm)
    transformed_γ = get_transformed_γ(rffgm)
    θ = calc_θ(rffgm.odegrad, transformed_θ)
    γ = calc_γ(rffgm.odegrad, transformed_γ)
    analytic_grad = ∇tθ_ulogpdf_e(rffgm, X, W, θ, γ)
    autodiff_grad = ∇tθ_ulogpdf_e_autodiff(rffgm, W, transformed_θ, γ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# ∇γ -----
# TESTED
function validate_∇tγ_logpdf_γ(gm::AbstractGM; atol=1e-5)
    function ∇tγ_logpdf_γ_autodiff(gm, transformed_γ)
        function logpdf_fn(transformed_γ)
            γ = calc_γ(gm.odegrad, transformed_γ)
            return logpdf_γ(gm, γ)
        end
        return [ForwardDiff.derivative(logpdf_fn, transformed_γ)]
    end

    γ = get_γ(gm)
    transformed_γ = get_transformed_γ(gm)
    analytic_grad = ∇tγ_logpdf_γ(gm, γ, transformed_γ)
    autodiff_grad = ∇tγ_logpdf_γ_autodiff(gm, transformed_γ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇tγ_ulogpdf_e(gpgm::GPGM; atol=1e-5)
    function ∇tγ_ulogpdf_e_autodiff(gpgm, X, θ, transformed_γ)
        logpdf_fn = transformed_γ -> ulogpdf_e(gpgm, X, θ, calc_γ(gpgm.odegrad, transformed_γ))
        return [ForwardDiff.derivative(logpdf_fn, transformed_γ)]
    end

    X = get_X(gpgm)
    transformed_θ = get_transformed_θ(gpgm)
    transformed_γ = get_transformed_γ(gpgm)
    θ = calc_θ(gpgm.odegrad, transformed_θ)
    γ = calc_γ(gpgm.odegrad, transformed_γ)
    analytic_grad = ∇tγ_ulogpdf_e(gpgm, X, θ, γ)
    autodiff_grad = ∇tγ_ulogpdf_e_autodiff(gpgm, X, θ, transformed_γ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇tγ_ulogpdf_e(rffgm::RFFGM; atol=1e-5)
    function ∇tγ_ulogpdf_e_autodiff(rffgm, W, θ, transformed_γ)
        logpdf_fn = transformed_γ -> ulogpdf_e(rffgm, W, θ, calc_γ(rffgm.odegrad, transformed_γ))
        return [ForwardDiff.derivative(logpdf_fn, transformed_γ)]
    end

    X = get_X(rffgm)
    W = get_W(rffgm)
    transformed_θ = get_transformed_θ(rffgm)
    transformed_γ = get_transformed_γ(rffgm)
    θ = calc_θ(rffgm.odegrad, transformed_θ)
    γ = calc_γ(rffgm.odegrad, transformed_γ)
    analytic_grad = ∇tγ_ulogpdf_e(rffgm, X, W, θ, γ)
    autodiff_grad = ∇tγ_ulogpdf_e_autodiff(rffgm, W, θ, transformed_γ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# ∇σ -----
# TESTED
function validate_∇tσ_logpdf_σ(gm::AbstractGM; atol=1e-5)
    function ∇tσ_logpdf_σ_autodiff(gm, transformed_σ)
        function logpdf_fn(transformed_σ)
            σ = calc_σ(gm.gp, transformed_σ)
            return logpdf_σ(gm, σ)
        end
        return ForwardDiff.gradient(logpdf_fn, transformed_σ)
    end

    σ = get_σ(gm)
    transformed_σ = get_transformed_σ(gm)
    analytic_grad = ∇tσ_logpdf_σ(gm, σ, transformed_σ)
    autodiff_grad = ∇tσ_logpdf_σ_autodiff(gm, transformed_σ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇tσ_logpdf_y(gpgm::Union{GPGM,MAGI}; atol=1e-5)
    function ∇tσ_logpdf_y_autodiff(gpgm, Y_std, X, transformed_σ)
        logpdf_fn = transformed_σ -> logpdf_y(gpgm, Y_std, X, calc_σ(gpgm.gp, transformed_σ))
        return ForwardDiff.gradient(logpdf_fn, transformed_σ)
    end

    Y_std = get_standardized_Y(gpgm)
    X = get_X(gpgm)
    transformed_σ = get_transformed_σ(gpgm)
    σ = calc_σ(gpgm.gp, transformed_σ)
    analytic_grad = ∇tσ_logpdf_y(gpgm, Y_std, X, σ)
    autodiff_grad = ∇tσ_logpdf_y_autodiff(gpgm, Y_std, X, transformed_σ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇tσ_logpdf_y(rffgm::RFFGM; atol=1e-5)
    function ∇tσ_logpdf_y_autodiff(rffgm, Y_std, W, transformed_σ)
        logpdf_fn = tσ ->  logpdf_y(rffgm, Y_std, W, calc_σ(rffgm.gp, tσ))
        return ForwardDiff.gradient(logpdf_fn, transformed_σ)
    end

    Y_std = get_standardized_Y(rffgm)
    W = get_W(rffgm)
    transformed_σ = get_transformed_σ(rffgm)
    σ = calc_σ(rffgm.gp, transformed_σ)
    analytic_grad = ∇tσ_logpdf_y(rffgm, Y_std, W, σ)
    autodiff_grad = ∇tσ_logpdf_y_autodiff(rffgm, Y_std, W, transformed_σ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇y_logpdf_y(gm::Union{GPGM,MAGI}; atol=1e-5)
    function ∇y_logpdf_y_autodiff(gm, Y, X, σ)
        logpdf_fn = Y -> logpdf_y(gm, calc_standardized_Y(gm.gp, Y), X, σ)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, Y)')
    end

    Y = get_Y(gm)
    Y_std = calc_standardized_Y(gm.gp, Y)
    X = get_X(gm)
    σ = get_σ(gm)
    analytic_grad = ∇y_logpdf_y(gm, Y_std, X, σ)
    autodiff_grad = ∇y_logpdf_y_autodiff(gm, Y, X, σ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# TESTED
function validate_∇y_logpdf_y(gm::RFFGM; atol=1e-5)
    function ∇y_logpdf_y_autodiff(gm, Y, W, σ)
        logpdf_fn = y -> logpdf_y(gm, calc_standardized_Y(gm.gp, y), W, σ)
        return reduce(vcat, ForwardDiff.gradient(logpdf_fn, Y)')
    end

    Y = get_Y(gm)
    Y_std = calc_standardized_Y(gm.gp, Y)
    W = get_W(gm)
    σ = get_σ(gm)
    analytic_grad = ∇y_logpdf_y(gm, Y_std, W, σ)
    autodiff_grad = ∇y_logpdf_y_autodiff(gm, Y, W, σ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# ∇ϕ -----
function validate_∇tϕ_logpdf_ϕ(gpgm::Union{GPGM,MAGI}; atol=1e-5)
    function ∇tϕ_logpdf_ϕ_autodiff(gpgm, transformed_ϕ)
        function logpdf_fn(tϕ)
            ϕ = calc_ϕ(gpgm.gp, reshape(tϕ, size(transformed_ϕ)))
            return logpdf_ϕ(gpgm, ϕ)
        end
        return ForwardDiff.gradient(logpdf_fn, vec(transformed_ϕ))
    end

    ϕ = get_ϕ(gpgm)
    transformed_ϕ = get_transformed_ϕ(gpgm)
    analytic_grad = ∇tϕ_logpdf_ϕ(gpgm, ϕ)
    autodiff_grad = ∇tϕ_logpdf_ϕ_autodiff(gpgm, transformed_ϕ)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end

# Central finite difference gradient (used when reconstruct_gp is not ForwardDiff-compatible)
function _finite_diff_gradient(f::Function, x::Vector{Float64}; ε=1e-5)
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x_plus = copy(x); x_plus[i] += ε
        x_minus = copy(x); x_minus[i] -= ε
        grad[i] = (f(x_plus) - f(x_minus)) / (2ε)
    end
    return grad
end

function validate_∇tϕ_logpdf_x(gpgm::Union{GPGM,MAGI}; atol=1e-4)
    ϕ = get_ϕ(gpgm)
    X = get_X(gpgm)
    transformed_ϕ = get_transformed_ϕ(gpgm)

    function logpdf_fn(tϕ)
        ϕ_mat = calc_ϕ(gpgm.gp, reshape(tϕ, size(transformed_ϕ)))
        gp_new = reconstruct_gp(gpgm.gp; ϕ=ϕ_mat)
        return logpdf_x(gp_new, X)
    end

    analytic_grad = ∇tϕ_logpdf_x(gpgm, X, ϕ)
    fd_grad = _finite_diff_gradient(logpdf_fn, vec(Float64.(transformed_ϕ)))
    @assert isapprox(analytic_grad, fd_grad, atol=atol) "∇tϕ_logpdf_x mismatch:\n  analytic=$analytic_grad\n  finite_diff=$fd_grad"
    return true
end

function validate_∇tϕ_logpdf_y(gpgm::Union{GPGM,MAGI}; atol=1e-4)
    Y_std = get_standardized_Y(gpgm)
    X = get_X(gpgm)
    σ = get_σ(gpgm)
    ϕ = get_ϕ(gpgm)
    transformed_ϕ = get_transformed_ϕ(gpgm)

    function logpdf_fn(tϕ)
        ϕ_mat = calc_ϕ(gpgm.gp, reshape(tϕ, size(transformed_ϕ)))
        gp_new = reconstruct_gp(gpgm.gp; ϕ=ϕ_mat)
        return logpdf_y(gp_new, Y_std, X, σ)
    end

    analytic_grad = ∇tϕ_logpdf_y(gpgm, Y_std, X, σ, ϕ)
    fd_grad = _finite_diff_gradient(logpdf_fn, vec(Float64.(transformed_ϕ)))
    @assert isapprox(analytic_grad, fd_grad, atol=atol) "∇tϕ_logpdf_y mismatch:\n  analytic=$analytic_grad\n  finite_diff=$fd_grad"
    return true
end

function validate_∇tϕ_ulogpdf_e(gpgm::GPGM; atol=1e-4)
    X = get_X(gpgm)
    θ = get_θ(gpgm)
    γ = get_γ(gpgm)
    ϕ = get_ϕ(gpgm)
    transformed_ϕ = get_transformed_ϕ(gpgm)

    function logpdf_fn(tϕ)
        ϕ_mat = calc_ϕ(gpgm.gp, reshape(tϕ, size(transformed_ϕ)))
        return ulogpdf_e(gpgm, X, θ, γ, ϕ_mat)
    end

    analytic_grad = ∇tϕ_ulogpdf_e(gpgm, X, θ, γ, ϕ)
    fd_grad = _finite_diff_gradient(logpdf_fn, vec(Float64.(transformed_ϕ)))
    @assert isapprox(analytic_grad, fd_grad, atol=atol) "∇tϕ_ulogpdf_e mismatch:\n  analytic=$analytic_grad\n  finite_diff=$fd_grad"
    return true
end

function validate_∇tϕ_ulogpdf_e(magi::MAGI; atol=1e-4)
    X = get_X(magi)
    θ = get_θ(magi)
    ϕ = get_ϕ(magi)
    transformed_ϕ = get_transformed_ϕ(magi)

    function logpdf_fn(tϕ)
        ϕ_mat = calc_ϕ(magi.gp, reshape(tϕ, size(transformed_ϕ)))
        return ulogpdf_e(magi.odegrad, magi.gp, X, θ, magi.γ_jitter, ϕ_mat)
    end

    analytic_grad = ∇tϕ_ulogpdf_e(magi.odegrad, magi.gp, X, θ, magi.γ_jitter, ϕ)
    fd_grad = _finite_diff_gradient(logpdf_fn, vec(Float64.(transformed_ϕ)))
    @assert isapprox(analytic_grad, fd_grad, atol=atol) "∇tϕ_ulogpdf_e mismatch:\n  analytic=$analytic_grad\n  finite_diff=$fd_grad"
    return true
end

# ∇ -----
# TESTED
function validate_∇ulogpdf(gm::AbstractGM, sample_target::Vector{Symbol}; atol=1e-5)
    function ∇ulogpdf_autodiff(params, gm, sample_target)
        function logpdf_fn(params)
            return ulogpdf(params, gm, sample_target)
        end
        return ForwardDiff.gradient(logpdf_fn, params)
    end

    params = pack_param_vec(gm, sample_target)
    analytic_grad = ∇ulogpdf(params, gm, sample_target)
    autodiff_grad = ∇ulogpdf_autodiff(params, gm, sample_target)
    @assert isapprox(analytic_grad, autodiff_grad, atol=atol)
    return true
end
