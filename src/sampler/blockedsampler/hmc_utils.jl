"""
Per-dimension step size HMC adaptation, following R MAGI (Sampler.cpp).

Key design: Identity metric + per-dimension step_low vector (R MAGI style).
This replaces the previous DiagEuclideanMetric + scalar step_low approach.

During burn-in (after iteration 10):
1. Rolling acceptance rate (last 100 iterations):
   - accept_rate > 0.9 → step_low .*= 1.005  (+0.5%)
   - accept_rate < 0.6 → step_low .*= 0.995  (-0.5%)
2. Per-dimension step size adaptation (every adapt_interval iterations):
   blended = α * (xthsd/mean(xthsd) * mean(step_low)) + (1−α) * step_low
   with α=0.05 (conservative, R MAGI default)
3. ε randomized each iteration per-dimension: ε_i ~ Uniform(step_low_i, 2*step_low_i)

After burn-in: all adaptation frozen. ε still randomized for ergodicity.
"""

const ADAPT_BLEND_ALPHA = 0.05  # R MAGI default (conservative)

"""Custom leapfrog integration with per-dimension step sizes."""
function leapfrog_perdim(q::AbstractVector{<:Real}, ε::AbstractVector{<:Real},
                         n_leapfrog::Int, ∂ℓπ∂θ::Function)
    lp, grad = ∂ℓπ∂θ(q)
    if !isfinite(lp) || any(!isfinite, grad)
        return q, lp, false
    end

    # Initialize momentum: p ~ N(0, I)
    p = randn(length(q))
    H_init = -lp + 0.5 * dot(p, p)

    q_new = copy(q)
    p_new = copy(p)

    # Leapfrog integration with per-dimension ε
    # Half step momentum
    p_new .+= 0.5 .* ε .* grad

    for step in 1:n_leapfrog
        # Full step position
        q_new .+= ε .* p_new

        # Compute gradient at new position
        lp_new, grad_new = ∂ℓπ∂θ(q_new)
        if !isfinite(lp_new) || any(!isfinite, grad_new)
            return q, lp, false  # reject on NaN/Inf
        end

        if step < n_leapfrog
            # Full step momentum (except last)
            p_new .+= ε .* grad_new
        else
            # Half step momentum (last)
            p_new .+= 0.5 .* ε .* grad_new
        end
        lp = lp_new
        grad = grad_new
    end

    # Metropolis acceptance
    H_final = -lp + 0.5 * dot(p_new, p_new)
    log_accept = H_init - H_final

    if log(rand()) < log_accept
        return q_new, lp, true
    else
        lp_orig, _ = ∂ℓπ∂θ(q)
        return q, lp_orig, false
    end
end

"""Adapt per-dimension step sizes and run one HMC step."""
function adjust_and_step!(burnin::Bool, blk::HMCBlock, param_vec::AbstractVector{<:Real})
    if burnin && length(blk.accept_history) > 10
        # --- 1. Rolling acceptance rate → gentle step_low adjustment ---
        n_recent = min(length(blk.accept_history), 100)
        recent = @view blk.accept_history[end-n_recent+1:end]
        accept_rate = sum(recent) / n_recent
        if accept_rate > 0.9
            blk.step_low .*= 1.005
        elseif accept_rate < 0.6
            blk.step_low .*= 0.995
        end

        # --- 2. Per-dimension step size adaptation ---
        blk.iter_counter += 1
        if blk.iter_counter >= blk.adapt_interval && length(blk.sample_history) >= 10
            xthsd = _empirical_std(blk.sample_history)
            mean_std = StatsBase.mean(xthsd)
            if mean_std > 1e-12 && all(xthsd .> 1e-12)
                mean_step = StatsBase.mean(blk.step_low)
                new_step = xthsd ./ mean_std .* mean_step
                blk.step_low .= ADAPT_BLEND_ALPHA .* new_step .+ (1 - ADAPT_BLEND_ALPHA) .* blk.step_low
            end
            blk.iter_counter = 0
            empty!(blk.sample_history)
        end
    end

    # Randomize ε per-dimension: ε_i ~ Uniform(step_low_i, 2*step_low_i)
    ε = blk.step_low .* (1.0 .+ rand(length(blk.step_low)))

    # Extract n_leapfrog from the existing trajectory
    n_leapfrog = blk.sampler.κ.τ.termination_criterion.L

    # Run custom leapfrog
    new_param_vec, lp, accepted = leapfrog_perdim(param_vec, ε, n_leapfrog, blk.h.∂ℓπ∂θ)

    return new_param_vec, accepted
end

# Keep old interface for backward compatibility (redirects to new)
function adjust_ϵ_heuristically!(burnin::Bool, blk::HMCBlock)
    # No-op now — adaptation happens inside adjust_and_step!
    # This function is kept for any code that still calls it
end

"""Record acceptance result into the HMCBlock's accept_history (capped at 100)."""
function record_accept!(blk::HMCBlock, accepted::Bool)
    push!(blk.accept_history, accepted)
    if length(blk.accept_history) > 100
        popfirst!(blk.accept_history)
    end
end

"""Record a sample into the HMCBlock's history buffer (for adaptation)."""
function record_sample!(blk::HMCBlock, param_vec::AbstractVector{<:Real})
    push!(blk.sample_history, Vector{Float64}(param_vec))
end

"""Compute per-dimension empirical standard deviation from sample history."""
function _empirical_std(samples::Vector{Vector{Float64}})
    n = length(samples)
    D = length(samples[1])
    μ = zeros(D)
    for s in samples
        μ .+= s
    end
    μ ./= n
    σ² = zeros(D)
    for s in samples
        σ² .+= (s .- μ) .^ 2
    end
    σ² ./= max(n - 1, 1)
    return sqrt.(σ²)
end
