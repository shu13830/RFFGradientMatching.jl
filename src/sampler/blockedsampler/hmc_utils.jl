"""
Unified HMC adaptation for all HMCBlock variants (HMC, NUTS, HMCDA),
following R MAGI (Sampler.cpp).

During burn-in (after iteration 10):
1. Rolling acceptance rate (last 100 iterations):
   - accept_rate > 0.9 → step_low *= 1.005  (+0.5%)
   - accept_rate < 0.6 → step_low *= 0.995  (-0.5%)
2. Diagonal mass matrix adaptation (every adapt_interval iterations):
   blended_M⁻¹ = α * new_M⁻¹ + (1−α) * old_M⁻¹   (α=0.2)
3. ε randomized each iteration: Uniform(step_low, 2*step_low)

After burn-in: all adaptation frozen (step_low, M⁻¹ fixed).
ε is still randomized for ergodicity.

Note: AdvancedHMC's built-in adaptors (StanHMCAdaptor, StepSizeAdaptor)
are NOT used because our BlockedSampler calls the low-level kernel API
directly. This function provides adaptation for all block types.
"""
function adjust_ϵ_heuristically!(burnin::Bool, blk::HMCBlock)
    if burnin && length(blk.accept_history) > 10
        # --- 1. Rolling acceptance rate → gentle step_low adjustment ---
        n_recent = min(length(blk.accept_history), 100)
        recent = @view blk.accept_history[end-n_recent+1:end]
        accept_rate = sum(recent) / n_recent
        if accept_rate > 0.9
            blk.step_low *= 1.005
        elseif accept_rate < 0.6
            blk.step_low *= 0.995
        end

        # --- 2. Diagonal mass matrix adaptation (DiagEuclideanMetric only) ---
        blk.iter_counter += 1
        if blk.iter_counter >= blk.adapt_interval && length(blk.sample_history) >= 10
            if blk.h.metric isa DiagEuclideanMetric
                xthsd = _empirical_std(blk.sample_history)
                mean_std = StatsBase.mean(xthsd)
                if mean_std > 1e-12 && all(xthsd .> 1e-12)
                    α = 0.2
                    new_M⁻¹ = (xthsd ./ mean_std) .^ 2
                    old_M⁻¹ = blk.h.metric.M⁻¹
                    blended_M⁻¹ = α .* new_M⁻¹ .+ (1 - α) .* old_M⁻¹
                    _update_metric!(blk, blended_M⁻¹)
                end
            end
            blk.iter_counter = 0
            empty!(blk.sample_history)
        end
    end
    # After burn-in: no adaptation (R MAGI freezes step_low, M⁻¹)

    # Randomize ε each iteration: Uniform(step_low, 2*step_low)
    ϵ = blk.step_low * (1.0 + rand())

    # Rebuild sampler with randomized ε, preserving trajectory type
    _rebuild_sampler_with_ϵ!(blk, ϵ)
end

"""Replace the integrator step size while preserving trajectory type (HMC/NUTS/HMCDA)."""
function _rebuild_sampler_with_ϵ!(blk::HMCBlock, ϵ::Float64)
    old_traj = blk.sampler.κ.τ
    new_traj = _replace_integrator(old_traj, Leapfrog(ϵ))
    new_kernel = AdvancedHMC.HMCKernel(new_traj)
    blk.sampler = HMCSampler(new_kernel, blk.h.metric, blk.sampler.adaptor)
end

"""Build a new Trajectory with a different integrator, preserving the trajectory
sampler type (EndPointTS / MultinomialTS) and termination criterion."""
_replace_integrator(traj::AdvancedHMC.Trajectory{TS}, integrator) where {TS} =
    AdvancedHMC.Trajectory{TS}(integrator, traj.termination_criterion)

"""Update the diagonal metric of an HMCBlock, rebuilding the Hamiltonian."""
function _update_metric!(blk::HMCBlock, M⁻¹::Vector{Float64})
    new_metric = DiagEuclideanMetric(M⁻¹)
    blk.h = AdvancedHMC.Hamiltonian(new_metric, blk.h.ℓπ, blk.h.∂ℓπ∂θ)
end

"""Record acceptance result into the HMCBlock's accept_history (capped at 100)."""
function record_accept!(blk::HMCBlock, accepted::Bool)
    push!(blk.accept_history, accepted)
    if length(blk.accept_history) > 100
        popfirst!(blk.accept_history)
    end
end

"""Record a sample into the HMCBlock's history buffer (for metric adaptation)."""
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
