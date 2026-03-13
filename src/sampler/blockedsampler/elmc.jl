"""
ELMCBlock: Embedded Lagrangian Monte Carlo on Monge Patches

Independent implementation based on the paper:
  Hartmann, Girolami & Klami (2022). "Lagrangian Manifold Monte Carlo on
  Monge Patches." Proc. 25th AISTATS, PMLR 151. (arXiv:2202.00755)

Key equations used (references to the paper):
  - Monge metric: G_M(x) = I_D + α² ∇ℓ(x)∇ℓ(x)ᵀ  [Eq. (1)]
  - Inverse via Sherman-Morrison: G_M⁻¹ = I_D - α²∇ℓ∇ℓᵀ/(1+α²‖∇ℓ‖²)  [Sec. 3.2]
  - Determinant: det G_M = 1 + α²‖∇ℓ‖²  [Sec. 3.2]
  - Energy: E(x,v) = -ℓ(x) - ½log(L) + ½‖v‖² + (α²/2)⟨∇ℓ,v⟩²  [Sec. 4]
    where L = 1 + α²‖∇ℓ‖², ℓ(x) = log π_X(x)
  - Velocity half-step updates from Table 1
  - Determinant ratio: det(G ± ε/2 Ω̃)/(1+α²‖∇ℓ‖²) = 1 ± (α²ε/2)⟨∇ℓ,Hv⟩
  - Acceptance: α_LMC = min{1, exp(-E_diff)|det J|}  [Sec. 4]

The Hessian H(x) = ∇²ℓ(x) is computed via ForwardDiff.
"""

"""Pre-allocated workspace for ELMC leapfrog computations."""
mutable struct ELMCCache
    D::Int
end

"""Sample Block for Embedded Lagrangian Monte Carlo (Monge patch metric).

Implements LMC with the Monge metric G_M(x) = I + α²∇ℓ∇ℓᵀ from
Hartmann et al. (2022), using explicit Lagrangian dynamics that avoid
implicit equations and matrix inversions required by RMHMC.
"""
mutable struct ELMCBlock <: AbstractSampleBlock
    vars::Vector{Symbol}
    D::Int
    n::Int
    # Functions
    ℓπ::Function           # p -> log density (scalar)
    ∇ℓπ::Function          # p -> gradient (vector)
    Hℓπ::Function          # p -> Hessian (matrix)
    # ELMC parameters
    n_leapfrog::Int
    α::Float64             # Embedding parameter (α=0 → Euclidean HMC)
    adapt_scheme::String   # "0" (fixed), "1" (L-adaptive)
    # Cache
    cachm::ELMCCache
    # R MAGI-style adaptation state
    accept_history::Vector{Bool}
    step_low::Float64
    sample_history::Vector{Vector{Float64}}
    adapt_interval::Int
    iter_counter::Int
end

function ELMCBlock(mod::AbstractGM,
    vars::Vector{Symbol};
    n_leapfrog::Int,
    step_size::Float64=0.01,
    α::Float64=1.0,
    adapt_scheme::String="0",
    adapt_interval::Int=10,
)
    ℓπ = (p) -> ulogpdf(p, mod, vars)
    ∇ℓπ = (p) -> ∇ulogpdf(p, mod, vars)
    Hℓπ = (p) -> ForwardDiff.hessian(q -> ulogpdf(q, mod, vars), p)

    D = length(pack_param_vec(mod, vars))
    cachm = ELMCCache(D)

    ELMCBlock(vars, D, 1,
              ℓπ, ∇ℓπ, Hℓπ,
              n_leapfrog, α, adapt_scheme,
              cachm,
              Bool[], step_size, Vector{Float64}[], adapt_interval, 0)
end

Base.show(io::IO, blk::ELMCBlock) = print(
    io, "ELMCBlock(
    D=$(blk.D),
    n=$(blk.n),
    vars=$(blk.vars),
    n_leapfrog=$(blk.n_leapfrog),
    α=$(blk.α),
    step_low=$(blk.step_low),
    adapt_scheme=$(blk.adapt_scheme),
)")

# ─────────────────────────────────────────────────
# Core algorithm: one MCMC step
# ─────────────────────────────────────────────────

"""
    elmc_step(blk::ELMCBlock, x::Vector{Float64}) -> (x_new, accepted)

One MCMC step of LMC in the Monge metric (Hartmann et al., 2022).

Algorithm:
  1. Compute ∇ℓ at current position, form L = 1 + α²‖∇ℓ‖²
  2. Sample velocity v ~ N(0, G_M^{1/2})
  3. Compute initial energy E_ini
  4. Run L_F leapfrog steps (Table 1 of the paper), accumulating det J
  5. Compute proposed energy E_new
  6. Accept with probability min{1, exp(-E_new + E_ini + log|det J|)}
"""
function elmc_step(blk::ELMCBlock, x::Vector{Float64})
    D = blk.D
    α = blk.α
    α² = α^2

    # Randomize step size (R MAGI style)
    ε = blk.step_low * (1.0 + rand())
    # Random sign for ε (paper: randomize direction)
    ε = rand() > 0.5 ? ε : -ε

    xn = copy(x)

    # ── 1. Gradient at initial position ──
    ∇ℓ = blk.∇ℓπ(xn)
    nrm²_∇ℓ = dot(∇ℓ, ∇ℓ)
    L = 1.0 + α² * nrm²_∇ℓ

    # ── 2. Sample velocity from N(0, G_M⁻¹) ──
    # v = G_M^{-1/2} z,  z ~ N(0, I)  so that v^T G_M v = z^T z ~ χ²(D)
    # G_M^{-1/2} = I + c·∇ℓ∇ℓᵀ  where c = (1/√L - 1)/‖∇ℓ‖²
    vn = randn(D)
    if α > eps() && nrm²_∇ℓ > 1e-10
        c = (1.0 / sqrt(L) - 1.0) / nrm²_∇ℓ
        vn .+= c .* dot(∇ℓ, vn) .* ∇ℓ
    end

    # ── 3. Initial energy [Sec. 4] ──
    # E(x,v) = -ℓ(x) - ½log(L) + ½‖v‖² + (α²/2)⟨∇ℓ,v⟩²
    Eini = -blk.ℓπ(xn) - 0.5 * log(L) +
           0.5 * dot(vn, vn) + 0.5 * α² * dot(∇ℓ, vn)^2

    # ── 4. Step-size adaptation ──
    ε_eff = if blk.adapt_scheme == "1"
        ε * sqrt(L)
    else
        ε
    end

    # ── 5. Leapfrog integration [Table 1] with log|det J| accumulation ──
    log_det_J = 0.0

    for _ in 1:blk.n_leapfrog
        # Half-step velocity update + log-det
        vn, Δ_half1 = _lmc_monge_velocity_halfstep(blk, xn, vn, ε_eff)
        log_det_J += Δ_half1

        # Full-step position update: x ← x + ε·v
        xn .+= ε_eff .* vn

        # Half-step velocity update + log-det
        vn, Δ_half2 = _lmc_monge_velocity_halfstep(blk, xn, vn, ε_eff)
        log_det_J += Δ_half2
    end

    # ── 6. Proposed energy ──
    ∇ℓ_new = blk.∇ℓπ(xn)
    L_new = 1.0 + α² * dot(∇ℓ_new, ∇ℓ_new)

    Enew = -blk.ℓπ(xn) - 0.5 * log(L_new) +
           0.5 * dot(vn, vn) + 0.5 * α² * dot(∇ℓ_new, vn)^2

    # ── 7. Metropolis-Hastings accept/reject [Sec. 4] ──
    # α_LMC = min{1, exp(-E_diff) |det J|}
    logratio = -Enew + Eini + log_det_J
    accepted = isfinite(logratio) && logratio > min(0.0, log(rand()))

    if accepted
        return xn, true
    else
        return x, false
    end
end

# ─────────────────────────────────────────────────
# Velocity half-step: Table 1 of Hartmann et al. (2022)
# ─────────────────────────────────────────────────

"""
Velocity half-step update from Table 1 of Hartmann et al. (2022).

Given position x and velocity v, compute the updated velocity v' and
the log-determinant contribution Δlog|det|.

Table 1 formula (velocity update):
  v^{n+1/2} = [I_D - (∇ℓ(∇ℓᵀ + ε/2 vᵀH)) / (∇ℓᵀ(∇ℓ + ε/2 Hv) + 1/(Lα²))]
              × {[(α²∇ℓᵀv + ε/2)I_D - εα²/(2+2α²‖∇ℓ‖²) H] ∇ℓ + v}

Determinant ratio [Sec. 4, simplified]:
  det(G ± ε/2 Ω̃) / det(G) = 1 ± (α²ε/2)⟨∇ℓ/√L, H/√L · v⟩

We compute two log-det terms: one before and one after the velocity update.
"""
function _lmc_monge_velocity_halfstep(blk::ELMCBlock,
    x::Vector{Float64},
    v::Vector{Float64},
    ε::Float64)

    D = blk.D
    α = blk.α
    α² = α^2

    # Gradient and Hessian at current position (raw, not normalized)
    g = blk.∇ℓπ(x)               # g = ∇ℓ(x)
    H = α > eps() ? blk.Hℓπ(x) : zeros(D, D)
    nrm²_g = dot(g, g)
    L = 1.0 + α² * nrm²_g        # det G_M

    # Precompute Hv and Hg
    Hv = H * v
    Hg = H * g
    gᵀv = dot(g, v)
    gᵀHv = dot(g, Hv)

    # ── Log-det contribution (before velocity update) ──
    # det(G_M + ε/2 Ω̃_M) / det(G_M) = 1 + α²ε/(2L) · gᵀHv  [Sec. 4]
    Δlogdet_pre = -log(abs(1.0 + α² * ε / (2.0 * L) * gᵀHv))

    # ── Velocity update from Table 1 (raw gradient form) ──
    # Inner vector:  w = [(α²gᵀv + ε/2)I - εα²/(2L) H] g + v
    ε_half = ε / 2.0
    coeff_I = α² * gᵀv + ε_half
    coeff_H = ε * α² / (2.0 * L)
    inner = coeff_I .* g .- coeff_H .* Hg .+ v

    # Projection matrix: P = I - g(gᵀ + ε/2 vᵀH) / (gᵀg + ε/2 gᵀHv + 1/α²)
    denom = nrm²_g + ε_half * gᵀHv + 1.0 / α²
    proj_num = dot(g, inner) + ε_half * dot(Hv, inner)
    proj_coeff = proj_num / denom
    v_new = inner .- proj_coeff .* g

    # ── Log-det contribution (after velocity update) ──
    # det(G_M - ε/2 Ω̃_M) / det(G_M) = 1 - α²ε/(2L) · gᵀH·v_new  [Sec. 4]
    gᵀH_vnew = dot(g, H * v_new)
    Δlogdet_post = log(abs(1.0 - α² * ε / (2.0 * L) * gᵀH_vnew))

    Δlogdet = Δlogdet_pre + Δlogdet_post

    return v_new, Δlogdet
end

# ─────────────────────────────────────────────────
# R MAGI-style adaptation (same pattern as HMCBlock)
# ─────────────────────────────────────────────────

"""R MAGI-style adaptation for ELMCBlock (mirrors HMCBlock adaptation)."""
function adjust_ϵ_heuristically!(burnin::Bool, blk::ELMCBlock)
    if burnin && length(blk.accept_history) > 10
        n_recent = min(length(blk.accept_history), 100)
        recent = @view blk.accept_history[end-n_recent+1:end]
        accept_rate = sum(recent) / n_recent
        if accept_rate > 0.9
            blk.step_low *= 1.005
        elseif accept_rate < 0.6
            blk.step_low *= 0.995
        end
        blk.iter_counter += 1
    end
end

function record_accept!(blk::ELMCBlock, accepted::Bool)
    push!(blk.accept_history, accepted)
    if length(blk.accept_history) > 100
        popfirst!(blk.accept_history)
    end
end

function record_sample!(blk::ELMCBlock, param_vec::AbstractVector{<:Real})
    push!(blk.sample_history, Vector{Float64}(param_vec))
end
