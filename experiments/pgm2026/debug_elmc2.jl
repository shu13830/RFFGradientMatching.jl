#!/usr/bin/env julia
# Debug ELMC v2: trace each sub-step of a single leapfrog iteration

include(joinpath(@__DIR__, "common.jl"))
using RFFGradientMatching: BlockedSamplerState, ulogpdf, ∇ulogpdf
using ForwardDiff, LinearAlgebra, Printf

Random.seed!(42)
config = ODE_CONFIGS["LV"]
times, y_obs, _, prob = generate_data(config; N=10, seed=42)
gm = setup_model(GPGM, config, times, y_obs, prob; anneal_length=1)
gm.anneal_iter[1] = gm.anneal_length
gm.β[1] = 1.0

vars = [:X, :θ]
x0 = pack_param_vec(gm, vars)
D = length(x0)

ℓπ = p -> ulogpdf(p, gm, vars)
∇ℓπ = p -> ∇ulogpdf(p, gm, vars)
Hℓπ = p -> ForwardDiff.hessian(q -> ulogpdf(q, gm, vars), p)

println("D = $D")
println("ℓ(x0) = $(ℓπ(x0))")

# Use the actual _lmc_monge_velocity_halfstep
blk = ELMCBlock(gm, vars; n_leapfrog=1, step_size=0.001, α=1.0)

α = 1.0; α² = 1.0

# Step-by-step trace for ε=0.0001
for ε in [0.0001, 0.001]
    println("\n========== ε = $ε ==========")
    Random.seed!(123)
    x = copy(x0)
    v = randn(D)

    # Velocity from G_M^{1/2}
    g0 = ∇ℓπ(x)
    nrm²_g0 = dot(g0, g0)
    L0 = 1.0 + α² * nrm²_g0
    c0 = (1.0/sqrt(L0) - 1.0) / nrm²_g0
    v .+= c0 * dot(g0, v) * g0

    @printf("  Initial: ‖x‖=%.4e, ‖v‖=%.4e, ℓ=%.4e\n", norm(x), norm(v), ℓπ(x))

    E_ini = -ℓπ(x) - 0.5*log(L0) + 0.5*dot(v,v) + 0.5*α²*dot(g0,v)^2
    @printf("  E_ini = %.4e\n", E_ini)

    # One leapfrog step: half-v, full-x, half-v
    # --- Half step 1 ---
    v1, Δld1 = RFFGradientMatching._lmc_monge_velocity_halfstep(blk, x, v, ε)
    @printf("  After half-step 1: ‖v‖=%.4e, Δlogdet=%.4e, ‖Δv‖=%.4e\n",
            norm(v1), Δld1, norm(v1-v))

    # --- Full position step ---
    x1 = x .+ ε .* v1
    @printf("  After pos step:    ‖x‖=%.4e, ‖Δx‖=%.4e, ℓ=%.4e\n",
            norm(x1), norm(x1-x), ℓπ(x1))

    # --- Half step 2 ---
    v2, Δld2 = RFFGradientMatching._lmc_monge_velocity_halfstep(blk, x1, v1, ε)
    @printf("  After half-step 2: ‖v‖=%.4e, Δlogdet=%.4e, ‖Δv‖=%.4e\n",
            norm(v2), Δld2, norm(v2-v1))

    # Final energy
    g_new = ∇ℓπ(x1)
    L_new = 1.0 + α² * dot(g_new, g_new)
    E_fin = -ℓπ(x1) - 0.5*log(L_new) + 0.5*dot(v2,v2) + 0.5*α²*dot(g_new,v2)^2
    @printf("  E_fin = %.4e, ΔE = %.4e\n", E_fin, E_fin - E_ini)

    # --- Detailed breakdown of half-step 1 ---
    println("\n  --- Half-step 1 internals (raw g) ---")
    g = ∇ℓπ(x)
    H = Hℓπ(x)
    nrm²_g = dot(g, g)
    L = 1.0 + α² * nrm²_g
    Hv = H * v
    Hg = H * g
    gᵀv = dot(g, v)
    gᵀHv = dot(g, Hv)

    @printf("    ‖g‖=%.4e, L=%.4e\n", sqrt(nrm²_g), L)
    @printf("    gᵀv=%.4e, gᵀHv=%.4e\n", gᵀv, gᵀHv)

    coeff_I = α² * gᵀv + ε/2.0
    coeff_H = ε * α² / (2.0 * L)
    @printf("    coeff_I=%.4e, coeff_H=%.4e\n", coeff_I, coeff_H)

    inner_raw = coeff_I .* g .- coeff_H .* Hg .+ v
    @printf("    ‖inner‖=%.4e\n", norm(inner_raw))
    @printf("    ‖coeff_I*g‖=%.4e, ‖coeff_H*Hg‖=%.4e, ‖v‖=%.4e\n",
            abs(coeff_I)*norm(g), coeff_H*norm(Hg), norm(v))

    denom = nrm²_g + ε/2.0 * gᵀHv + 1.0/α²
    proj_num = dot(g, inner_raw) + ε/2.0 * dot(Hv, inner_raw)
    proj_c = proj_num / denom
    @printf("    denom=%.4e, proj_num=%.4e, proj_coeff=%.4e\n", denom, proj_num, proj_c)

    v_result = inner_raw .- proj_c .* g
    @printf("    ‖v_result‖=%.4e, should match ‖v1‖=%.4e\n", norm(v_result), norm(v1))
    @printf("    v_result ≈ v1? err=%.4e\n", norm(v_result - v1))
end

# Also test: what does standard HMC look like?
println("\n========== Standard HMC (α=0) reference ==========")
for ε in [0.0001, 0.001, 0.01, 0.05]
    Random.seed!(123)
    x = copy(x0)
    v = randn(D)
    E_ini = -ℓπ(x) + 0.5*dot(v,v)

    # 10 leapfrog steps
    v .+= (ε/2) .* ∇ℓπ(x)
    for i in 1:9
        x .+= ε .* v
        v .+= ε .* ∇ℓπ(x)
    end
    x .+= ε .* v
    v .+= (ε/2) .* ∇ℓπ(x)

    E_fin = -ℓπ(x) + 0.5*dot(v,v)
    @printf("  ε=%.4f, n_lf=10: ΔE=%.4e, ‖Δx‖=%.4e\n", ε, E_fin-E_ini, norm(x-x0))
end
