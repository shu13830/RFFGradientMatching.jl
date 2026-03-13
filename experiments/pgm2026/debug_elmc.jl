#!/usr/bin/env julia
# Debug ELMC: step-by-step diagnostics

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
println("D = $D")

# Functions
ℓπ = p -> ulogpdf(p, gm, vars)
∇ℓπ = p -> ∇ulogpdf(p, gm, vars)
Hℓπ = p -> ForwardDiff.hessian(q -> ulogpdf(q, gm, vars), p)

# === Test 1: Gradient consistency ===
println("\n=== Test 1: Gradient consistency ===")
g_manual = ∇ℓπ(x0)
g_ad = ForwardDiff.gradient(ℓπ, x0)
println("  ‖∇ℓ_manual‖ = $(norm(g_manual))")
println("  ‖∇ℓ_AD‖     = $(norm(g_ad))")
println("  ‖diff‖       = $(norm(g_manual - g_ad))")
println("  rel_err      = $(norm(g_manual - g_ad) / max(norm(g_ad), 1e-10))")

# === Test 2: Log density and gradient magnitudes ===
println("\n=== Test 2: Log density and gradient at x0 ===")
ℓ0 = ℓπ(x0)
println("  ℓ(x0) = $ℓ0")
println("  ‖∇ℓ(x0)‖ = $(norm(g_manual))")
H0 = Hℓπ(x0)
println("  ‖H(x0)‖_F = $(norm(H0))")
println("  H eigenvalues range: [$(minimum(eigvals(Symmetric(H0)))), $(maximum(eigvals(Symmetric(H0))))]")

# === Test 3: α=0 leapfrog should be standard leapfrog ===
println("\n=== Test 3: α=0 manual leapfrog vs ELMC ===")
ε = 0.001
n_lf = 5
v0 = randn(D)

# Standard leapfrog (Euclidean)
x_std = copy(x0)
v_std = copy(v0)
v_std .+= (ε/2) .* ∇ℓπ(x_std)
for i in 1:n_lf-1
    x_std .+= ε .* v_std
    v_std .+= ε .* ∇ℓπ(x_std)
end
x_std .+= ε .* v_std
v_std .+= (ε/2) .* ∇ℓπ(x_std)
E_std_ini = -ℓπ(x0) + 0.5*dot(v0,v0)
E_std_fin = -ℓπ(x_std) + 0.5*dot(v_std,v_std)
println("  Standard leapfrog: ΔE = $(E_std_fin - E_std_ini)")

# ELMC α=0 leapfrog (should match)
x_elmc = copy(x0)
v_elmc = copy(v0)  # α=0: G=I, so velocity sampling unchanged
α = 0.0
α² = α^2

# ELMC uses: position step xn += ε*vn, with velocity half-steps around it
# But ELMC does TWO half-steps per leapfrog iteration (Table 1 structure).
# Let's trace what happens...

# In elmc_step with α=0:
# L = 1 + 0*||∇ℓ||² = 1
# Eini = -ℓ(x) - 0 + 0.5*||v||² + 0 = -ℓ(x) + 0.5*||v||²  ✓
# Velocity sampling: no change ✓
# Each leapfrog step: half-step v, full-step x, half-step v

# Let's check what _lmc_monge_velocity_halfstep does with α=0
println("\n=== Test 4: Trace single ELMC half-step (α=0) ===")
∇ℓ = ∇ℓπ(x0)
nrm² = dot(∇ℓ, ∇ℓ)
L = 1.0 + α²*nrm²
sL = sqrt(L)
∇ℓ_n = ∇ℓ ./ sL  # = ∇ℓ (since sL=1)
H = zeros(D, D)  # α=0 → H=0
H_n = H ./ sL

# Log-det pre
H_n_v = H_n * v0  # = 0
Δlogdet_pre = -log(abs(1.0 + α²*(ε/2)*dot(∇ℓ_n, H_n_v)))  # = -log(1) = 0
println("  Δlogdet_pre = $Δlogdet_pre")

# Inner vector
ε_half = ε / 2.0
∇ℓᵀv = dot(∇ℓ_n, v0)
coeff_I = α² * L * ∇ℓᵀv + ε_half * sL  # = 0 + ε/2
coeff_H = ε * α² / 2.0  # = 0
inner = coeff_I .* ∇ℓ_n .- coeff_H .* (H_n * ∇ℓ_n) .+ v0  # = ε/2 * ∇ℓ + v0
println("  inner should be v0 + ε/2 * ∇ℓ: err = $(norm(inner - (v0 .+ ε_half .* ∇ℓ)))")

# Projection denominator
Hv_scaled = H_n_v .* ε_half  # = 0
denom = dot(∇ℓ_n, ∇ℓ_n) + dot(Hv_scaled, ∇ℓ_n) + 1.0 / (L * α²)
# With α=0: 1/(L*α²) = 1/0 = Inf!
println("  denom = $denom  (expect Inf due to 1/(L*α²))")

# proj_coeff = (dot(∇ℓ_n, inner) + ε_half*dot(H_n_v, inner)) / denom
# = finite / Inf = 0
proj_coeff = (dot(∇ℓ_n, inner) + ε_half*dot(H_n_v, inner)) / denom
println("  proj_coeff = $proj_coeff  (should be 0)")

v_new = inner .- proj_coeff .* ∇ℓ_n
println("  v_new == v0 + ε/2 * ∇ℓ? err = $(norm(v_new - (v0 .+ ε_half .* ∇ℓ)))")

# So α=0 half-step: v → v + ε/2 * ∇ℓ, and Δlogdet=0.  ✓
# But ELMC does: for each leapfrog step: half_step(v), full_step(x), half_step(v)
# This means each leapfrog step does TWO velocity half-steps and ONE position step.
# Standard leapfrog does: initial half-step, then (full position, full velocity) repeated, final half-step.
# The ELMC layout gives: v += ε/2 ∇ℓ, x += ε v, v += ε/2 ∇ℓ = net v += ε ∇ℓ per step.
# But standard stores the final half-step into the next initial half-step (leapfrog-stormer-verlet).
# ELMC does NOT combine the half-steps between successive leapfrog iterations!
# Step 1: v += ε/2 ∇ℓ(x), x += ε v, v += ε/2 ∇ℓ(x')
# Step 2: v += ε/2 ∇ℓ(x'), x += ε v, v += ε/2 ∇ℓ(x'')
# So between step 1 and 2, v gets ε/2 ∇ℓ(x') + ε/2 ∇ℓ(x') = ε ∇ℓ(x') ✓ (same as standard)
# Wait — that IS correct! The two half-steps at the boundary evaluate ∇ℓ at the SAME position.
# So it's equivalent to standard leapfrog. Good.

println("\n=== Test 5: Full ELMC trajectory (α=0) vs standard leapfrog ===")
x_e = copy(x0)
v_e = copy(v0)
log_det_J = 0.0

for step in 1:n_lf
    # Half-step velocity
    g = ∇ℓπ(x_e)
    v_e .+= (ε/2) .* g  # α=0 simplified
    # Full-step position
    x_e .+= ε .* v_e
    # Half-step velocity
    g = ∇ℓπ(x_e)
    v_e .+= (ε/2) .* g
end

println("  ‖x_elmc - x_std‖ = $(norm(x_e - x_std))")
println("  ‖v_elmc - v_std‖ = $(norm(v_e - v_std))")

# OK now test with α > 0
println("\n=== Test 6: α=1.0 single leapfrog step energy conservation ===")
α = 1.0
α² = α^2

for ε_test in [0.0001, 0.001, 0.01]
    x_t = copy(x0)
    v_t = randn(D)

    # Sample v from G_M^{1/2}
    ∇ℓ_t = ∇ℓπ(x_t)
    nrm²_t = dot(∇ℓ_t, ∇ℓ_t)
    L_t = 1.0 + α² * nrm²_t
    if nrm²_t > 1e-10
        c = (sqrt(L_t) - 1.0) / nrm²_t
        v_t .+= c * dot(∇ℓ_t, v_t) * ∇ℓ_t
    end

    E_ini = -ℓπ(x_t) - 0.5*log(L_t) + 0.5*dot(v_t,v_t) + 0.5*α²*dot(∇ℓ_t,v_t)^2

    # One leapfrog step using the actual ELMC code
    blk = ELMCBlock(gm, vars; n_leapfrog=1, step_size=ε_test, α=α)
    # We'll manually call the internals
    log_det_J = 0.0
    v_t2 = copy(v_t)
    x_t2 = copy(x_t)

    v_t2, Δ1 = RFFGradientMatching._lmc_monge_velocity_halfstep(blk, x_t2, v_t2, ε_test)
    log_det_J += Δ1
    x_t2 .+= ε_test .* v_t2
    v_t2, Δ2 = RFFGradientMatching._lmc_monge_velocity_halfstep(blk, x_t2, v_t2, ε_test)
    log_det_J += Δ2

    ∇ℓ_new = ∇ℓπ(x_t2)
    L_new = 1.0 + α² * dot(∇ℓ_new, ∇ℓ_new)
    E_fin = -ℓπ(x_t2) - 0.5*log(L_new) + 0.5*dot(v_t2,v_t2) + 0.5*α²*dot(∇ℓ_new,v_t2)^2

    ΔE = E_fin - E_ini
    logratio = -ΔE + log_det_J

    @printf("  ε=%.4f: ΔE=%.6e, log_det_J=%.6e, logratio=%.6e, |Δx|=%.6e, |Δv|=%.6e\n",
            ε_test, ΔE, log_det_J, logratio, norm(x_t2-x_t), norm(v_t2-v_t))
end

# === Test 7: Check if the projection is doing something weird ===
println("\n=== Test 7: Velocity half-step detail (α=1.0, ε=0.001) ===")
α = 1.0; α² = 1.0; ε = 0.001
v_test = randn(D)
∇ℓ_t = ∇ℓπ(x0)
H_t = Hℓπ(x0)
nrm² = dot(∇ℓ_t, ∇ℓ_t)
L_t = 1.0 + α² * nrm²
sL = sqrt(L_t)
∇ℓ_n = ∇ℓ_t ./ sL
H_n = H_t ./ sL

println("  ‖∇ℓ‖ = $(norm(∇ℓ_t)),  L = $L_t,  √L = $sL")
println("  ‖H‖_F = $(norm(H_t))")
println("  ‖∇ℓ_n‖ = $(norm(∇ℓ_n)),  ‖H_n‖_F = $(norm(H_n))")

H_n_v = H_n * v_test
ε_half = ε / 2.0
∇ℓᵀv = dot(∇ℓ_n, v_test)

# inner vector
coeff_I = α² * L_t * ∇ℓᵀv + ε_half * sL
coeff_H = ε * α² / 2.0
inner = coeff_I .* ∇ℓ_n .- coeff_H .* (H_n * ∇ℓ_n) .+ v_test

# projection
Hv_scaled = H_n_v .* ε_half
denom = dot(∇ℓ_n, ∇ℓ_n) + dot(Hv_scaled, ∇ℓ_n) + 1.0 / (L_t * α²)
proj_coeff = (dot(∇ℓ_n, inner) + ε_half * dot(H_n_v, inner)) / denom
v_new = inner .- proj_coeff .* ∇ℓ_n

println("  ‖inner‖ = $(norm(inner))")
println("  denom = $denom")
println("  proj_coeff = $proj_coeff")
println("  ‖v_test‖ = $(norm(v_test)),  ‖v_new‖ = $(norm(v_new))")
println("  ‖v_new - v_test‖ = $(norm(v_new - v_test))")
println("  Ratio ‖v_new‖/‖v_test‖ = $(norm(v_new)/norm(v_test))")

# Check: does the velocity grow over multiple half-steps?
println("\n=== Test 8: Velocity norm evolution over 10 half-steps ===")
x_ev = copy(x0)
v_ev = copy(v_test)
blk = ELMCBlock(gm, vars; n_leapfrog=1, step_size=0.001, α=1.0)
for i in 1:10
    v_ev, _ = RFFGradientMatching._lmc_monge_velocity_halfstep(blk, x_ev, v_ev, 0.001)
    x_ev .+= 0.001 .* v_ev
    @printf("  step %2d: ‖v‖=%.6e, ‖x-x0‖=%.6e, ℓ=%.6e\n",
            i, norm(v_ev), norm(x_ev - x0), ℓπ(x_ev))
end
