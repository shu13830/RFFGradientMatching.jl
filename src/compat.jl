# Generalized RFF integration layer.
# Extends the kernel parameter system (params/kernel_inner/only_params from gp.jl)
# to support GeneralizedRandomFourierFeatures kernels, and provides a unified build_rff_basis() function.

# ── Part A: Kernel parameter dispatch for generalized kernels ────

# Base-case params: generalized kernels have unit ℓ=1.0, α=1.0 by default.
# ScaledKernel/TransformedKernel wrapping (already in gp.jl) handles user-specified ℓ and σ².
params(k::GeneralizedRandomFourierFeatures.GeneralizedCauchyKernel) = (k, 1.0, 1.0)
params(k::GeneralizedRandomFourierFeatures.GammaExponentialKernel) = (k, 1.0, 1.0)

only_params(k::GeneralizedRandomFourierFeatures.GeneralizedCauchyKernel) = [1.0, 1.0]
only_params(k::GeneralizedRandomFourierFeatures.GammaExponentialKernel) = [1.0, 1.0]

# kernel_inner: called by reconstruct_kernel() to rebuild kernel with new lengthscale
kernel_inner(k::GeneralizedRandomFourierFeatures.GeneralizedCauchyKernel, inner::AbstractVector{<:Real}) =
    with_lengthscale(GeneralizedCauchyKernel(only(k.α), only(k.β)), inner[1])
kernel_inner(k::GeneralizedRandomFourierFeatures.GammaExponentialKernel, inner::AbstractVector{<:Real}) =
    with_lengthscale(GeneralizedRandomFourierFeatures.GammaExponentialKernel(γ=only(k.γ)), inner[1])

# Matern52Kernel: convert to MaternKernel(ν=2.5) for GeneralizedRandomFourierFeatures compatibility.
# Matern52Kernel already has params/kernel_inner/only_params in gp.jl, so no dispatch needed there.

# ── Part B: Base kernel conversion for GeneralizedRandomFourierFeatures ─────────────

# Convert kernel to a type that GeneralizedRandomFourierFeatures.sample_generalized_rff_basis accepts
_to_grff_base(k::GeneralizedRandomFourierFeatures.GeneralizedCauchyKernel) = k
_to_grff_base(k::GeneralizedRandomFourierFeatures.GammaExponentialKernel) = k
_to_grff_base(::Matern52Kernel) = KernelFunctions.MaternKernel(ν=2.5)
_to_grff_base(k) = k  # SqExponentialKernel etc. pass through as-is

# ── Part C: Unified RFF basis builder ────────────────────────────

"""
    build_rff_basis(k::KernelFunctions.Kernel, input_dims::Int, n_rff::Int)

Build an `RFFBasis` for any supported kernel (standard or generalized).
Handles ScaledKernel/TransformedKernel wrapping to extract lengthscale and
output scale, then delegates to `GeneralizedRandomFourierFeatures.sample_generalized_rff_basis`.
"""
function build_rff_basis(k::KernelFunctions.Kernel, input_dims::Int, n_rff::Int)
    base_k, inner, outer = params(k)
    grff_base = _to_grff_base(base_k)
    h_raw = GeneralizedRandomFourierFeatures.sample_generalized_rff_basis(
        Random.default_rng(), grff_base, input_dims, n_rff)
    ℓ = isa(inner, Tuple) ? inner[1] : inner
    new_outer = outer * h_raw.outer_weights
    return RFFBasis(ℓ, new_outer, h_raw.ω, h_raw.τ, h_raw.sample_params)
end

# ── Part D: RFF approximation error metric ───────────────────────

"""
    rff_approx_error(h::RFFBasis, k::KernelFunctions.Kernel, t::AbstractVector{<:Real})

Compute the normalized Frobenius error ||ΦΦ' - K||_F / N between the RFF
kernel approximation and the exact kernel matrix at time points `t`.
"""
function rff_approx_error(h::RFFBasis, k::KernelFunctions.Kernel, t::AbstractVector{<:Real})
    N = length(t)
    K_exact = KernelFunctions.kernelmatrix(k, t)
    H = h(RowVecs(reshape(t, :, 1))).X  # N × n_rff
    K_rff = H * H'
    return LinearAlgebra.norm(K_rff - K_exact) / N
end
