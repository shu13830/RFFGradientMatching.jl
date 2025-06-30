struct SigmoidKernel <: KernelFunctions.Kernel
    b::Float64  # Scaling parameter to adjust influence
    a::Float64  # Bias term

    SigmoidKernel(b::Float64, a::Float64) = new(b, a)
end

Base.show(io::IO, k::SigmoidKernel) = print(io, "SigmoidKernel(b = $(k.b), a = $(k.a))")

# Default constructor to provide initial values if none are given
SigmoidKernel() = SigmoidKernel(1.0, 0.0)

function (κ::SigmoidKernel)(x, y)
    return asin(
        (κ.a + κ.b*dot(x, y))
        /
        sqrt(
            (1 + κ.a + κ.b*sum(abs2, x)) *
            (1 + κ.a + κ.b*sum(abs2, y)))
    )
end

function kernelmatrix(
    k::SigmoidKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
)
    return kernelmatrix(k, _to_colvecs(x), _to_colvecs(y))
end

function kernelmatrix(k::SigmoidKernel, x::AbstractVector{<:Real})
    return kernelmatrix(k, _to_colvecs(x))
end

function kernelmatrix_diag(
    k::SigmoidKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
)
    return kernelmatrix_diag(k, _to_colvecs(x), _to_colvecs(y))
end

function kernelmatrix_diag(k::SigmoidKernel, x::AbstractVector{<:Real})
    return kernelmatrix_diag(k, _to_colvecs(x))
end

function kernelmatrix(κ::SigmoidKernel, x::ColVecs, y::ColVecs)
    validate_inputs(x, y)
    X_2 = sum(x.X .* x.X; dims=1) .* κ.b .+ κ.a .+ 1
    Y_2 = sum(y.X .* y.X; dims=1) .* κ.b .+ κ.a .+ 1
    XY = κ.b .* (x.X' * y.X) .+ κ.a
    return asin.(XY ./ sqrt.(X_2' * Y_2))
end

function kernelmatrix(κ::SigmoidKernel, x::ColVecs)
    X_2_1 = sum(x.X .* x.X; dims=1) .* κ.b .+ κ.a .+ 1
    XX = κ.b .* (x.X' * x.X) .+ κ.a
    return asin.(XX ./ sqrt.(X_2_1' * X_2_1))
end

function kernelmatrix_diag(κ::SigmoidKernel, x::ColVecs)
    x_2 = vec(sum(x.X .* x.X; dims=1) .* κ.b .+ κ.a .+ 1)
    return asin.((κ.b .* x_2 .+ κ.a) ./ x_2)
end

function kernelmatrix_diag(κ::SigmoidKernel, x::ColVecs, y::ColVecs)
    validate_inputs(x, y)
    x_2 = vec(sum(x.X .* x.X; dims=1) .* κ.b .+ κ.a .+ 1)
    y_2 = vec(sum(y.X .* y.X; dims=1) .* κ.b .+ κ.a .+ 1)
    xy = vec(sum(x.X' .* y.X'; dims=2) .* κ.b .+ κ.a)
    return asin.(xy ./ sqrt.(x_2 .* y_2))
end

# RowVecs kernelmatrix implementation
function kernelmatrix(κ::SigmoidKernel, x::RowVecs, y::RowVecs)
    validate_inputs(x, y)
    X_2 = sum(x.X .* x.X; dims=2) .* κ.b .+ κ.a .+ 1
    Y_2 = sum(y.X .* y.X; dims=2) .* κ.b .+ κ.a .+ 1
    XY = κ.b .* (x.X * y.X') .+ κ.a
    return asin.(XY ./ sqrt.(X_2 * Y_2'))
end

function kernelmatrix(κ::SigmoidKernel, x::RowVecs)
    X_2_1 = sum(x.X .* x.X; dims=2) .* κ.b .+ κ.a .+ 1
    XX = κ.b .* (x.X * x.X') .+ κ.a
    return asin.(XX ./ sqrt.(X_2_1 * X_2_1'))
end

function kernelmatrix_diag(κ::SigmoidKernel, x::RowVecs)
    x_2 = vec(sum(x.X .* x.X; dims=2) .* κ.b .+ κ.a .+ 1)
    return asin.((κ.b .* x_2 .+ κ.a) ./ x_2)
end

function kernelmatrix_diag(κ::SigmoidKernel, x::RowVecs, y::RowVecs)
    validate_inputs(x, y)
    x_2 = vec(sum(x.X .* x.X; dims=2) .* κ.b .+ κ.a .+ 1)
    y_2 = vec(sum(y.X .* y.X; dims=2) .* κ.b .+ κ.a .+ 1)
    xy = vec(sum(x.X .* y.X; dims=2) .* κ.b .+ κ.a)
    return asin.(xy ./ sqrt.(x_2 .* y_2))
end
