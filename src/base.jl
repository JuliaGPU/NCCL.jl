import NCCL_jll: is_available

"""
    NCCL.is_available() :: Bool

Is the NCCL library available?
"""
is_available

"""
    NCCL.version() :: VersionNumber

Get the version of the current NCCL library.
"""
function version()
    ver_r = Ref{Cint}()
    ncclGetVersion(ver_r)
    ver = ver_r[]

    if ver < 2900
        major, ver = divrem(ver, 1000)
        minor, patch = divrem(ver, 100)
    else
        major, ver = divrem(ver, 10000)
        minor, patch = divrem(ver, 100)
    end
    VersionNumber(major, minor, patch)
end

import .LibNCCL: ncclRedOp_t, ncclDataType_t

ncclRedOp_t(::typeof(+)) = ncclSum
ncclRedOp_t(::typeof(*)) = ncclProd
ncclRedOp_t(::typeof(max)) = ncclMax
ncclRedOp_t(::typeof(min)) = ncclMin
# Handles the case where user directly passed in the ncclRedOp_t (eg. `NCCL.avg`)
ncclRedOp_t(x::ncclRedOp_t) = x

"""
    NCCL.avg

Perform an average operation, i.e. a sum across all ranks, divided by the number
of ranks. For integers, the sum wraps around on overflow, and the division truncates
towards zero.
"""
const avg = ncclAvg

"""
    NCCL.PreMulSum(scalar::Real)

Reduction operator that multiplies the values of every rank by `scalar` before summing
them: reducing `x` over the ranks results in `scalar*x₀ + scalar*x₁ + ...`. It can be used
with every reduction ([`Allreduce!`](@ref), [`Reduce!`](@ref) and
[`ReduceScatter!`](@ref)), e.g. `NCCL.Allreduce!(buf, NCCL.PreMulSum(0.5f0), comm)`.

The multiplication happens in the element type of the buffers: `scalar` is converted to it
with `convert` when the operation is issued. For floating-point buffers, it is rounded to
the nearest representable value. For integer buffers, it has to be representable exactly
(e.g. `NCCL.PreMulSum(0.5)` throws an `InexactError` with an `Int32` buffer), and the
products and sums wrap around on overflow, as Julia's integer arithmetic does. Complex
buffers are scaled by a real `scalar`, which is converted to the type of their real and
imaginary parts. `Bool` buffers cannot be reduced with `PreMulSum`.

Only real scalars are supported: NCCL multiplies the real and imaginary parts of complex
numbers separately, so it cannot multiply them by a complex number.

# External links
- [`ncclRedOpCreatePreMulSum`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum)
"""
struct PreMulSum{T<:Real}
    scalar::T
end

PreMulSum(scalar::Complex) = throw(ArgumentError(
    "NCCL.PreMulSum only supports real scalars: NCCL multiplies the real and imaginary parts of complex numbers separately, which cannot express multiplication by the complex number $scalar"))

function Base.show(io::IO, op::PreMulSum)
    print(io, "NCCL.PreMulSum(")
    show(io, op.scalar)
    print(io, ")")
end
