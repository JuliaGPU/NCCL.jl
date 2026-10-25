# Element types
#
# NCCL supports a fixed set of element types. Buffers of other element types are passed to
# NCCL as a buffer of one of these, with a different element count: complex numbers as
# pairs of real numbers, booleans as bytes, and any other isbits type as the bytes of its
# values. How an element type is lowered depends on the operation: data movement
# (broadcasts, gathers, sends and receives) only copies values, while a reduction has to
# combine them the way Julia would, which restricts the operators and types that can be
# used.

const NativeFloat = Union{Float16,Float32,Float64,BFloat16}
const NativeType = Union{Int8,UInt8,Int32,UInt32,Int64,UInt64,NativeFloat}

ncclDataType_t(::Type{Int8}) = ncclInt8
ncclDataType_t(::Type{UInt8}) = ncclUint8
ncclDataType_t(::Type{Int32}) = ncclInt32
ncclDataType_t(::Type{UInt32}) = ncclUint32
ncclDataType_t(::Type{Int64}) = ncclInt64
ncclDataType_t(::Type{UInt64}) = ncclUint64
ncclDataType_t(::Type{Float16}) = ncclFloat16
ncclDataType_t(::Type{Float32}) = ncclFloat32
ncclDataType_t(::Type{Float64}) = ncclFloat64
ncclDataType_t(::Type{BFloat16}) = ncclBfloat16


## reduction operators

const builtin_ops = (ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg)

# validate a reduction operator, returning the NCCL operator it corresponds to
reduction_op(op::Union{typeof(+),typeof(*),typeof(max),typeof(min)}) = ncclRedOp_t(op)
function reduction_op(op::ncclRedOp_t)
    op in builtin_ops || throw(ArgumentError(
        "unsupported reduction operator $op; use +, *, max, min, NCCL.avg or NCCL.PreMulSum"))
    return op
end
reduction_op(op::PreMulSum) = op
reduction_op(op) = throw(ArgumentError(
    "unsupported reduction operator $op; use +, *, max, min, NCCL.avg or NCCL.PreMulSum"))

opname(op) = string(op)
function opname(op::ncclRedOp_t)
    i = findfirst(==(op), builtin_ops)
    i === nothing ? string(op) : ("+", "*", "max", "min", "NCCL.avg")[i]
end

# Complex numbers are reduced as pairs of real numbers, which is only correct for operators
# that act on the real and imaginary parts independently. That includes multiplying by the
# real scalar of a `PreMulSum` (complex scalars are rejected when constructing one).
componentwise(op) = reduction_op(op) in (ncclSum, ncclAvg)
componentwise(::PreMulSum) = true

# Booleans are reduced as bytes holding 0 or 1. The maximum and minimum of those are the
# logical or and and, while sums and averages don't result in booleans.
logical(op) = reduction_op(op) in (ncclMax, ncclMin)
logical(::PreMulSum) = false


## lowering

# The NCCL element type and the number of its elements per element of type `T`, to copy
# values of type `T`.
movement_type(::Type{T}) where {T<:NativeType} = (ncclDataType_t(T), 1)
movement_type(::Type{Complex{T}}) where {T<:NativeFloat} = (ncclDataType_t(T), 2)
movement_type(::Type{Bool}) = (ncclUint8, 1)
function movement_type(::Type{T}) where {T}
    # (`T` is an isbits type, see `check_isbits`)
    # values of a zero-size type carry no data, and arrays of them don't have memory that
    # NCCL could access; rather than silently doing nothing, reject them
    sizeof(T) == 0 && throw(ArgumentError(
        "unsupported element type $T; values of this type have no data to communicate"))
    # the stride of the elements in the array, including any padding
    return ncclUint8, Base.aligned_sizeof(T)
end

# The NCCL element type and the number of its elements per element of type `T`, to reduce
# values of type `T` with `op`.
reduction_type(::Type{T}, op) where {T<:NativeType} = (ncclDataType_t(T), 1)
function reduction_type(::Type{Complex{T}}, op) where {T<:NativeFloat}
    componentwise(op) || throw(ArgumentError(
        "cannot reduce $(Complex{T}) values with $(opname(op)): NCCL reduces their real and imaginary parts separately, which is only correct for +, NCCL.avg and NCCL.PreMulSum"))
    return ncclDataType_t(T), 2
end
function reduction_type(::Type{Bool}, op)
    logical(op) || throw(ArgumentError(
        "cannot reduce Bool values with $(opname(op)); use max (logical or) or min (logical and), or convert to an integer type first"))
    return ncclUint8, 1
end
reduction_type(::Type{T}, op) where {T} = throw(ArgumentError(
    "cannot reduce $T values; reductions support Int8, UInt8, Int32, UInt32, Int64, UInt64, Float16, Float32, Float64, BFloat16, Bool, and Complex numbers of these floating-point types (other isbits types can only be copied)"))

# Arrays only store isbits values inline, so that NCCL can access them as a range of memory.
# (isbits unions are stored inline too, but with their type tags in a separate part of the
# array.) This has to be checked before dispatching on the element type, as an isbits union
# of supported types, e.g. `Union{Int8,UInt8}`, is a subtype of `NativeType` as well.
function check_isbits(::Type{T}) where {T}
    isbitstype(T) || throw(ArgumentError(
        "unsupported element type $T; NCCL operations only support isbits element types"))
    return
end

# NCCL counts elements of the lowered type
lowered_count(len::Int, multiplier::Int) = Csize_t(Base.checked_mul(len, multiplier))

# The NCCL element type and element count to copy a buffer of `len` elements of type `T`.
function lower_movement(::Type{T}, len::Int) where {T}
    check_isbits(T)
    datatype, multiplier = movement_type(T)
    return datatype, lowered_count(len, multiplier)
end

# The NCCL operator to reduce values of type `T`. NCCL multiplies by the scalar of a
# `PreMulSum` in the element type it reduces, so the scalar is converted to that type (the
# real type of complex numbers). This happens before `submit`, as converting a custom
# `Real` could yield.
lower_op(op::ncclRedOp_t, ::Type) = op
lower_op(op::PreMulSum, ::Type{T}) where {T} = PreMulSum(convert(real(T), op.scalar))

# The NCCL element type, element count and operator to reduce a buffer of `len` elements of
# type `T` with `op`. The operator is to be passed to `with_reduction_op`.
function lower_reduction(::Type{T}, len::Int, op) where {T}
    redop = reduction_op(op)
    check_isbits(T)
    datatype, multiplier = reduction_type(T, op)
    return datatype, lowered_count(len, multiplier), lower_op(redop, T)
end

# Call `f` with the handle of the lowered reduction operator `op` for `datatype`. A
# `PreMulSum` operator is created on the communicator for the duration of the call, so `f`
# must issue the NCCL operation using it. NCCL copies a custom operator when an operation
# using it is issued, also inside a group (`taskAppend` in its enqueue.cc), and the operator
# may be destroyed as soon as that call returns. Has to be called from the `submit`
# callback: creating and destroying an operator are NCCL calls on the communicator too.
with_reduction_op(f, op::ncclRedOp_t, datatype, comm_handle) = f(op)
function with_reduction_op(f, op::PreMulSum{T}, datatype, comm_handle) where {T}
    handle = Ref{ncclRedOp_t}()
    # with `ncclScalarHostImmediate`, NCCL copies the scalar before returning
    scalar = Ref{T}(op.scalar)
    GC.@preserve scalar begin
        ptr = Ptr{Cvoid}(Base.unsafe_convert(Ptr{T}, scalar))
        ncclRedOpCreatePreMulSum(handle, ptr, datatype, ncclScalarHostImmediate, comm_handle)
    end
    ret = try
        f(handle[])
    catch
        # the operation's error is the relevant one
        LibNCCL.unchecked_ncclRedOpDestroy(handle[], comm_handle)
        rethrow()
    end
    ncclRedOpDestroy(handle[], comm_handle)
    return ret
end
