```@meta
CurrentModule = NCCL
```

# NCCL.jl

 A Julia wrapper for the [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl).

# API

Buffers passed to collective and point-to-point operations must be stored contiguously in
memory of the device associated with the communicator: a `CuArray`, or a contiguous view or
reshape of one. Operations temporarily activate that device, and restore the previously
active device when they return. They are issued on the `stream` keyword argument, which
defaults to the task-local stream of the communicator's device (`CUDACore.stream()` while
that device is active), and must belong to that device; the per-thread default stream
(`per_thread_stream()`) is not supported. Arguments are validated before
calling NCCL: buffers whose lengths don't fit the operation throw a `DimensionMismatch`, and
other invalid arguments (e.g. non-contiguous buffers, mismatched element types, ranks out of
range, or partially overlapping buffers) an `ArgumentError`.

## Element types

NCCL natively supports buffers of `Int8`, `UInt8`, `Int32`, `UInt32`, `Int64`, `UInt64`,
`Float16`, `Float32`, `Float64` and `BFloat16` (from BFloat16s.jl) elements, which can be
used with every operation and reduction operator. Some other element types are passed to
NCCL as one of these:

- complex numbers with `Float16`, `Float32`, `Float64` or `BFloat16` parts are passed as
  pairs of real numbers. NCCL then reduces the real and imaginary parts separately, which is
  only correct for `+` and [`NCCL.avg`](@ref): reducing complex numbers with `*`, `max` or
  `min` throws an `ArgumentError`.
- `Bool`s are passed as bytes. They can only be reduced with `max` (logical or) and `min`
  (logical and), since sums and averages of booleans are not booleans.
- other `isbits` types (e.g. `Int16`, `Char`, tuples or immutable structs) are passed as
  bytes, and can only be copied: they are supported by data movement operations
  ([`Broadcast!`](@ref), [`Allgather!`](@ref), [`Send`](@ref) and [`Recv!`](@ref)), but not
  by reductions. Padding bytes in such values are copied as well.

Other element types, i.e. types that aren't `isbits` (including `isbits` unions), and types
of size zero (e.g. `Nothing`), are not supported.

## Library

```@docs
NCCL_jll.is_available
version()
```

## Errors

```@docs
NCCLError
```

## Communicators

```@docs
Communicator
Communicators
UniqueID
Config
rank(comm::Communicator)
nranks(comm::Communicator)
device(comm::Communicator)
destroy
abort
```

## Groups

```@docs
group
```

## Collective communication

```@docs
Allreduce!
Broadcast!
Reduce!
Allgather!
ReduceScatter!
avg
```

## Point-to-point communication

```@docs
Send
Recv!
```
