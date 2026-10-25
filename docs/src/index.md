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

## Creating communicators with MPI

The ranks of a communicator need to agree on an [`NCCL.UniqueID`](@ref), which one of them
creates and sends to the others. In a program that uses MPI, loading MPI.jl enables
`NCCL.Communicator(mpicomm)`, which does this over an MPI communicator: rank 0 creates the
identifier and broadcasts it, and every MPI rank becomes the NCCL rank with the same index.

Each process has to use a different GPU of its node. Unless each process can only see a
single GPU (e.g. by setting `CUDA_VISIBLE_DEVICES` per process), select one based on the
rank of the process among the processes on the same node:

```julia
using MPI, CUDACore, NCCL
MPI.Init()

world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
node_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, world_rank)
MPI.Comm_size(node_comm) <= ndevices() ||
    error("more processes on this node than GPUs")
CUDACore.device!(MPI.Comm_rank(node_comm))

NCCL.Communicator(MPI.COMM_WORLD) do comm
    buf = CUDACore.fill(Float32(world_rank), 1024)
    NCCL.Allreduce!(buf, +, comm)
end
```

Run it with MPI.jl's launcher, e.g. `` `$(MPI.mpiexec()) -n 4 julia --project script.jl` ``.

## Element types

NCCL natively supports buffers of `Int8`, `UInt8`, `Int32`, `UInt32`, `Int64`, `UInt64`,
`Float16`, `Float32`, `Float64` and `BFloat16` (from BFloat16s.jl) elements, which can be
used with every operation and reduction operator. Some other element types are passed to
NCCL as one of these:

- complex numbers with `Float16`, `Float32`, `Float64` or `BFloat16` parts are passed as
  pairs of real numbers. NCCL then reduces the real and imaginary parts separately, which is
  only correct for `+`, [`NCCL.avg`](@ref) and [`NCCL.PreMulSum`](@ref) with a real scalar:
  reducing complex numbers with `*`, `max` or `min` throws an `ArgumentError`.
- `Bool`s are passed as bytes. They can only be reduced with `max` (logical or) and `min`
  (logical and), since sums and averages of booleans are not booleans.
- other `isbits` types (e.g. `Int16`, `Char`, tuples or immutable structs) are passed as
  bytes, and can only be copied: they are supported by data movement operations
  ([`Broadcast!`](@ref), [`Allgather!`](@ref), [`Alltoall!`](@ref), [`Gather!`](@ref),
  [`Scatter!`](@ref), [`Send`](@ref) and [`Recv!`](@ref)), but not by reductions. Padding bytes in such values are copied as well.

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
split
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

Reductions combine the values of all ranks with `+`, `*`, `max`, `min`, [`NCCL.avg`](@ref),
or [`NCCL.PreMulSum`](@ref), which multiplies each rank's values by a scalar before summing
them.

```@docs
Allreduce!
Broadcast!
Reduce!
Allgather!
ReduceScatter!
Alltoall!
Gather!
Scatter!
avg
PreMulSum
```

## Point-to-point communication

```@docs
Send
Recv!
```
