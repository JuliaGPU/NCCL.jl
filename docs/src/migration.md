# Migrating from NCCL.jl 0.1

NCCL.jl 1.0 changes how communicators are created and destroyed, and checks the arguments of
operations more strictly. This page lists the changes that can break code written for 0.1.


## Dependencies

NCCL.jl now depends on CUDACore, the core of CUDA.jl, instead of on CUDA.jl itself. Code
that relied on `using NCCL` to load CUDA.jl has to load it, or CUDACore, itself:

```julia
using CUDA, NCCL       # or: using CUDACore, NCCL
```


## Creating communicators

The `UniqueID` is now a required positional argument. In 0.1, it defaulted to a new
identifier, which differs on every rank, so the default only worked for a single rank:

```julia
# 0.1
comm = NCCL.Communicator(nranks, rank; unique_id=id)
comm = NCCL.Communicator(1, 0)

# 1.0
comm = NCCL.Communicator(nranks, rank, id)
comm = NCCL.Communicator(1, 0, NCCL.UniqueID())
```

See [Getting started](@ref) for ways to share the identifier between processes, including
the new MPI integration.

`NCCL.Communicators` throws an `ArgumentError` for an empty list of devices, or when a
device occurs more than once.


## Destroying communicators

Communicators are no longer destroyed by the garbage collector, as that could deadlock
(see [Communicator lifetime](@ref)). Destroy them explicitly, or use the do-block form:

```julia
# 0.1
comm = NCCL.Communicator(nranks, rank; unique_id=id)
# ... (destroyed by the garbage collector at some point)

# 1.0
comm = NCCL.Communicator(nranks, rank, id)
# ...
NCCL.destroy(comm)

# or
NCCL.Communicator(nranks, rank, id) do comm
    # ...
end
```

Communicators that one task drives have to be destroyed together:

```julia
# 0.1
NCCL.destroy.(comms)

# 1.0
NCCL.destroy(comms)
```

`NCCL.destroy` now waits for outstanding operations and synchronizes with the other ranks
(`ncclCommFinalize`) before freeing the communicator. After `NCCL.destroy` or `NCCL.abort`,
using the communicator throws an `InvalidStateException`; `NCCL.abort` used to leave the
object pointing to freed memory. Neither function can be called inside `NCCL.group`.


## Renamed and removed functions

```julia
# 0.1
NCCL.size(comm)
# 1.0
NCCL.nranks(comm)
```

`NCCL.groupStart()` and `NCCL.groupEnd()` are removed, as code between them was not
protected against exceptions or task switches. Use `NCCL.group` instead:

```julia
# 0.1
NCCL.groupStart()
for (comm, buf) in zip(comms, bufs)
    NCCL.Allreduce!(buf, +, comm)
end
NCCL.groupEnd()

# 1.0
NCCL.group() do
    for (comm, buf) in zip(comms, bufs)
        NCCL.Allreduce!(buf, +, comm)
    end
end
```

`NCCL.default_device_stream(comm)` is removed. Operations use the stream it returned when no
`stream` is passed:

```julia
# 0.1
NCCL.Allreduce!(buf, +, comm; stream=NCCL.default_device_stream(comm))

# 1.0
NCCL.Allreduce!(buf, +, comm)
# or, to get that stream:
s = CUDACore.device!(CUDACore.stream, NCCL.device(comm))
```

`NCCL.rank`, `NCCL.nranks` and `NCCL.device` now return values cached when the communicator
was created.


## Group semantics

`NCCL.group` now pins the calling task to its thread, permanently, and NCCL calls from other
tasks on the same thread throw a `ConcurrencyViolationError` while the group is open. See
[Groups](@ref) for the rules.


## Operations

Operations now validate their arguments before calling NCCL. Code that worked by accident
may now throw:

- a `DimensionMismatch` for buffers whose lengths don't fit the operation, which used to fail
  an `@assert` or go unnoticed;
- an `ArgumentError` for non-contiguous views (which used to send the wrong data), host
  arrays and other array types, mismatched element types, roots or peers out of range,
  partially overlapping buffers, streams of another device, and the per-thread default
  stream.

Reductions with an unsupported operator throw an `ArgumentError` instead of a `MethodError`.
A raw `ncclRedOp_t` other than the built-in operators (e.g. created with
`ncclRedOpCreatePreMulSum`) is rejected; use [`NCCL.PreMulSum`](@ref) instead.

`NCCL.Recv!` returns the receive buffer, instead of its internal storage:

```julia
# 0.1
NCCL.Recv!(buf, comm; source)   # returned buf.data
# 1.0
NCCL.Recv!(buf, comm; source)   # returns buf
```


## Errors

`NCCLError` now holds NCCL's error message in its `msg` field (a `String`, empty when NCCL
didn't report a new one), instead of the description of the result code, and is displayed
with `showerror`.


## Environment

When `NCCL_COMM_BLOCKING` is not set, loading NCCL.jl sets it to 1, and creating a
communicator throws when it is set to 0 (see [Nonblocking communicators](@ref)). Processes
started from Julia inherit the variable.


## Low-level bindings

The bindings in `NCCL.LibNCCL` have been regenerated against NCCL 2.32. Element counts are
now `Csize_t` (they were wrongly `Cint`), `ncclMemAlloc` takes a `Ptr{CuPtr{Cvoid}}`, and
all functions are called with `@gcsafe_ccall`.


## New features

These don't break existing code: the [`Alltoall!`](@ref NCCL.Alltoall!),
[`Gather!`](@ref NCCL.Gather!) and [`Scatter!`](@ref NCCL.Scatter!) collectives;
communicator configuration with [`NCCL.Config`](@ref); [`NCCL.split`](@ref); creating
communicators from MPI communicators; [`NCCL.PreMulSum`](@ref); support for complex numbers,
`Bool`, `BFloat16` and other `isbits` element types; `nothing` for buffers that only matter on
the root; single-buffer in-place forms of `Allgather!` and `ReduceScatter!`; and do-block
forms of the constructors.
