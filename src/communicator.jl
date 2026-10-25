# Communicator
export Communicator

const UniqueID = LibNCCL.ncclUniqueId

function UniqueID()
    r = Ref{UniqueID}()
    ncclGetUniqueId(r)
    return r[]
end

"""
    NCCL.Communicator

An NCCL communicator: one rank of a group of ranks that perform collective and
point-to-point operations together.

Communicators are not destroyed by the garbage collector. Destroying a communicator involves
communication with the other ranks, which can block, or deadlock when the ranks don't do so
at the same time, so it needs to happen at a well-defined point in the program: call
[`NCCL.destroy`](@ref) (or [`NCCL.abort`](@ref) after an error) when you are done with it,
or use the do-block form of the constructors. A communicator that is never destroyed keeps
its resources until the process exits.

After a communicator has been destroyed or aborted, it is closed (see `isopen`), and using
it throws an `InvalidStateException`.
"""
mutable struct Communicator
    handle::ncclComm_t      # C_NULL once destroyed or aborted
end

Base.isopen(comm::Communicator) = comm.handle != C_NULL

function check_open(comm::Communicator)
    isopen(comm) ||
        throw(InvalidStateException("the NCCL communicator has been destroyed or aborted",
                                    :closed))
    return
end

function Base.unsafe_convert(::Type{LibNCCL.ncclComm_t}, comm::Communicator)
    check_open(comm)
    comm.handle
end

function _buffer_device(buffer)
    applicable(CUDACore.device, buffer) && return CUDACore.device(buffer)
    if applicable(parent, buffer)
        parent_buffer = parent(buffer)
        parent_buffer === buffer || return _buffer_device(parent_buffer)
    end
    throw(ArgumentError("cannot determine the CUDA device for $(typeof(buffer))"))
end

function _check_buffer_devices(comm_device, buffers...)
    for buffer in buffers
        buffer_device = _buffer_device(buffer)
        buffer_device == comm_device || throw(ArgumentError(
            "buffer is on device $buffer_device, but the communicator is on device $comm_device"))
    end
end

# Issue an NCCL operation on `comm`, calling `f` with a raw device pointer for each of
# `buffers`, followed by the communicator and stream handles.
#
# Converting a buffer to a pointer can yield (CUDACore may synchronize), and while this task
# is suspended another task on the same thread may open a group. The ownership check
# therefore only happens once every argument has been converted, so that nothing can yield
# between the check and the native call that would join the group. For the same reason,
# callers have to convert scalar arguments (roots, peers) to their C type before calling
# `submit`: converting a custom `Integer` can yield as well.
function submit(f, comm::Communicator, stream::CuStream, buffers...)
    check_open(comm)
    comm_device = device(comm)
    _check_buffer_devices(comm_device, buffers...)
    CUDACore.device!(comm_device) do
        cbuffers = map(buf -> Base.cconvert(CuPtr{Cvoid}, buf), buffers)
        GC.@preserve buffers cbuffers comm stream begin
            ptrs = map(buf -> Base.unsafe_convert(CuPtr{Cvoid}, buf), cbuffers)
            comm_handle = Base.unsafe_convert(ncclComm_t, comm)
            stream_handle = Base.unsafe_convert(LibNCCL.cudaStream_t, stream)
            check_group_owner()
            f(ptrs..., comm_handle, stream_handle)
        end
    end
end


## destruction

# destroying or aborting a communicator inside a group would only take effect when the group
# ends, while the communicator would already be marked as closed
function check_no_group(f)
    check_group_owner()
    group_state() === nothing ||
        throw(ArgumentError("NCCL.$f cannot be called inside NCCL.group"))
    return
end

# Mark the communicator as closed before releasing it, so that it is never released twice.
# If the release fails, NCCL may or may not have freed the communicator (some error paths
# return before freeing it); trying again could free it twice, so it is leaked instead.
function release(f, comm::Communicator)
    handle = comm.handle
    comm.handle = C_NULL
    f(handle)
    return
end

"""
    NCCL.destroy(comm::Communicator)
    NCCL.destroy(comms::AbstractVector{Communicator})

Destroy the communicator `comm` (or all communicators in `comms`), after waiting for its
outstanding operations to complete. Afterwards, the communicator is closed and can no
longer be used. Destroying a closed communicator does nothing.

This first finalizes the communicator (`ncclCommFinalize`), which flushes its outstanding
operations and synchronizes with the other ranks, and then frees its local resources
(`ncclCommDestroy`). Finalization is an intra-node collective operation: all ranks on the
same node need to destroy their communicator, or this call blocks. When a task drives
several ranks of the same communicator, e.g. the communicators returned by
[`Communicators`](@ref), destroy them together by passing a vector: their finalization is
then grouped, as NCCL requires. Destroying them one by one may deadlock.

If finalization fails, the communicator remains open; use [`NCCL.abort`](@ref) to free it.

This function must not be called inside [`NCCL.group`](@ref).

# External links
- [`ncclCommFinalize`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommfinalize)
- [`ncclCommDestroy`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommdestroy)
"""
function destroy(comm::Communicator)
    isopen(comm) || return
    check_no_group(:destroy)
    ncclCommFinalize(comm)
    release(ncclCommDestroy, comm)
end

function destroy(comms::AbstractVector{Communicator})
    # finalizing a communicator twice in the same group would fail or hang
    comms = unique(filter(isopen, comms))
    isempty(comms) && return
    check_no_group(:destroy)
    group() do
        foreach(ncclCommFinalize, comms)
    end
    # after finalization, destroying is a local operation
    foreach(comm -> release(ncclCommDestroy, comm), comms)
end

"""
    NCCL.abort(comm::Communicator)
    NCCL.abort(comms::AbstractVector{Communicator})

Abort the communicator `comm` (or all communicators in `comms`): cancel its uncompleted
operations and free its resources. Afterwards, the communicator is closed and can no longer
be used. Aborting a closed communicator does nothing.

This is meant to release a communicator after an error. It does not wait for outstanding
operations or synchronize with the other ranks, but NCCL expects every rank to abort its
communicator, and peers that keep using theirs may block. Aborting a communicator is
not guaranteed to be able to interrupt an operation that is blocked inside NCCL, because
NCCL.jl only supports blocking communicators.

This function must not be called inside [`NCCL.group`](@ref).

# External links
- [`ncclCommAbort`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommabort)
- [Fault tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#fault-tolerance)
"""
function abort(comm::Communicator)
    isopen(comm) || return
    check_no_group(:abort)
    release(ncclCommAbort, comm)
end

function abort(comms::AbstractVector{Communicator})
    # abort as many communicators as possible, even if some fail
    err = nothing
    for comm in comms
        try
            abort(comm)
        catch ex
            err === nothing && (err = ex)
        end
    end
    err === nothing || throw(err)
    return
end

# Run `f(obj)`, and destroy `obj` afterwards, or abort it if `f` throws.
function with_communicators(f, obj)
    ret = try
        f(obj)
    catch
        try
            abort(obj)
        catch
            # the original exception is more informative
        end
        rethrow()
    end
    destroy(obj)
    return ret
end


## construction

"""
    NCCL.Communicator(nranks, rank; [unique_id]) :: Communicator
    NCCL.Communicator(f, nranks, rank; [unique_id])

Create a single communicator for use in a multi-threaded or multi-process
environment. `nranks` is the number of ranks in the communicator, and `rank`
is the 0-based index of the current rank. `unique_id` is an optional unique
identifier for the communicator.

With the do-block form, the communicator is passed to `f`, and destroyed when `f` returns
(see [`NCCL.destroy`](@ref)). If `f` throws, the communicator is aborted instead (see
[`NCCL.abort`](@ref)) and the exception is rethrown. This only cleans up the local
communicator: the other ranks still need to abort theirs, or they may block.

# Examples
```
comm = Communicator(length(CUDACore.devices()), id, myid())
# this blocks until all other ranks have connected
```

# External links
- [`ncclCommInitRank`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclCommInitRank)
"""
function Communicator(nranks::Integer, rank::Integer;
                      unique_id::UniqueID=UniqueID())
    0 <= rank < nranks || throw(ArgumentError("rank must be in [0, nranks)"))
    handle_ref = Ref{ncclComm_t}(C_NULL)
    nranks = Cint(nranks)
    rank = Cint(rank)
    check_group_owner()
    ncclCommInitRank(handle_ref, nranks, unique_id, rank)
    return Communicator(handle_ref[])
end

function Communicator(f, nranks::Integer, rank::Integer; kwargs...)
    with_communicators(f, Communicator(nranks, rank; kwargs...))
end

"""
    NCCL.Communicators(devices) :: Vector{Communicator}
    NCCL.Communicators(f, devices)

Construct and initialize a clique of NCCL Communicators over the devices
on a single host.

`devices` can either be a collection of identifiers, or `CuDevice`s.

Destroy the communicators together with `NCCL.destroy(comms)`, or use the do-block form,
which destroys them when `f` returns, and aborts them if it throws (see
[`NCCL.Communicator`](@ref)).

# Examples
```
comms = NCCL.Communicators(CUDACore.devices())
```

# External links
- [`ncclCommInitAll`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitall)
"""
function Communicators(deviceids::Vector{Cint})
    ndev    = length(deviceids)
    comms   = Vector{ncclComm_t}(undef, ndev)
    check_group_owner()
    ncclCommInitAll(comms, ndev, deviceids)
    return map(Communicator, comms)
end
function Communicators(deviceids::AbstractVector{<:Integer})
    Communicators(Cint[d for d in deviceids])
end
function Communicators(devices)
    Communicators(Cint[deviceid(d) for d in devices])
end

Communicators(f, devices) = with_communicators(f, Communicators(devices))

"""
    NCCL.device(comm::Communicator) :: CuDevice

The device of the communicator

# External Links
- [`ncclCommCuDevice`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommcudevice)
"""
function device(comm::Communicator)
    dev_ref = Ref{Cint}(C_NULL)
    ncclCommCuDevice(comm, dev_ref)
    return CuDevice(dev_ref[])
end

"""
    NCCL.size(comm::Communicator) :: Int

The number of communicators in the clique.

# External links
- [`ncclCommCount`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommcount)
"""
function size(comm::Communicator)
    size_ref = Ref{Cint}(C_NULL)
    ncclCommCount(comm, size_ref)
    return Int(size_ref[])
end

"""
    NCCL.rank(comm::Communicator) :: Int

The 0-based index of the communicator in the clique.

# External links
- [`ncclCommUserRank`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommuserrank)
"""
function rank(comm::Communicator)
    rank_ref = Ref{Cint}(C_NULL)
    ncclCommUserRank(comm, rank_ref)
    return Int(rank_ref[])
end

"""
    NCCL.default_device_stream(comm::Communicator) :: CuStream

Get the default stream for device `devid`, or the device corresponding to
communicator `comm`.
"""
function default_device_stream(comm::Communicator)
    dev = device(comm)
    device!(dev) do
        stream()
    end
end
