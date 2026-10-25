# Communicator
export Communicator

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
it throws an `InvalidStateException`. Its [`NCCL.rank`](@ref), [`NCCL.nranks`](@ref) and
[`NCCL.device`](@ref) remain available.

Communicators are always blocking; see [`NCCL.Config`](@ref).
"""
mutable struct Communicator
    handle::ncclComm_t      # C_NULL once destroyed or aborted
    const device::CuDevice
    const context::CuContext
    const rank::Int
    const nranks::Int
    # NCCL keeps pointers to the strings of the configuration
    const config::Union{Config,Nothing}

    function Communicator(handle::ncclComm_t, dev::CuDevice, rank::Integer, nranks::Integer,
                          config::Union{Config,Nothing})
        comm = new(handle, dev, CUDACore.context(dev), rank, nranks, config)
        @lock open_lock push!(open_communicators, comm)
        return comm
    end
end

# NCCL uses the configuration of a communicator (see above) until it is destroyed, which
# can be after the Julia object would have been collected, as communicators are only freed
# explicitly. Open communicators are therefore kept alive here.
const open_communicators = Base.IdSet{Communicator}()
const open_lock = Threads.SpinLock()

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

function check_stream(comm::Communicator, stream::CuStream)
    # the per-thread default stream is a different stream on every OS thread, while CUDACore
    # identifies streams by their handle: a task that migrates to another thread would not
    # wait for its own work. the legacy default stream (also `default_stream()`, as neither
    # CUDACore nor NCCL use the per-thread API variants) is a single stream per context,
    # which synchronizes with CUDACore's (blocking) task streams, so it is safe to use.
    if stream.handle == CUDACore.CU_STREAM_PER_THREAD
        throw(ArgumentError(
            "NCCL operations cannot be issued on the per-thread default stream; use a `CuStream` or the task's stream instead"))
    end
    # the default streams aren't bound to a context
    if stream.ctx !== nothing && stream.ctx != comm.context
        throw(ArgumentError(
            "stream belongs to a different context than the communicator, which is on device $(comm.device)"))
    end
    return
end

# Issue an NCCL operation on `comm`, calling `f` with a raw device pointer for each of
# `buffers`, followed by the communicator and stream handles. The buffers have to be
# validated by the caller (see `check_buffer`).
#
# Unless a stream is given, the operation is issued on the task-local stream of the
# communicator's device. The buffers are converted to pointers while that stream is the
# task's current stream, as CUDACore attributes the use of memory whose pointer is taken to
# the current stream: it then waits for work on other streams that used the memory before,
# and later uses of the memory on other streams wait for the NCCL operation.
#
# Converting a buffer to a pointer can yield (CUDACore may synchronize), and while this task
# is suspended another task on the same thread may open a group. The ownership check
# therefore only happens once every argument has been converted, so that nothing can yield
# between the check and the native call that would join the group. For the same reason,
# callers have to convert scalar arguments (roots, peers) to their C type before calling
# `submit`: converting a custom `Integer` can yield as well.
#
# Inside a group, NCCL only enqueues the operation when the outermost group ends, so the
# arguments are kept alive until then.
function submit(f, comm::Communicator, stream::Union{CuStream,Nothing}, buffers...)
    check_open(comm)
    CUDACore.context!(comm.context) do
        # the task's stream can be set to anything with `stream!`
        s = stream === nothing ? CUDACore.stream() : stream
        check_stream(comm, s)
        CUDACore.stream!(s) do
            cbuffers = map(buf -> Base.cconvert(CuPtr{Cvoid}, buf), buffers)
            GC.@preserve buffers cbuffers comm s begin
                ptrs = map(buf -> Base.unsafe_convert(CuPtr{Cvoid}, buf), cbuffers)
                comm_handle = Base.unsafe_convert(ncclComm_t, comm)
                stream_handle = Base.unsafe_convert(LibNCCL.cudaStream_t, s)
                check_group_owner()
                retain_in_group(buffers, cbuffers, comm, s)
                f(ptrs..., comm_handle, stream_handle)
            end
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
    @lock open_lock delete!(open_communicators, comm)
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

# Abort the communicators that NCCL allocated before their initialization failed, as NCCL
# leaves that to the user. The error that caused the failure is more informative than one
# from aborting, so the latter is ignored.
function abort_failed(handles)
    # inside a group, aborting would be deferred until the group ends, when NCCL discards
    # it together with the rest of the failed group, so such communicators are leaked
    group_state() === nothing || return
    for handle in handles
        handle == C_NULL && continue
        try
            ncclCommAbort(handle)
        catch
        end
    end
end

function check_rank(nranks, rank)
    nranks >= 1 || throw(ArgumentError("nranks must be positive, got $nranks"))
    0 <= rank < nranks || throw(ArgumentError("rank must be in [0, $nranks), got $rank"))
    return
end

"""
    NCCL.Communicator(nranks, rank, id::UniqueID;
                      device=CUDACore.device(), config=nothing) :: Communicator
    NCCL.Communicator(f, nranks, rank, id::UniqueID; device, config)

Create the communicator for rank `rank` (0-based) of a communicator with `nranks` ranks, on
`device`. All ranks have to call this with the same `nranks` and the same `id`, which one
of them created with [`NCCL.UniqueID()`](@ref) and shared with the others, e.g. using MPI
or a file. Each rank uses a different device. The call blocks until all ranks have joined,
so the ranks have to be created concurrently: from different processes, or from different
tasks running on different threads.

`config` can be an [`NCCL.Config`](@ref) to configure the communicator.

If initialization fails, the local communicator is aborted before the error is thrown. Inside
an [`NCCL.group`](@ref), initialization only happens when the group ends; if it fails
there, the error is thrown by `NCCL.group`, and the communicator has to be aborted.

With the do-block form, the communicator is passed to `f`, and destroyed when `f` returns
(see [`NCCL.destroy`](@ref)). If `f` throws, the communicator is aborted instead (see
[`NCCL.abort`](@ref)) and the exception is rethrown. This only cleans up the local
communicator: it does not recover from the error, and the other ranks need to abort their
communicators as well, or they may block.

# Examples
```julia
# on every rank, e.g. a process using a single GPU
id = rank == 0 ? NCCL.UniqueID() : receive_id_from_rank0()
Communicator(nranks, rank, id) do comm
    NCCL.Allreduce!(buf, +, comm)
end
```

# External links
- [`ncclCommInitRankConfig`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrankconfig)
- [Creating a communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
"""
function Communicator(nranks::Integer, rank::Integer, id::UniqueID;
                      device::Union{CuDevice,Integer}=CUDACore.device(),
                      config::Union{Config,Nothing}=nothing)
    device isa CuDevice || (device = CuDevice(device))
    nranks = Cint(nranks)
    rank = Cint(rank)
    check_rank(nranks, rank)
    check_blocking()
    config = something(config, Config())
    raw_config = Ref(ncclConfig_t(config))
    handle_ref = Ref{ncclComm_t}(C_NULL)
    GC.@preserve config try
        CUDACore.device!(device) do
            check_group_owner()
            ncclCommInitRankConfig(handle_ref, nranks, id, rank, raw_config)
        end
    catch
        abort_failed((handle_ref[],))
        rethrow()
    end
    return Communicator(handle_ref[], device, rank, nranks, config)
end

function Communicator(f, nranks::Integer, rank::Integer, id::UniqueID; kwargs...)
    with_communicators(f, Communicator(nranks, rank, id; kwargs...))
end

"""
    NCCL.Communicators(devices; config=nothing) :: Vector{Communicator}
    NCCL.Communicators(f, devices; config=nothing)

Create a communicator with one rank per device in `devices`, all driven from the current
task. Rank `i-1` uses `devices[i]`; the devices must be distinct. `devices` can contain
`CuDevice`s or device ordinals.

`config` can be an [`NCCL.Config`](@ref) to configure the communicator. Without it, this
uses `ncclCommInitAll`, otherwise one `ncclCommInitRankConfig` per device in a group.

Operations on several of these communicators have to be issued in an [`NCCL.group`](@ref),
or the first one blocks waiting for the others. Destroy the communicators together, with
`NCCL.destroy(comms)`, or use the do-block form, which destroys them when `f` returns, and
aborts them if it throws (see [`NCCL.Communicator`](@ref)).

# Examples
```julia
NCCL.Communicators(CUDACore.devices()) do comms
    NCCL.group() do
        for (comm, buf) in zip(comms, bufs)
            NCCL.Allreduce!(buf, +, comm)
        end
    end
end
```

# External links
- [`ncclCommInitAll`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitall)
- [`ncclCommInitRankConfig`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrankconfig)
"""
function Communicators(devices; config::Union{Config,Nothing}=nothing)
    devs = CuDevice[dev isa CuDevice ? dev : CuDevice(dev) for dev in devices]
    isempty(devs) && throw(ArgumentError("no devices given"))
    allunique(map(deviceid, devs)) || throw(ArgumentError("devices must be distinct"))
    check_blocking()
    ndev = length(devs)
    handles = fill(ncclComm_t(C_NULL), ndev)
    GC.@preserve config handles try
        if config === nothing
            check_group_owner()
            ncclCommInitAll(handles, ndev, Cint[deviceid(dev) for dev in devs])
        else
            id = UniqueID()
            raw_config = Ref(ncclConfig_t(config))
            group() do
                for (i, dev) in enumerate(devs)
                    CUDACore.device!(dev) do
                        ncclCommInitRankConfig(pointer(handles, i), ndev, id, i-1, raw_config)
                    end
                end
            end
        end
    catch
        abort_failed(handles)
        rethrow()
    end
    return [Communicator(handles[i], devs[i], i-1, ndev, config) for i in 1:ndev]
end

function Communicators(f, devices; kwargs...)
    with_communicators(f, Communicators(devices; kwargs...))
end


## properties

"""
    NCCL.device(comm::Communicator) :: CuDevice

The device of the communicator.
"""
device(comm::Communicator) = comm.device

"""
    NCCL.nranks(comm::Communicator) :: Int

The number of ranks in the communicator.
"""
nranks(comm::Communicator) = comm.nranks

"""
    NCCL.rank(comm::Communicator) :: Int

The 0-based index of this rank in the communicator.
"""
rank(comm::Communicator) = comm.rank

function Base.show(io::IO, comm::Communicator)
    print(io, "NCCL.Communicator(rank ", comm.rank, " of ", comm.nranks, " on ", comm.device)
    isopen(comm) || print(io, ", closed")
    print(io, ")")
end
