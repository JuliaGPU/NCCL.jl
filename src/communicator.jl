# Communicator
export Communicator

const UniqueID = LibNCCL.ncclUniqueId

function UniqueID()
    r = Ref{UniqueID}()
    ncclGetUniqueId(r)
    return r[]
end

mutable struct Communicator
    handle::ncclComm_t
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

function destroy(comm::Communicator)
    if comm.handle != C_NULL
        check_group_owner()
        ncclCommDestroy(comm)
        comm.handle = ncclComm_t(C_NULL)
    end
    return nothing
end
Base.unsafe_convert(::Type{LibNCCL.ncclComm_t}, comm::Communicator) = comm.handle

"""
    NCCL.Communicator(nranks, rank; [unique_id]) :: Communicator

Create a single communicator for use in a multi-threaded or multi-process
environment. `nranks` is the number of ranks in the communicator, and `rank`
is the 0-based index of the current rank. `unique_id` is an optional unique
identifier for the communicator.

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
    c = Communicator(handle_ref[])
    return finalizer(destroy, c)
end

"""
    NCCL.Communicators(devices) :: Vector{Communicator}

Construct and initialize a clique of NCCL Communicators over the devices
on a single host.

`devices` can either be a collection of identifiers, or `CuDevice`s.

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
    return map(comms) do ch
        c = Communicator(ch)
        finalizer(destroy, c)
    end
end
function Communicators(deviceids::AbstractVector{<:Integer})
    Communicators(Cint[d for d in deviceids])
end
function Communicators(devices)
    Communicators(Cint[deviceid(d) for d in devices])
end

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
    NCCL.abort(comm::Communicator)

Frees resources that are allocated to `comm`. Will abort any
uncompleted operations before destroying the communicator.

# External links
- [`ncclCommAbort`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcommabort)
"""
function abort(comm::Communicator)
    check_group_owner()
    ncclCommAbort(comm)
    return
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
