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

function destroy(comm::Communicator)
    if comm.handle != C_NULL
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
comm = Communicator(length(CUDA.devices()), id, myid())
# this blocks until all other ranks have connected
```

# External links
- [`ncclCommInitRank`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclCommInitRank)
"""
function Communicator(nranks::Integer, rank::Integer;
                      unique_id::UniqueID=UniqueID())
    0 <= rank < nranks || throw(ArgumentError("rank must be in [0, nranks)"))
    handle_ref = Ref{ncclComm_t}(C_NULL)
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
comms = NCCL.Communicators(CUDA.devices())
```

# External links
- [`ncclCommInitAll`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitall)
"""
function Communicators(deviceids::Vector{Cint})
    ndev    = length(deviceids)
    comms   = Vector{ncclComm_t}(undef, ndev)
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
