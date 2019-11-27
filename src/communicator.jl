# Communicator

export UniqueID, Communicator, rank

import CUDAdrv: device

const NCCL_UNIQUE_ID_BYTES = 128
const ncclUniqueId_t = NTuple{NCCL_UNIQUE_ID_BYTES, Cchar}

struct UniqueID
    internal::ncclUniqueId_t

    function UniqueID()
        buf = zeros(Cchar, NCCL_UNIQUE_ID_BYTES)
        ncclGetUniqueId(buf)
        new(Tuple(buf))
    end
end

Base.convert(::Type{ncclUniqueId_t}, id::UniqueID) = id.internal


const ncclComm_t = Ptr{Cvoid}

struct Communicator
    handle::ncclComm_t
end


# creates a new communicator (multi thread/process version)
"""
   Communicator(nranks, uid, rank)

Creates a new Communicator (multi thread/process version)
`rank` must be between `0` and `nranks-1` and unique within a communicator
clique. Each rank is associated to a CUDA device which has to be set before
calling `Communicator`. Implicitly synchroniszed with other ranks so it must
be called by different threads/processes or used within `group`.
"""
function Communicator(nranks, comm_id, rank)
    handle_ref = Ref{ncclComm_t}(C_NULL)
    ncclCommInitRank(handle_ref, nranks, comm_id.internal, rank)
    c = Communicator(handle_ref[])
    finalizer(c) do x
        ncclCommDestroy(x.handle)
    end
    return c
end

# creates a clique of communicators (single process version)
function Communicator(devices::Union{CUDAdrv.DeviceSet,AbstractVector})
    ndev    = length(devices)
    comms   = Vector{ncclComm_t}(undef, ndev)
    devlist = Cint.([i-1 for i in 1:ndev])
    ncclCommInitAll(comms, ndev, devlist)
    cs = Communicator.(comms)
    finalizer(cs) do xs
        for x in xs
            ncclCommDestroy(x.handle)
        end
    end
    return cs
end

function device(comm::Communicator)
    dev_ref = Ref{Cint}(C_NULL)
    ncclCommCuDevice(comm.handle, dev_ref)
    return dev_ref[]
end

function Base.size(comm::Communicator)
    size_ref = Ref{Cint}(C_NULL)
    ncclCommCount(comm.handle, size_ref)
    return size_ref[]
end

function rank(comm::Communicator)
    rank_ref = Ref{Cint}(C_NULL)
    ncclCommUserRank(comm.handle, rank_ref)
    return rank_ref[]
end

function abort(comm::Communicator)
    ncclCommAbort(comm.handle)
end

function getError(comm::Communicator)
    ref = Ref{ncclResult_t}()
    ncclCommGetAsyncError(comm.handle, ref)
    return NCCLError(ref[])
end
