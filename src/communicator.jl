# Communicator
export Communicator, rank

const UniqueId = LibNCCL.ncclUniqueId

function UniqueId()
    r = Ref{UniqueId}()
    ncclGetUniqueId(r)
    return r[]
end

mutable struct Communicator
    handle::ncclComm_t
end

function destroy(comm::Communicator)
    ncclCommDestroy(comm.handle)
    x.handle = ncclComm_t(C_NULL)
    return nothing
end


# creates a new communicator (multi thread/process version)
function Communicator(nranks::Integer, comm_id::UniqueId, rank::Integer)
    handle_ref = Ref{ncclComm_t}(C_NULL)
    ncclCommInitRank(handle_ref, nranks, comm_id, rank)
    c = Communicator(handle_ref[])
    return finalizer(destroy, c)
end

# creates a clique of communicators (single process version)
function Communicator_all(device_ids::AbstractVector{<:Integer})
    ndev    = length(device_ids)
    comms   = Vector{ncclComm_t}(undef, ndev)
    devlist = Cint.(device_ids)
    ncclCommInitAll(comms, ndev, devlist)
    return map(comms) do ch
        c = Communicator(ch)
        finalizer(destroy, c)
    end
end
function Communicator_all(devices)
    Communicator_all([CUDA.deviceid(d) for d in devices])
end

function CUDA.deviceid(comm::Communicator)
    dev_ref = Ref{Cint}(C_NULL)
    ncclCommCuDevice(comm.handle, dev_ref)
    return Int(dev_ref[])
end

function size(comm::Communicator)
    size_ref = Ref{Cint}(C_NULL)
    ncclCommCount(comm.handle, size_ref)
    return Int(size_ref[])
end

function rank(comm::Communicator)
    rank_ref = Ref{Cint}(C_NULL)
    ncclCommUserRank(comm.handle, rank_ref)
    return Int(rank_ref[])
end

function abort(comm::Communicator)
    ncclCommAbort(comm.handle)
    return
end
