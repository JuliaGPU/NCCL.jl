# Communicator


const NCCL_UNIQUE_ID_BYTES = 128
const ncclUniqueId_t = NTuple{NCCL_UNIQUE_ID_BYTES, Cchar}

struct UniqueID
    internal::ncclUniqueId_t

    function UniqueID()
        buf = zeros(Cchar, NCCL_UNIQUE_ID_BYTES)
        @apicall(:ncclGetUniqueId, (Ptr{Cchar},), buf)
        new(Tuple(buf))
    end
end

# FIXME: why not unsafe_convert? same with CuEvent vs CuDevice
Base.convert(::Type{ncclUniqueId_t}, id::UniqueID) = id.internal


const ncclComm_t = Ptr{Cvoid}

struct Communicator
    handle::ncclComm_t

    function Communicator(nranks, comm_id, rank)
        handle_ref = Ref{ncclComm_t}()
        @apicall(:ncclCommInitRank, (Ptr{ncclComm_t}, Cint, ncclUniqueId_t, Cint), 
                 handle_ref,nranks, comm_id, rank)

        new(handle_ref[])
    end
end