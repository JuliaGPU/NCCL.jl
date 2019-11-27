# Collective calls

function Allreduce!(sendbuf, recvbuf, count::Integer, op, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    @apicall(:ncclAllReduce, (CuPtr{Cvoid}, CuPtr{Cvoid}, Cint, ncclDataType_t, ncclRedOp_t, ncclComm_t, CUstream),
             sendbuf, recvbuf, count, data_type, op, comm.handle, stream)
    return recvbuf
end

function Broadcast!(sendbuf, recvbuf, count::Integer, root::Int, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    @apicall(:ncclBroadcast, (CuPtr{Cvoid}, CuPtr{Cvoid}, Cint, ncclDataType_t, Cint, ncclComm_t, CUstream),
             sendbuf, recvbuf, count, data_type, root, comm.handle, stream)
    return recvbuf
end

function Reduce!(sendbuf, recvbuf, count::Integer, op, root::Int, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    @apicall(:ncclReduce, (CuPtr{Cvoid}, CuPtr{Cvoid}, Cint, ncclDataType_t, ncclRedOp_t, Cint, ncclComm_t, CUstream),
             sendbuf, recvbuf, count, data_type, op, root, comm.handle, stream)
    return recvbuf
end

function Allgather!(sendbuf, recvbuf, sendcount::Integer, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    @apicall(:ncclAllGather, (CuPtr{Cvoid}, CuPtr{Cvoid}, Cint, ncclDataType_t, ncclComm_t, CUstream),
             sendbuf, recvbuf, sendcount, data_type, comm.handle, stream)
    return recvbuf
end

function ReduceScatter!(sendbuf, recvbuf, recvcount::Integer, op, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    @apicall(:ncclReduceScatter, (CuPtr{Cvoid}, CuPtr{Cvoid}, Cint, ncclDataType_t, ncclRedOp_t, ncclComm_t, CUstream),
             sendbuf, recvbuf, recvcount, data_type, op, comm.handle, stream)
    return recvbuf
end