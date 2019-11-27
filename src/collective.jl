# Collective calls

export Allreduce!, Broadcast!, Reduce!, Allgather!, ReduceScatter!

function allReduce!(::Op, sendbuf, recvbuf, comm::Communicator; stream=CUDAdrv.CuDefaultStream()) where Op
    op = ncclReductionOp(Op)
    @assert size(sendbuf) == size(recvbuf)
    Allreduce!(sendbuf, recvbuf, length(sendbuf), op, comm, stream=stream)
end

function Allreduce!(sendbuf, recvbuf, count::Integer, op, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    ncclAllReduce(sendbuf, recvbuf, count, data_type, op, comm.handle, stream)
    return recvbuf
end

function Broadcast!(sendbuf, recvbuf, count::Integer, root::Int, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    ncclBroadcast(sendbuf, recvbuf, count, data_type, root, comm.handle, stream)
    return recvbuf
end

function Reduce!(sendbuf, recvbuf, count::Integer, op, root::Int, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    ncclReduce(sendbuf, recvbuf, count, data_type, op, root, comm.handle, stream)
    return recvbuf
end

function Allgather!(sendbuf, recvbuf, sendcount::Integer, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    ncclAllGather(sendbuf, recvbuf, sendcount, data_type, comm.handle, stream)
    return recvbuf
end

function ReduceScatter!(sendbuf, recvbuf, recvcount::Integer, op, comm::Communicator; stream::CuStream=CuDefaultStream() )
    data_type = ncclDataType(eltype(recvbuf))
    ncclReduceScatter(sendbuf, recvbuf, recvcount, data_type, op, comm.handle, stream)
    return recvbuf
end
