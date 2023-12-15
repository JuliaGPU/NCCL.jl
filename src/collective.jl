
function Allreduce!(sendbuf, recvbuf, op, comm::Communicator; stream::CuStream=CUDA.stream())
    count = length(recvbuf)
    data_type = ncclDataType(eltype(recvbuf))
    ncclAllReduce(sendbuf, recvbuf, count, data_type, op, comm.handle, stream)
    return recvbuf
end
Allreduce!(sendrecvbuf, op, comm::Communicator; stream::CuStream=CUDA.stream() ) =
    Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)

function Broadcast!(sendbuf, recvbuf, comm::Communicator;  root::Integer=0, stream::CuStream=CUDA.stream())
    data_type = ncclDataType(eltype(recvbuf))
    count = length(recvbuf)
    ncclBroadcast(sendbuf, recvbuf, count, data_type, root, comm.handle, stream)
    return recvbuf
end
Broadcast!(sendrecvbuf, comm::Communicator;  root::Integer=0, stream::CuStream=CUDA.stream()) =
    Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)

function Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer=0, stream::CuStream=CUDA.stream())
    data_type = ncclDataType(eltype(recvbuf))
    count = length(recvbuf)
    ncclReduce(sendbuf, recvbuf, count, data_type, op, root, comm.handle, stream)
    return recvbuf
end
Reduce!(sendrecvbuf, op, comm::Communicator; root::Integer=0, stream::CuStream=CUDA.stream()) =
    Reduce!(sendrecvbuf, sendrecvbuf, op, comm; root, stream)

function Allgather!(sendbuf, recvbuf, comm::Communicator; stream::CuStream=CUDA.stream())
    data_type = ncclDataType(eltype(recvbuf))
    sendcount = length(sendbuf)
    ncclAllGather(sendbuf, recvbuf, sendcount, data_type, comm.handle, stream)
    return recvbuf
end

function ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator; stream::CuStream=CUDA.stream() )
    recvcount = length(recvbuf)
    data_type = ncclDataType(eltype(recvbuf))
    ncclReduceScatter(sendbuf, recvbuf, recvcount, data_type, op, comm.handle, stream)
    return recvbuf
end
