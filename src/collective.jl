"""
    NCCL.Allreduce!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::CuStream=CUDA.stream())
    NCCL.Allreduce!(
        sendrecvbuf, op, comm::Communicator;
        stream::CuStream=CUDA.stream())

Reduce arrays using `op` (one of `+`, `*`, `min`, `max`, or `NCCL.avg`)

# External links
- [`ncclAllReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce)
"""
function Allreduce!(sendbuf, recvbuf, op, comm::Communicator; stream::CuStream=CUDA.stream())
    count = length(recvbuf)
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclAllReduce(sendbuf, recvbuf, count, data_type, _op, comm.handle, stream)
    return recvbuf
end
Allreduce!(sendrecvbuf, op, comm::Communicator; stream::CuStream=CUDA.stream() ) =
    Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)

function Broadcast!(sendbuf, recvbuf, comm::Communicator;  root::Integer=0, stream::CuStream=CUDA.stream())
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    ncclBroadcast(sendbuf, recvbuf, count, data_type, root, comm.handle, stream)
    return recvbuf
end
Broadcast!(sendrecvbuf, comm::Communicator;  root::Integer=0, stream::CuStream=CUDA.stream()) =
    Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)

function Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer=0, stream::CuStream=CUDA.stream())
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    _op = ncclRedOp_t(op)
    ncclReduce(sendbuf, recvbuf, count, data_type, _op, root, comm.handle, stream)
    return recvbuf
end
Reduce!(sendrecvbuf, op, comm::Communicator; root::Integer=0, stream::CuStream=CUDA.stream()) =
    Reduce!(sendrecvbuf, sendrecvbuf, op, comm; root, stream)

function Allgather!(sendbuf, recvbuf, comm::Communicator; stream::CuStream=CUDA.stream())
    data_type = ncclDataType_t(eltype(recvbuf))
    sendcount = length(sendbuf)
    ncclAllGather(sendbuf, recvbuf, sendcount, data_type, comm.handle, stream)
    return recvbuf
end

function ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator; stream::CuStream=CUDA.stream() )
    recvcount = length(recvbuf)
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclReduceScatter(sendbuf, recvbuf, recvcount, data_type, _op, comm.handle, stream)
    return recvbuf
end
