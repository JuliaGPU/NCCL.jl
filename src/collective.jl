"""
    NCCL.Allreduce!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::CuStream=default_device_stream(comm))
    NCCL.Allreduce!(
        sendrecvbuf, op, comm::Communicator;
        stream::CuStream=default_device_stream(comm))

Reduce arrays using `op` (one of `+`, `*`, `min`, `max`, or `NCCL.avg`)

# External links
- [`ncclAllReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce)
"""
function Allreduce!(sendbuf, recvbuf, op, comm::Communicator; stream::CuStream=default_device_stream(comm))
    count = length(recvbuf)
    @assert length(sendbuf) == count
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclAllReduce(sendbuf, recvbuf, count, data_type, _op, comm, stream)
    return recvbuf
end
Allreduce!(sendrecvbuf, op, comm::Communicator; stream::CuStream=default_device_stream(comm) ) =
    Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)

function Broadcast!(sendbuf, recvbuf, comm::Communicator; root::Integer=0, stream::CuStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    ncclBroadcast(sendbuf, recvbuf, count, data_type, root, comm, stream)
    return recvbuf
end
Broadcast!(sendrecvbuf, comm::Communicator;  root::Integer=0, stream::CuStream=default_device_stream(comm)) =
    Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)

function Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer=0, stream::CuStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    _op = ncclRedOp_t(op)
    ncclReduce(sendbuf, recvbuf, count, data_type, _op, root, comm, stream)
    return recvbuf
end
Reduce!(sendrecvbuf, op, comm::Communicator; root::Integer=0, stream::CuStream=default_device_stream(comm)) =
    Reduce!(sendrecvbuf, sendrecvbuf, op, comm; root, stream)

function Allgather!(sendbuf, recvbuf, comm::Communicator; stream::CuStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    sendcount = length(sendbuf)
    @assert length(recvbuf) == count * size(comm)
    ncclAllGather(sendbuf, recvbuf, sendcount, data_type, comm, stream)
    return recvbuf
end

function ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator; stream::CuStream=default_device_stream(comm) )
    recvcount = length(recvbuf)
    @assert length(sendbuf) == recvcount * size(comm)
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclReduceScatter(sendbuf, recvbuf, recvcount, data_type, _op, comm, stream)
    return recvbuf
end
