"""
    NCCL.Allreduce!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::CuStream=default_device_stream(comm))

Reduce array `sendbuf` using `op` (one of `+`, `*`, `min`, `max`,
or [`NCCL.avg`](@ref)), writing the result to `recvbuf` to all ranks.

# External links
- [`ncclAllReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce)
"""
function Allreduce!(sendbuf, recvbuf, op, comm::Communicator;
                    stream::CuStream=default_device_stream(comm))
    count = length(recvbuf)
    @assert length(sendbuf) == count
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclAllReduce(sendbuf, recvbuf, count, data_type, _op, comm, stream)
    return recvbuf
end

"""
    NCCL.Allreduce!(
        sendrecvbuf, op, comm::Communicator;
        stream::CuStream = default_device_stream(comm))


Reduce the array `sendrecvbuf` using `op` (one of `+`, `*`, `min`, `max`,
or `[`NCCL.avg`](@ref)`), writing the result inplace to all ranks.
"""
function Allreduce!(sendrecvbuf, op, comm::Communicator;
                    stream::CuStream=default_device_stream(comm))
    Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)
end

"""
    NCCL.Broadcast!(
        sendbuf, recvbuf, comm::Communicator;
        root = 0,
        stream::CuStream = default_device_stream(comm))

Copies array the `sendbuf` on rank `root` to `recvbuf` on all ranks.

# External links
- [`ncclBroadcast`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast)
"""
function Broadcast!(sendbuf, recvbuf, comm::Communicator; root::Integer=0,
                    stream::CuStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    ncclBroadcast(sendbuf, recvbuf, count, data_type, root, comm, stream)
    return recvbuf
end
function Broadcast!(sendrecvbuf, comm::Communicator;  root::Integer=0,
                    stream::CuStream=default_device_stream(comm))
    Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)
end


"""
    NCCL.Reduce!(
        sendbuf, recvbuf, comm::Communicator;
        root = 0,
        stream::CuStream = default_device_stream(comm))

Reduce the array `sendrecvbuf` using `op` (one of `+`, `*`, `min`, `max`,
or `[`NCCL.avg`](@ref)`), writing the result to `recvbuf` on rank `root`.

# External links
- [`ncclReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreduce)
"""
function Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer=0,
                 stream::CuStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    _op = ncclRedOp_t(op)
    ncclReduce(sendbuf, recvbuf, count, data_type, _op, root, comm, stream)
    return recvbuf
end
function Reduce!(sendrecvbuf, op, comm::Communicator; root::Integer=0,
                 stream::CuStream=default_device_stream(comm))
    Reduce!(sendrecvbuf, sendrecvbuf, op, comm; root, stream)
end

"""
    NCCL.Allgather!(
        sendbuf, recvbuf, comm::Communicator;
        stream::CuStream = default_device_stream(comm))
    )

Concatenate `sendbuf` from each rank into `recvbuf` on all ranks.

# External links
- [`ncclAllGather`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather)
"""
function Allgather!(sendbuf, recvbuf, comm::Communicator;
                    stream::CuStream=default_device_stream(comm))
    data_type = ncclDataType_t(eltype(recvbuf))
    sendcount = length(sendbuf)
    @assert length(recvbuf) == sendcount * size(comm)
    ncclAllGather(sendbuf, recvbuf, sendcount, data_type, comm, stream)
    return recvbuf
end

"""
    NCCL.ReduceScatter!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::CuStream = default_device_stream(comm))
    )

Reduce `sendbuf` from each rank using `op`, and leave the reduced result
scattered over the devices such that `recvbuf` on each rank will contain the
`i`th block of the result.

# External links
- [`ncclReduceScatter`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter)
"""
function ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator;
                        stream::CuStream=default_device_stream(comm))
    recvcount = length(recvbuf)
    @assert length(sendbuf) == recvcount * size(comm)
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    ncclReduceScatter(sendbuf, recvbuf, recvcount, data_type, _op, comm, stream)
    return recvbuf
end
