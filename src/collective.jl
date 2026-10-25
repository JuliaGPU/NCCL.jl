"""
    NCCL.Allreduce!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::Union{CuStream,Nothing} = nothing)

Reduce array `sendbuf` using `op` (one of `+`, `*`, `min`, `max`,
or [`NCCL.avg`](@ref)), writing the result to `recvbuf` to all ranks.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `sendbuf` and `recvbuf` must not be used until then.

# External links
- [`ncclAllReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce)
"""
function Allreduce!(sendbuf, recvbuf, op, comm::Communicator;
                    stream::Union{CuStream,Nothing}=nothing)
    count = length(recvbuf)
    @assert length(sendbuf) == count
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        ncclAllReduce(sendptr, recvptr, count, data_type, _op, comm_handle, stream_handle)
    end
    return recvbuf
end

"""
    NCCL.Allreduce!(
        sendrecvbuf, op, comm::Communicator;
        stream::Union{CuStream,Nothing} = nothing)


Reduce the array `sendrecvbuf` using `op` (one of `+`, `*`, `min`, `max`,
or `[`NCCL.avg`](@ref)`), writing the result inplace to all ranks.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `sendrecvbuf` must not be used until then.
"""
function Allreduce!(sendrecvbuf, op, comm::Communicator;
                    stream::Union{CuStream,Nothing}=nothing)
    Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)
end

"""
    NCCL.Broadcast!(
        sendbuf, recvbuf, comm::Communicator;
        root = 0,
        stream::Union{CuStream,Nothing} = nothing)

Copies array the `sendbuf` on rank `root` to `recvbuf` on all ranks.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `sendbuf` and `recvbuf` must not be used until then.

# External links
- [`ncclBroadcast`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast)
"""
function Broadcast!(sendbuf, recvbuf, comm::Communicator; root::Integer=0,
                    stream::Union{CuStream,Nothing}=nothing)
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    root = Cint(root)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        ncclBroadcast(sendptr, recvptr, count, data_type, root, comm_handle, stream_handle)
    end
    return recvbuf
end
function Broadcast!(sendrecvbuf, comm::Communicator;  root::Integer=0,
                    stream::Union{CuStream,Nothing}=nothing)
    Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)
end


"""
    NCCL.Reduce!(
        sendbuf, recvbuf, comm::Communicator;
        root = 0,
        stream::Union{CuStream,Nothing} = nothing)

Reduce the array `sendrecvbuf` using `op` (one of `+`, `*`, `min`, `max`,
or `[`NCCL.avg`](@ref)`), writing the result to `recvbuf` on rank `root`.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `sendbuf` and `recvbuf` must not be used until then.

# External links
- [`ncclReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreduce)
"""
function Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer=0,
                 stream::Union{CuStream,Nothing}=nothing)
    data_type = ncclDataType_t(eltype(recvbuf))
    count = length(recvbuf)
    _op = ncclRedOp_t(op)
    root = Cint(root)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        ncclReduce(sendptr, recvptr, count, data_type, _op, root, comm_handle, stream_handle)
    end
    return recvbuf
end
function Reduce!(sendrecvbuf, op, comm::Communicator; root::Integer=0,
                 stream::Union{CuStream,Nothing}=nothing)
    Reduce!(sendrecvbuf, sendrecvbuf, op, comm; root, stream)
end

"""
    NCCL.Allgather!(
        sendbuf, recvbuf, comm::Communicator;
        stream::Union{CuStream,Nothing} = nothing)
    )

Concatenate `sendbuf` from each rank into `recvbuf` on all ranks.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `sendbuf` and `recvbuf` must not be used until then.

# External links
- [`ncclAllGather`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather)
"""
function Allgather!(sendbuf, recvbuf, comm::Communicator;
                    stream::Union{CuStream,Nothing}=nothing)
    data_type = ncclDataType_t(eltype(recvbuf))
    sendcount = length(sendbuf)
    @assert length(recvbuf) == sendcount * nranks(comm)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        ncclAllGather(sendptr, recvptr, sendcount, data_type, comm_handle, stream_handle)
    end
    return recvbuf
end

"""
    NCCL.ReduceScatter!(
        sendbuf, recvbuf, op, comm::Communicator;
        stream::Union{CuStream,Nothing} = nothing)
    )

Reduce `sendbuf` from each rank using `op`, and leave the reduced result
scattered over the devices such that `recvbuf` on each rank will contain the
`i`th block of the result.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `sendbuf` and `recvbuf` must not be used until then.

# External links
- [`ncclReduceScatter`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter)
"""
function ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator;
                        stream::Union{CuStream,Nothing}=nothing)
    recvcount = length(recvbuf)
    @assert length(sendbuf) == recvcount * nranks(comm)
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        ncclReduceScatter(sendptr, recvptr, recvcount, data_type, _op, comm_handle,
                          stream_handle)
    end
    return recvbuf
end
