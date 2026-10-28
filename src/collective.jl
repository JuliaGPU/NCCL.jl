"""
    NCCL.Allreduce!(sendbuf, recvbuf, op, comm::Communicator;
                    stream::Union{CuStream,Nothing} = nothing)
    NCCL.Allreduce!(sendrecvbuf, op, comm::Communicator; stream)

Reduce `sendbuf` over all ranks using `op` (one of `+`, `*`, `min`, `max`, or
[`NCCL.avg`](@ref)), and write the result to `recvbuf` on all ranks. Returns `recvbuf`.

`sendbuf` and `recvbuf` must have the same element type and length. With a single buffer
`sendrecvbuf`, the operation is performed in place.

The buffers must be stored contiguously in memory of the device of `comm`: a `CuArray`, or
a contiguous view or reshape of one. The operation is enqueued on `stream`, which defaults to
the task's stream on that device, and is ordered like other work on that stream. Inside an
[`NCCL.group`](@ref), it is only enqueued when the group ends, and its buffers must not be
used until then.

# External links
- [`ncclAllReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce)
"""
function Allreduce!(sendbuf, recvbuf, op, comm::Communicator;
                    stream::Union{CuStream,Nothing}=nothing)
    check_buffer(comm, sendbuf, "sendbuf")
    check_buffer(comm, recvbuf, "recvbuf")
    check_eltypes(sendbuf, recvbuf)
    count = length(recvbuf)
    check_length(sendbuf, count, "sendbuf")
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        check_overlap(sendptr, sendbuf, recvptr, recvbuf)
        ncclAllReduce(sendptr, recvptr, count, data_type, _op, comm_handle, stream_handle)
    end
    return recvbuf
end

function Allreduce!(sendrecvbuf, op, comm::Communicator;
                    stream::Union{CuStream,Nothing}=nothing)
    Allreduce!(sendrecvbuf, sendrecvbuf, op, comm; stream)
end

"""
    NCCL.Broadcast!(sendbuf, recvbuf, comm::Communicator; root::Integer = 0,
                    stream::Union{CuStream,Nothing} = nothing)
    NCCL.Broadcast!(sendrecvbuf, comm::Communicator; root, stream)

Copy `sendbuf` on rank `root` to `recvbuf` on all ranks. Returns `recvbuf`.

`sendbuf` and `recvbuf` must have the same element type and length; `sendbuf` is only read
on the root. With a single buffer `sendrecvbuf`, the operation is performed in place.

The buffers must be stored contiguously in memory of the device of `comm`: a `CuArray`, or
a contiguous view or reshape of one. The operation is enqueued on `stream`, which defaults to
the task's stream on that device, and is ordered like other work on that stream. Inside an
[`NCCL.group`](@ref), it is only enqueued when the group ends, and its buffers must not be
used until then.

# External links
- [`ncclBroadcast`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast)
"""
function Broadcast!(sendbuf, recvbuf, comm::Communicator; root::Integer=0,
                    stream::Union{CuStream,Nothing}=nothing)
    check_buffer(comm, sendbuf, "sendbuf")
    check_buffer(comm, recvbuf, "recvbuf")
    check_eltypes(sendbuf, recvbuf)
    count = length(recvbuf)
    check_length(sendbuf, count, "sendbuf")
    data_type = ncclDataType_t(eltype(recvbuf))
    root = Cint(root)
    check_peer(comm, root, "root")
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        check_overlap(sendptr, sendbuf, recvptr, recvbuf)
        ncclBroadcast(sendptr, recvptr, count, data_type, root, comm_handle, stream_handle)
    end
    return recvbuf
end

function Broadcast!(sendrecvbuf, comm::Communicator; root::Integer=0,
                    stream::Union{CuStream,Nothing}=nothing)
    Broadcast!(sendrecvbuf, sendrecvbuf, comm; root, stream)
end

"""
    NCCL.Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer = 0,
                 stream::Union{CuStream,Nothing} = nothing)
    NCCL.Reduce!(sendrecvbuf, op, comm::Communicator; root, stream)

Reduce `sendbuf` over all ranks using `op` (one of `+`, `*`, `min`, `max`, or
[`NCCL.avg`](@ref)), and write the result to `recvbuf` on rank `root`. Returns `recvbuf`.

`sendbuf` and `recvbuf` must have the same element type and length; `recvbuf` is only
written on the root. With a single buffer `sendrecvbuf`, the operation is performed in
place.

The buffers must be stored contiguously in memory of the device of `comm`: a `CuArray`, or
a contiguous view or reshape of one. The operation is enqueued on `stream`, which defaults to
the task's stream on that device, and is ordered like other work on that stream. Inside an
[`NCCL.group`](@ref), it is only enqueued when the group ends, and its buffers must not be
used until then.

# External links
- [`ncclReduce`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreduce)
"""
function Reduce!(sendbuf, recvbuf, op, comm::Communicator; root::Integer=0,
                 stream::Union{CuStream,Nothing}=nothing)
    check_buffer(comm, sendbuf, "sendbuf")
    check_buffer(comm, recvbuf, "recvbuf")
    check_eltypes(sendbuf, recvbuf)
    count = length(recvbuf)
    check_length(sendbuf, count, "sendbuf")
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    root = Cint(root)
    check_peer(comm, root, "root")
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        check_overlap(sendptr, sendbuf, recvptr, recvbuf)
        ncclReduce(sendptr, recvptr, count, data_type, _op, root, comm_handle, stream_handle)
    end
    return recvbuf
end

function Reduce!(sendrecvbuf, op, comm::Communicator; root::Integer=0,
                 stream::Union{CuStream,Nothing}=nothing)
    Reduce!(sendrecvbuf, sendrecvbuf, op, comm; root, stream)
end

"""
    NCCL.Allgather!(sendbuf, recvbuf, comm::Communicator;
                    stream::Union{CuStream,Nothing} = nothing)

Concatenate `sendbuf` from each rank, in rank order, into `recvbuf` on all ranks. Returns
`recvbuf`.

`sendbuf` and `recvbuf` must have the same element type, and `recvbuf` must have
`nranks(comm)` times as many elements as `sendbuf`. The operation is performed in place
when `sendbuf` is the part of `recvbuf` that receives the data of this rank.

The buffers must be stored contiguously in memory of the device of `comm`: a `CuArray`, or
a contiguous view or reshape of one. The operation is enqueued on `stream`, which defaults to
the task's stream on that device, and is ordered like other work on that stream. Inside an
[`NCCL.group`](@ref), it is only enqueued when the group ends, and its buffers must not be
used until then.

# External links
- [`ncclAllGather`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather)
"""
function Allgather!(sendbuf, recvbuf, comm::Communicator;
                    stream::Union{CuStream,Nothing}=nothing)
    check_buffer(comm, sendbuf, "sendbuf")
    check_buffer(comm, recvbuf, "recvbuf")
    check_eltypes(sendbuf, recvbuf)
    sendcount = length(sendbuf)
    check_length(recvbuf, sendcount * comm.nranks, "recvbuf")
    data_type = ncclDataType_t(eltype(recvbuf))
    inplace_offset = comm.rank * sendcount * Base.elsize(sendbuf)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        check_overlap(sendptr, sendbuf, recvptr, recvbuf; inplace_offset,
                      inplace="sendbuf must be the part of recvbuf for this rank")
        ncclAllGather(sendptr, recvptr, sendcount, data_type, comm_handle, stream_handle)
    end
    return recvbuf
end

"""
    NCCL.ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator;
                        stream::Union{CuStream,Nothing} = nothing)

Reduce `sendbuf` over all ranks using `op` (one of `+`, `*`, `min`, `max`, or
[`NCCL.avg`](@ref)), and scatter the result over the ranks: `recvbuf` on rank `i` receives
the `i`-th block of the result (counting from 0). Returns `recvbuf`.

`sendbuf` and `recvbuf` must have the same element type, and `sendbuf` must have
`nranks(comm)` times as many elements as `recvbuf`. The operation is performed in place
when `recvbuf` is the block of `sendbuf` for this rank.

The buffers must be stored contiguously in memory of the device of `comm`: a `CuArray`, or
a contiguous view or reshape of one. The operation is enqueued on `stream`, which defaults to
the task's stream on that device, and is ordered like other work on that stream. Inside an
[`NCCL.group`](@ref), it is only enqueued when the group ends, and its buffers must not be
used until then.

# External links
- [`ncclReduceScatter`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreducescatter)
"""
function ReduceScatter!(sendbuf, recvbuf, op, comm::Communicator;
                        stream::Union{CuStream,Nothing}=nothing)
    check_buffer(comm, sendbuf, "sendbuf")
    check_buffer(comm, recvbuf, "recvbuf")
    check_eltypes(sendbuf, recvbuf)
    recvcount = length(recvbuf)
    check_length(sendbuf, recvcount * comm.nranks, "sendbuf")
    data_type = ncclDataType_t(eltype(recvbuf))
    _op = ncclRedOp_t(op)
    inplace_offset = -comm.rank * recvcount * Base.elsize(recvbuf)
    submit(comm, stream, sendbuf, recvbuf) do sendptr, recvptr, comm_handle, stream_handle
        check_overlap(sendptr, sendbuf, recvptr, recvbuf; inplace_offset,
                      inplace="recvbuf must be the block of sendbuf for this rank")
        ncclReduceScatter(sendptr, recvptr, recvcount, data_type, _op, comm_handle,
                          stream_handle)
    end
    return recvbuf
end
