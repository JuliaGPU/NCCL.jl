"""
    NCCL.Send(sendbuf, comm::Communicator; dest::Integer,
              stream::Union{CuStream,Nothing} = nothing)

Send `sendbuf` to rank `dest`, which must call [`NCCL.Recv!`](@ref) with a buffer of the
same element type and length.

The buffer must be stored contiguously in memory of the device of `comm`: a `CuArray`, or a
contiguous view or reshape of one. The operation is enqueued on `stream`, which defaults to
the task's stream on that device, and is ordered like other work on that stream. Inside an
[`NCCL.group`](@ref), it is only enqueued when the group ends, and its buffer must not be
used until then.

# External links
- [`ncclSend`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend)
"""
function Send(sendbuf, comm::Communicator; dest::Integer,
              stream::Union{CuStream,Nothing}=nothing)
    check_buffer(comm, sendbuf, "sendbuf")
    count = length(sendbuf)
    datatype = ncclDataType_t(eltype(sendbuf))
    dest = Cint(dest)
    check_peer(comm, dest, "dest")
    submit(comm, stream, sendbuf) do sendptr, comm_handle, stream_handle
        ncclSend(sendptr, count, datatype, dest, comm_handle, stream_handle)
    end
    return nothing
end

"""
    NCCL.Recv!(recvbuf, comm::Communicator; source::Integer,
               stream::Union{CuStream,Nothing} = nothing)

Receive the data that rank `source` sends with [`NCCL.Send`](@ref) into `recvbuf`.

The buffer must be stored contiguously in memory of the device of `comm`: a `CuArray`, or a
contiguous view or reshape of one. The operation is enqueued on `stream`, which defaults to
the task's stream on that device, and is ordered like other work on that stream. Inside an
[`NCCL.group`](@ref), it is only enqueued when the group ends, and its buffer must not be
used until then.

# External links
- [`ncclRecv`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv)
"""
function Recv!(recvbuf, comm::Communicator; source::Integer,
               stream::Union{CuStream,Nothing}=nothing)
    check_buffer(comm, recvbuf, "recvbuf")
    count = length(recvbuf)
    datatype = ncclDataType_t(eltype(recvbuf))
    source = Cint(source)
    check_peer(comm, source, "source")
    submit(comm, stream, recvbuf) do recvptr, comm_handle, stream_handle
        ncclRecv(recvptr, count, datatype, source, comm_handle, stream_handle)
    end
    return recvbuf.data
end
