"""
    NCCL.Send(
        sendbuf, comm::Communicator;
        dest::Integer,
        stream::Union{CuStream,Nothing} = nothing)
    )

Send data from `sendbuf` to rank `dest`. A matching [`Recv!`](@ref) must also be
called.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `sendbuf` must not be used until then.

# External links
- [`ncclSend`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend)
"""
function Send(sendbuf, comm::Communicator; dest::Integer,
              stream::Union{CuStream,Nothing}=nothing)
    count = length(sendbuf)
    datatype = ncclDataType_t(eltype(sendbuf))
    dest = Cint(dest)
    submit(comm, stream, sendbuf) do sendptr, comm_handle, stream_handle
        ncclSend(sendptr, count, datatype, dest, comm_handle, stream_handle)
    end
    return nothing
end

"""
    NCCL.Recv!(
        recvbuf, comm::Communicator;
        source::Integer,
        stream::Union{CuStream,Nothing} = nothing)
    )

Write the data from a matching [`Send`](@ref) on rank `source` into `recvbuf`.

The operation is enqueued on `stream`, which defaults to the task's stream on the device of
`comm`, and is ordered like other work on that stream. Inside an [`NCCL.group`](@ref), it is
only enqueued when the group ends, and `recvbuf` must not be used until then.

# External links
- [`ncclRecv`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv)

"""
function Recv!(recvbuf, comm::Communicator; source::Integer,
               stream::Union{CuStream,Nothing}=nothing)
    count = length(recvbuf)
    datatype = ncclDataType_t(eltype(recvbuf))
    source = Cint(source)
    submit(comm, stream, recvbuf) do recvptr, comm_handle, stream_handle
        ncclRecv(recvptr, count, datatype, source, comm_handle, stream_handle)
    end
    return recvbuf.data
end
