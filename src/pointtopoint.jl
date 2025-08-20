"""
    NCCL.Send(
        sendbuf, comm::Communicator;
        dest::Integer,
        stream::CuStream = default_device_stream(comm))
    )

Send data from `sendbuf` to rank `dest`. A matching [`Recv!`](@ref) must also be
called.

# External links
- [`ncclSend`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend)
"""
function Send(sendbuf, comm::Communicator; dest::Integer,
              stream::CuStream=default_device_stream(comm))
    a_count = count(sendbuf)
    datatype = ncclDataType_t(eltype(sendbuf))
    ncclSend(sendbuf, a_count, datatype, dest, comm, stream)
    return nothing
end

"""
    NCCL.Recv!(
        recvbuf, comm::Communicator;
        source::Integer,
        stream::CuStream = default_device_stream(comm))
    )

Write the data from a matching [`Send`](@ref) on rank `source` into `recvbuf`.

# External links
- [`ncclRecv`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv)

"""
function Recv!(recvbuf, comm::Communicator; source::Integer,
               stream::CuStream=default_device_stream(comm))
    a_count = count(recvbuf)
    datatype = ncclDataType_t(eltype(recvbuf))
    ncclRecv(recvbuf, a_count, datatype, source, comm, stream)
    return recvbuf.data
end
