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
    count = length(sendbuf)
    datatype = ncclDataType_t(eltype(sendbuf))
    comm_device = device(comm)
    _check_buffer_devices(comm_device, sendbuf)
    CUDA.device!(comm_device) do
        ncclSend(sendbuf, count, datatype, dest, comm, stream)
    end
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
    count = length(recvbuf)
    datatype = ncclDataType_t(eltype(recvbuf))
    comm_device = device(comm)
    _check_buffer_devices(comm_device, recvbuf)
    CUDA.device!(comm_device) do
        ncclRecv(recvbuf, count, datatype, source, comm, stream)
    end
    return recvbuf.data
end
