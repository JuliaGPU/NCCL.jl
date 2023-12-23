function Send(sendbuf, comm::Communicator; dest::Integer, stream::CuStream=default_device_stream(comm))
    count = length(sendbuf)
    datatype = ncclDataType_t(eltype(sendbuf))
    ncclSend(sendbuf, count, datatype, dest, comm, stream)
    return nothing
end
function Recv!(recvbuf, comm::Communicator; source::Integer, stream::CuStream=default_device_stream(comm))
    count = length(recvbuf)
    datatype = ncclDataType_t(eltype(recvbuf))
    ncclRecv(recvbuf, count, datatype, source, comm, stream)
    return recvbuf.data
end
