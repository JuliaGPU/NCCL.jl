function Send(sendbuf, peer::Integer, comm::Communicator; stream::CuStream=CUDA.stream())
    count = length(sendbuf)
    datatype = ncclDataType(eltype(sendbuf))
    ncclSend(sendbuf, count, datatype, peer, comm, stream)
    return nothing
end
function Recv!(recvbuf, peer::Integer, comm::Communicator; stream::CuStream=CUDA.stream())
    count = length(recvbuf)
    datatype = ncclDataType(eltype(recvbuf))
    ncclRecv(recvbuf, count, datatype, peer, comm, stream)
    return recvbuf.data
end
