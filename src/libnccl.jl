# Julia wrapper for header: nccl.h
# Automatically generated using Clang.jl


function ncclGetVersion(version)
    @check @runtime_ccall((:ncclGetVersion, libnccl), ncclResult_t,
                          (Ptr{Cint},),
                          version)
end

function pncclGetVersion(version)
    @check @runtime_ccall((:pncclGetVersion, libnccl), ncclResult_t,
                          (Ptr{Cint},),
                          version)
end

function ncclGetUniqueId(uniqueId)
    @check @runtime_ccall((:ncclGetUniqueId, libnccl), ncclResult_t,
                          (Ptr{ncclUniqueId},),
                          uniqueId)
end

function pncclGetUniqueId(uniqueId)
    @check @runtime_ccall((:pncclGetUniqueId, libnccl), ncclResult_t,
                          (Ptr{ncclUniqueId},),
                          uniqueId)
end

function ncclCommInitRank(comm, nranks, commId, rank)
    @check @runtime_ccall((:ncclCommInitRank, libnccl), ncclResult_t,
                          (Ptr{ncclComm_t}, Cint, ncclUniqueId, Cint),
                          comm, nranks, commId, rank)
end

function pncclCommInitRank(comm, nranks, commId, rank)
    @check @runtime_ccall((:pncclCommInitRank, libnccl), ncclResult_t,
                          (Ptr{ncclComm_t}, Cint, ncclUniqueId, Cint),
                          comm, nranks, commId, rank)
end

function ncclCommInitAll(comm, ndev, devlist)
    @check @runtime_ccall((:ncclCommInitAll, libnccl), ncclResult_t,
                          (Ptr{ncclComm_t}, Cint, Ptr{Cint}),
                          comm, ndev, devlist)
end

function pncclCommInitAll(comm, ndev, devlist)
    @check @runtime_ccall((:pncclCommInitAll, libnccl), ncclResult_t,
                          (Ptr{ncclComm_t}, Cint, Ptr{Cint}),
                          comm, ndev, devlist)
end

function ncclCommDestroy(comm)
    @check @runtime_ccall((:ncclCommDestroy, libnccl), ncclResult_t,
                          (ncclComm_t,),
                          comm)
end

function pncclCommDestroy(comm)
    @check @runtime_ccall((:pncclCommDestroy, libnccl), ncclResult_t,
                          (ncclComm_t,),
                          comm)
end

function ncclCommAbort(comm)
    @check @runtime_ccall((:ncclCommAbort, libnccl), ncclResult_t,
                          (ncclComm_t,),
                          comm)
end

function pncclCommAbort(comm)
    @check @runtime_ccall((:pncclCommAbort, libnccl), ncclResult_t,
                          (ncclComm_t,),
                          comm)
end

function ncclGetErrorString(result)
    @runtime_ccall((:ncclGetErrorString, libnccl), Cstring,
                   (ncclResult_t,),
                   result)
end

function pncclGetErrorString(result)
    @runtime_ccall((:pncclGetErrorString, libnccl), Cstring,
                   (ncclResult_t,),
                   result)
end

function ncclCommGetAsyncError(comm, asyncError)
    @check @runtime_ccall((:ncclCommGetAsyncError, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{ncclResult_t}),
                          comm, asyncError)
end

function pncclCommGetAsyncError(comm, asyncError)
    @check @runtime_ccall((:pncclCommGetAsyncError, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{ncclResult_t}),
                          comm, asyncError)
end

function ncclCommCount(comm, count)
    @check @runtime_ccall((:ncclCommCount, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{Cint}),
                          comm, count)
end

function pncclCommCount(comm, count)
    @check @runtime_ccall((:pncclCommCount, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{Cint}),
                          comm, count)
end

function ncclCommCuDevice(comm, device)
    @check @runtime_ccall((:ncclCommCuDevice, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{Cint}),
                          comm, device)
end

function pncclCommCuDevice(comm, device)
    @check @runtime_ccall((:pncclCommCuDevice, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{Cint}),
                          comm, device)
end

function ncclCommUserRank(comm, rank)
    @check @runtime_ccall((:ncclCommUserRank, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{Cint}),
                          comm, rank)
end

function pncclCommUserRank(comm, rank)
    @check @runtime_ccall((:pncclCommUserRank, libnccl), ncclResult_t,
                          (ncclComm_t, Ptr{Cint}),
                          comm, rank)
end

function ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    @check @runtime_ccall((:ncclReduce, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclRedOp_t, Cint, ncclComm_t, CUstream),
                          sendbuff, recvbuff, count, datatype, op, root, comm, stream)
end

function pncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    @check @runtime_ccall((:pncclReduce, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclRedOp_t, Cint, ncclComm_t, CUstream),
                          sendbuff, recvbuff, count, datatype, op, root, comm, stream)
end

function ncclBcast(buff, count, datatype, root, comm, stream)
    @check @runtime_ccall((:ncclBcast, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, Csize_t, ncclDataType_t, Cint, ncclComm_t,
                           CUstream),
                          buff, count, datatype, root, comm, stream)
end

function pncclBcast(buff, count, datatype, root, comm, stream)
    @check @runtime_ccall((:pncclBcast, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, Csize_t, ncclDataType_t, Cint, ncclComm_t,
                           CUstream),
                          buff, count, datatype, root, comm, stream)
end

function ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    @check @runtime_ccall((:ncclBroadcast, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t, Cint,
                           ncclComm_t, CUstream),
                          sendbuff, recvbuff, count, datatype, root, comm, stream)
end

function pncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    @check @runtime_ccall((:pncclBroadcast, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t, Cint,
                           ncclComm_t, CUstream),
                          sendbuff, recvbuff, count, datatype, root, comm, stream)
end

function ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    @check @runtime_ccall((:ncclAllReduce, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclRedOp_t, ncclComm_t, CUstream),
                          sendbuff, recvbuff, count, datatype, op, comm, stream)
end

function pncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    @check @runtime_ccall((:pncclAllReduce, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclRedOp_t, ncclComm_t, CUstream),
                          sendbuff, recvbuff, count, datatype, op, comm, stream)
end

function ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    @check @runtime_ccall((:ncclReduceScatter, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclRedOp_t, ncclComm_t, CUstream),
                          sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
end

function pncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    @check @runtime_ccall((:pncclReduceScatter, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclRedOp_t, ncclComm_t, CUstream),
                          sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
end

function ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    @check @runtime_ccall((:ncclAllGather, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclComm_t, CUstream),
                          sendbuff, recvbuff, sendcount, datatype, comm, stream)
end

function pncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    @check @runtime_ccall((:pncclAllGather, libnccl), ncclResult_t,
                          (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, ncclDataType_t,
                           ncclComm_t, CUstream),
                          sendbuff, recvbuff, sendcount, datatype, comm, stream)
end

function ncclGroupStart()
    @check @runtime_ccall((:ncclGroupStart, libnccl), ncclResult_t, ())
end

function ncclGroupEnd()
    @check @runtime_ccall((:ncclGroupEnd, libnccl), ncclResult_t, ())
end
