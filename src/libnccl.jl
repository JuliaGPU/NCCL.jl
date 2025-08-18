module LibNCCL

using NCCL_jll
export NCCL_jll

using CEnum

const NULL = C_NULL
const INT_MIN = typemin(Cint)

import CUDA: @checked, CuPtr, CUstream

function check(f)
    res = f()::ncclResult_t
    if res != ncclSuccess
        throw(NCCLError(res))
    end
    return
end

struct ncclConfig_v21700
    size::Cint
    magic::Cuint
    version::Cuint
    blocking::Cint
    cgaClusterSize::Cint
    minCTAs::Cint
    maxCTAs::Cint
    netName::Cstring
    splitShare::Cint
end

const ncclConfig_t = ncclConfig_v21700

mutable struct ncclComm end

const ncclComm_t = Ptr{ncclComm}

struct ncclUniqueId
    internal::NTuple{128,Cchar}
end

@cenum ncclResult_t::UInt32 begin
    ncclSuccess = 0
    ncclUnhandledCudaError = 1
    ncclSystemError = 2
    ncclInternalError = 3
    ncclInvalidArgument = 4
    ncclInvalidUsage = 5
    ncclRemoteError = 6
    ncclInProgress = 7
    ncclNumResults = 8
end

@checked function ncclMemAlloc(ptr, size)
    @ccall libnccl.ncclMemAlloc(ptr::Ptr{Ptr{Cvoid}}, size::Cint)::ncclResult_t
end

@checked function pncclMemAlloc(ptr, size)
    @ccall libnccl.pncclMemAlloc(ptr::Ptr{Ptr{Cvoid}}, size::Cint)::ncclResult_t
end

@checked function ncclMemFree(ptr)
    @ccall libnccl.ncclMemFree(ptr::CuPtr{Cvoid})::ncclResult_t
end

@checked function pncclMemFree(ptr)
    @ccall libnccl.pncclMemFree(ptr::CuPtr{Cvoid})::ncclResult_t
end

@checked function ncclGetVersion(version)
    @ccall libnccl.ncclGetVersion(version::Ptr{Cint})::ncclResult_t
end

@checked function pncclGetVersion(version)
    @ccall libnccl.pncclGetVersion(version::Ptr{Cint})::ncclResult_t
end

@checked function ncclGetUniqueId(uniqueId)
    @ccall libnccl.ncclGetUniqueId(uniqueId::Ptr{ncclUniqueId})::ncclResult_t
end

@checked function pncclGetUniqueId(uniqueId)
    @ccall libnccl.pncclGetUniqueId(uniqueId::Ptr{ncclUniqueId})::ncclResult_t
end

@checked function ncclCommInitRankConfig(comm, nranks, commId, rank, config)
    @ccall libnccl.ncclCommInitRankConfig(comm::Ptr{ncclComm_t}, nranks::Cint,
                                          commId::ncclUniqueId, rank::Cint,
                                          config::Ptr{ncclConfig_t})::ncclResult_t
end

@checked function pncclCommInitRankConfig(comm, nranks, commId, rank, config)
    @ccall libnccl.pncclCommInitRankConfig(comm::Ptr{ncclComm_t}, nranks::Cint,
                                           commId::ncclUniqueId, rank::Cint,
                                           config::Ptr{ncclConfig_t})::ncclResult_t
end

@checked function ncclCommInitRank(comm, nranks, commId, rank)
    @ccall libnccl.ncclCommInitRank(comm::Ptr{ncclComm_t}, nranks::Cint,
                                    commId::ncclUniqueId, rank::Cint)::ncclResult_t
end

@checked function pncclCommInitRank(comm, nranks, commId, rank)
    @ccall libnccl.pncclCommInitRank(comm::Ptr{ncclComm_t}, nranks::Cint,
                                     commId::ncclUniqueId, rank::Cint)::ncclResult_t
end

@checked function ncclCommInitAll(comm, ndev, devlist)
    @ccall libnccl.ncclCommInitAll(comm::Ptr{ncclComm_t}, ndev::Cint,
                                   devlist::Ptr{Cint})::ncclResult_t
end

@checked function pncclCommInitAll(comm, ndev, devlist)
    @ccall libnccl.pncclCommInitAll(comm::Ptr{ncclComm_t}, ndev::Cint,
                                    devlist::Ptr{Cint})::ncclResult_t
end

@checked function ncclCommFinalize(comm)
    @ccall libnccl.ncclCommFinalize(comm::ncclComm_t)::ncclResult_t
end

@checked function pncclCommFinalize(comm)
    @ccall libnccl.pncclCommFinalize(comm::ncclComm_t)::ncclResult_t
end

@checked function ncclCommDestroy(comm)
    @ccall libnccl.ncclCommDestroy(comm::ncclComm_t)::ncclResult_t
end

@checked function pncclCommDestroy(comm)
    @ccall libnccl.pncclCommDestroy(comm::ncclComm_t)::ncclResult_t
end

@checked function ncclCommAbort(comm)
    @ccall libnccl.ncclCommAbort(comm::ncclComm_t)::ncclResult_t
end

@checked function pncclCommAbort(comm)
    @ccall libnccl.pncclCommAbort(comm::ncclComm_t)::ncclResult_t
end

@checked function ncclCommSplit(comm, color, key, newcomm, config)
    @ccall libnccl.ncclCommSplit(comm::ncclComm_t, color::Cint, key::Cint,
                                 newcomm::Ptr{ncclComm_t},
                                 config::Ptr{ncclConfig_t})::ncclResult_t
end

@checked function pncclCommSplit(comm, color, key, newcomm, config)
    @ccall libnccl.pncclCommSplit(comm::ncclComm_t, color::Cint, key::Cint,
                                  newcomm::Ptr{ncclComm_t},
                                  config::Ptr{ncclConfig_t})::ncclResult_t
end

function ncclGetErrorString(result)
    @ccall libnccl.ncclGetErrorString(result::ncclResult_t)::Cstring
end

function pncclGetErrorString(result)
    @ccall libnccl.pncclGetErrorString(result::ncclResult_t)::Cstring
end

function ncclGetLastError(comm)
    @ccall libnccl.ncclGetLastError(comm::ncclComm_t)::Cstring
end

function pncclGetLastError(comm)
    @ccall libnccl.pncclGetLastError(comm::ncclComm_t)::Cstring
end

@checked function ncclCommGetAsyncError(comm, asyncError)
    @ccall libnccl.ncclCommGetAsyncError(comm::ncclComm_t,
                                         asyncError::Ptr{ncclResult_t})::ncclResult_t
end

@checked function pncclCommGetAsyncError(comm, asyncError)
    @ccall libnccl.pncclCommGetAsyncError(comm::ncclComm_t,
                                          asyncError::Ptr{ncclResult_t})::ncclResult_t
end

@checked function ncclCommCount(comm, count)
    @ccall libnccl.ncclCommCount(comm::ncclComm_t, count::Ptr{Cint})::ncclResult_t
end

@checked function pncclCommCount(comm, count)
    @ccall libnccl.pncclCommCount(comm::ncclComm_t, count::Ptr{Cint})::ncclResult_t
end

@checked function ncclCommCuDevice(comm, device)
    @ccall libnccl.ncclCommCuDevice(comm::ncclComm_t, device::Ptr{Cint})::ncclResult_t
end

@checked function pncclCommCuDevice(comm, device)
    @ccall libnccl.pncclCommCuDevice(comm::ncclComm_t, device::Ptr{Cint})::ncclResult_t
end

@checked function ncclCommUserRank(comm, rank)
    @ccall libnccl.ncclCommUserRank(comm::ncclComm_t, rank::Ptr{Cint})::ncclResult_t
end

@checked function pncclCommUserRank(comm, rank)
    @ccall libnccl.pncclCommUserRank(comm::ncclComm_t, rank::Ptr{Cint})::ncclResult_t
end

@cenum ncclRedOp_dummy_t::UInt32 begin
    ncclNumOps_dummy = 5
end

@cenum ncclRedOp_t::UInt32 begin
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5
    ncclMaxRedOp = 2147483647
end

@cenum ncclDataType_t::UInt32 begin
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclNumTypes = 9
end

@cenum ncclScalarResidence_t::UInt32 begin
    ncclScalarDevice = 0
    ncclScalarHostImmediate = 1
end

@checked function ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)
    @ccall libnccl.ncclRedOpCreatePreMulSum(op::Ptr{ncclRedOp_t}, scalar::Ptr{Cvoid},
                                            datatype::ncclDataType_t,
                                            residence::ncclScalarResidence_t,
                                            comm::ncclComm_t)::ncclResult_t
end

@checked function pncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)
    @ccall libnccl.pncclRedOpCreatePreMulSum(op::Ptr{ncclRedOp_t}, scalar::Ptr{Cvoid},
                                             datatype::ncclDataType_t,
                                             residence::ncclScalarResidence_t,
                                             comm::ncclComm_t)::ncclResult_t
end

@checked function ncclRedOpDestroy(op, comm)
    @ccall libnccl.ncclRedOpDestroy(op::ncclRedOp_t, comm::ncclComm_t)::ncclResult_t
end

@checked function pncclRedOpDestroy(op, comm)
    @ccall libnccl.pncclRedOpDestroy(op::ncclRedOp_t, comm::ncclComm_t)::ncclResult_t
end

@checked function ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    @ccall libnccl.ncclReduce(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid}, count::Cint,
                              datatype::ncclDataType_t, op::ncclRedOp_t, root::Cint,
                              comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function pncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    @ccall libnccl.pncclReduce(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid}, count::Cint,
                               datatype::ncclDataType_t, op::ncclRedOp_t, root::Cint,
                               comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function ncclBcast(buff, count, datatype, root, comm, stream)
    @ccall libnccl.ncclBcast(buff::CuPtr{Cvoid}, count::Cint, datatype::ncclDataType_t,
                             root::Cint, comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function pncclBcast(buff, count, datatype, root, comm, stream)
    @ccall libnccl.pncclBcast(buff::CuPtr{Cvoid}, count::Cint, datatype::ncclDataType_t,
                              root::Cint, comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    @ccall libnccl.ncclBroadcast(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                 count::Cint, datatype::ncclDataType_t, root::Cint,
                                 comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function pncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    @ccall libnccl.pncclBroadcast(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                  count::Cint, datatype::ncclDataType_t, root::Cint,
                                  comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    @ccall libnccl.ncclAllReduce(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                 count::Cint, datatype::ncclDataType_t, op::ncclRedOp_t,
                                 comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function pncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    @ccall libnccl.pncclAllReduce(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                  count::Cint, datatype::ncclDataType_t, op::ncclRedOp_t,
                                  comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm,
                                    stream)
    @ccall libnccl.ncclReduceScatter(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                     recvcount::Cint, datatype::ncclDataType_t,
                                     op::ncclRedOp_t, comm::ncclComm_t,
                                     stream::CUstream)::ncclResult_t
end

@checked function pncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm,
                                     stream)
    @ccall libnccl.pncclReduceScatter(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                      recvcount::Cint, datatype::ncclDataType_t,
                                      op::ncclRedOp_t, comm::ncclComm_t,
                                      stream::CUstream)::ncclResult_t
end

@checked function ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    @ccall libnccl.ncclAllGather(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                 sendcount::Cint, datatype::ncclDataType_t,
                                 comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function pncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    @ccall libnccl.pncclAllGather(sendbuff::CuPtr{Cvoid}, recvbuff::CuPtr{Cvoid},
                                  sendcount::Cint, datatype::ncclDataType_t,
                                  comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function ncclSend(sendbuff, count, datatype, peer, comm, stream)
    @ccall libnccl.ncclSend(sendbuff::CuPtr{Cvoid}, count::Cint, datatype::ncclDataType_t,
                            peer::Cint, comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function pncclSend(sendbuff, count, datatype, peer, comm, stream)
    @ccall libnccl.pncclSend(sendbuff::CuPtr{Cvoid}, count::Cint, datatype::ncclDataType_t,
                             peer::Cint, comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function pncclRecv(recvbuff, count, datatype, peer, comm, stream)
    @ccall libnccl.pncclRecv(recvbuff::CuPtr{Cvoid}, count::Cint, datatype::ncclDataType_t,
                             peer::Cint, comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

@checked function ncclRecv(recvbuff, count, datatype, peer, comm, stream)
    @ccall libnccl.ncclRecv(recvbuff::CuPtr{Cvoid}, count::Cint, datatype::ncclDataType_t,
                            peer::Cint, comm::ncclComm_t, stream::CUstream)::ncclResult_t
end

# no prototype is found for this function at nccl.h:416:15, please use with caution
@checked function ncclGroupStart()
    @ccall libnccl.ncclGroupStart()::ncclResult_t
end

# no prototype is found for this function at nccl.h:417:14, please use with caution
@checked function pncclGroupStart()
    @ccall libnccl.pncclGroupStart()::ncclResult_t
end

# no prototype is found for this function at nccl.h:426:15, please use with caution
@checked function ncclGroupEnd()
    @ccall libnccl.ncclGroupEnd()::ncclResult_t
end

# no prototype is found for this function at nccl.h:427:14, please use with caution
@checked function pncclGroupEnd()
    @ccall libnccl.pncclGroupEnd()::ncclResult_t
end

@checked function ncclCommRegister(comm, buff, size, handle)
    @ccall libnccl.ncclCommRegister(comm::ncclComm_t, buff::CuPtr{Cvoid}, size::Cint,
                                    handle::Ptr{Ptr{Cvoid}})::ncclResult_t
end

@checked function pncclCommRegister(comm, buff, size, handle)
    @ccall libnccl.pncclCommRegister(comm::ncclComm_t, buff::CuPtr{Cvoid}, size::Cint,
                                     handle::Ptr{Ptr{Cvoid}})::ncclResult_t
end

@checked function ncclCommDeregister(comm, handle)
    @ccall libnccl.ncclCommDeregister(comm::ncclComm_t, handle::CuPtr{Cvoid})::ncclResult_t
end

@checked function pncclCommDeregister(comm, handle)
    @ccall libnccl.pncclCommDeregister(comm::ncclComm_t, handle::CuPtr{Cvoid})::ncclResult_t
end

const NCCL_MAJOR = 2

const NCCL_MINOR = 19

const NCCL_PATCH = 4

const NCCL_SUFFIX = ""

const NCCL_VERSION_CODE = 21904

const NCCL_COMM_NULL = NULL

const NCCL_UNIQUE_ID_BYTES = 128

const NCCL_CONFIG_UNDEF_INT = INT_MIN

const NCCL_CONFIG_UNDEF_PTR = NULL

const NCCL_SPLIT_NOCOLOR = -1

# Skipping MacroDefinition: NCCL_CONFIG_INITIALIZER { sizeof ( ncclConfig_t ) , /* size */ 0xcafebeef , /* magic */ NCCL_VERSION ( NCCL_MAJOR , NCCL_MINOR , NCCL_PATCH ) , /* version */ NCCL_CONFIG_UNDEF_INT , /* blocking */ NCCL_CONFIG_UNDEF_INT , /* cgaClusterSize */ NCCL_CONFIG_UNDEF_INT , /* minCTAs */ NCCL_CONFIG_UNDEF_INT , /* maxCTAs */ NCCL_CONFIG_UNDEF_PTR , /* netName */ NCCL_CONFIG_UNDEF_INT /* splitShare */ \
#}

export NCCLError

struct NCCLError <: Exception
    code::ncclResult_t
    msg::AbstractString
end
Base.show(io::IO, err::NCCLError) = print(io, "NCCLError(code $(err.code), $(err.msg))")

function NCCLError(code::ncclResult_t)
    msg = status_message(code)
    return NCCLError(code, msg)
end

function status_message(status)
    if status == ncclSuccess
        return "function succeeded"
    elseif status == ncclUnhandledCudaError
        return "a call to a CUDA function failed"
    elseif status == ncclSystemError
        return "a call to the system failed"
    elseif status == ncclInternalError
        return "an internal check failed. This is either a bug in NCCL or due to memory corruption"
    elseif status == ncclInvalidArgument
        return "one argument has an invalid value"
    elseif status == ncclInvalidUsage
        return "the call to NCCL is incorrect. This is usually reflecting a programming error"
    elseif status == ncclRemoteError
        return "A call failed possibly due to a network error or a remote process exiting prematurely."
    elseif status == ncclInProgress
        return "A NCCL operation on the communicator is being enqueued and is being progressed in the background."
    else
        return "unknown status"
    end
end

# exports
const PREFIXES = ["nccl"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
