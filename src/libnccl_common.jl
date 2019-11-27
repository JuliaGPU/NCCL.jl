# Automatically generated using Clang.jl


const NCCL_MAJOR = 2
const NCCL_MINOR = 4
const NCCL_PATCH = 8
const NCCL_SUFFIX = ""
const NCCL_VERSION_CODE = 2408

# Skipping MacroDefinition: NCCL_VERSION ( X , Y , Z ) ( ( X ) * 1000 + ( Y ) * 100 + ( Z ) )

const NCCL_UNIQUE_ID_BYTES = 128
const ncclComm = Cvoid
const ncclComm_t = Ptr{ncclComm}

struct ncclUniqueId
    internal::NTuple{128, UInt8}
end

@cenum ncclResult_t::UInt32 begin
    ncclSuccess = 0
    ncclUnhandledCudaError = 1
    ncclSystemError = 2
    ncclInternalError = 3
    ncclInvalidArgument = 4
    ncclInvalidUsage = 5
    ncclNumResults = 6
end

@cenum ncclRedOp_t::UInt32 begin
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclNumOps = 4
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

