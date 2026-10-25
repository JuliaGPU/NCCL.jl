import NCCL_jll: is_available

"""
    NCCL.is_available() :: Bool

Is the NCCL library available?
"""
is_available

"""
    NCCL.version() :: VersionNumber

Get the version of the current NCCL library.
"""
function version()
    ver_r = Ref{Cint}()
    ncclGetVersion(ver_r)
    ver = ver_r[]

    if ver < 2900
        major, ver = divrem(ver, 1000)
        minor, patch = divrem(ver, 100)
    else
        major, ver = divrem(ver, 10000)
        minor, patch = divrem(ver, 100)
    end
    VersionNumber(major, minor, patch)
end

import .LibNCCL: ncclRedOp_t, ncclDataType_t

ncclRedOp_t(::typeof(+)) = ncclSum
ncclRedOp_t(::typeof(*)) = ncclProd
ncclRedOp_t(::typeof(max)) = ncclMax
ncclRedOp_t(::typeof(min)) = ncclMin
# Handles the case where user directly passed in the ncclRedOp_t (eg. `NCCL.avg`)
ncclRedOp_t(x::ncclRedOp_t) = x

"""
    NCCL.avg

Perform an average operation, i.e. a sum across all ranks, divided by the number
of ranks. For integers, the sum wraps around on overflow, and the division truncates
towards zero.
"""
const avg = ncclAvg
