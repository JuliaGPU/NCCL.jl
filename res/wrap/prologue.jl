const NULL = C_NULL
const INT_MIN = typemin(Cint)

using CUDACore: CuPtr, CUstream
using GPUToolbox: @checked

function check(f)
    res = f()::ncclResult_t
    if res != ncclSuccess
        throw(NCCLError(res))
    end
    return
end
