const NULL = C_NULL
const INT_MIN = typemin(Cint)
const INT_MAX = typemax(Cint)

using CUDACore: CuPtr, PtrOrCuPtr, CUstream, CUevent
using GPUToolbox: @checked, @gcsafe_ccall

# types from the CUDA runtime headers, which aren't wrapped here
const cudaStream_t = CUstream
const cudaEvent_t = CUevent

function check(f)
    res = f()::ncclResult_t
    if res != ncclSuccess
        throw(NCCLError(res))
    end
    return
end
