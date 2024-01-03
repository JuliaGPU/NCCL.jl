const NULL = C_NULL
const INT_MIN = typemin(Cint)

import CUDA.APIUtils: @checked
import CUDA: CuPtr, CUstream

function check(f)
    res = f()::ncclResult_t
    if res != ncclSuccess
        throw(NCCLError(err))
    end
    return
end
