module NCCL

using CUDACore, CEnum
using BFloat16s: BFloat16

include("libnccl.jl")
using .LibNCCL

include("base.jl")
include("eltypes.jl")
include("group.jl")
include("config.jl")
include("communicator.jl")
include("buffers.jl")
include("pointtopoint.jl")
include("collective.jl")

function __init__()
    force_blocking()
    Base.Experimental.register_error_hint(mpi_hint, MethodError)
end

end
