module NCCL

using CUDACore, CEnum

include("libnccl.jl")
using .LibNCCL

include("base.jl")
include("group.jl")
include("config.jl")
include("communicator.jl")
include("buffers.jl")
include("pointtopoint.jl")
include("collective.jl")

__init__() = force_blocking()

end
