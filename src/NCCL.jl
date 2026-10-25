module NCCL

using CUDACore, CEnum

include("libnccl.jl")
using .LibNCCL

include("base.jl")
include("group.jl")
include("communicator.jl")
include("pointtopoint.jl")
include("collective.jl")

end
