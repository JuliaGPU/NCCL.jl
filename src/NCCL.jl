module NCCL

using CUDA, CEnum

include("libnccl.jl")
using .LibNCCL

include("base.jl")
include("communicator.jl")
include("group.jl")
include("pointtopoint.jl")
include("collective.jl")

end
