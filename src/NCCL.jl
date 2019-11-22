module NCCL

using Printf

using CuArrays
import CUDAdrv, CUDAnative
using CUDAdrv: CUstream, CuStream, CuPtr

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("NCCL.jl has not been built, please run Pkg.build(\"NCCL\").")
include(ext)
if !configured
    # default (non-functional) values for critical variables,
    # making it possible to _load_ the package at all times.
    const libnccl = nothing
end

include("base.jl")
include("communicator.jl")
include("group.jl")
include("collective.jl")

export UniqueID, Communicator, rank, groupStart, groupEnd, Allreduce!, Broadcast!, Reduce!

end
