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

# the API that isn't exported, to be used as `NCCL.name`
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public,
        :Communicators, :UniqueID, :Config, :rank, :nranks, :device, :split, :destroy, :abort,
        :Allreduce!, :Broadcast!, :Reduce!, :Allgather!, :ReduceScatter!, :Alltoall!,
        :Gather!, :Scatter!, :Send, :Recv!, :avg, :PreMulSum, :group, :version,
        :is_available, :NCCLError))
end

function __init__()
    force_blocking()
    Base.Experimental.register_error_hint(mpi_hint, MethodError)
end

end
