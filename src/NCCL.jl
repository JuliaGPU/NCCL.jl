module NCCL

using Printf

using CuArrays

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream, CuStream, CuPtr

using CEnum

const libnccl = "libnccl"

# core library
include("libnccl_common.jl")
include("error.jl")
include("libnccl.jl")

include("base.jl")
include("communicator.jl")
include("group.jl")
include("collective.jl")

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false")) || precompiling
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional()
        verbose && @warn "NCCL.jl did not initialize because CUDAdrv.jl failed to"
        return
    end

    try
        if version() < v"2"
            silent || @warn "NCCL.jl only supports NCCL 2.x (you are using $(version()))"
        end
    catch ex
        # don't actually fail to keep the package loadable
        if !silent
            if verbose
                @error "NCCL.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "NCCL.jl failed to initialize, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

end
