module NCCL

using Printf

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("NCCL.jl has not been built, please run Pkg.build(\"NCCL\").")
include(ext)
if !configured
    # default (non-functional) values for critical variables,
    # making it possible to _load_ the package at all times.
    const libnccl = nothing
end

include("base.jl")

end
