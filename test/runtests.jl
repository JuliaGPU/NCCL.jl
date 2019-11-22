using NCCL, CUDAdrv
using Test

@testset "NCCL.jl" begin
    #id    = UniqueID()
    #comm  = Communicator(length(CUDAdrv.devices()), id, 0)
    comms = Communicator(CUDAdrv.devices())
end
