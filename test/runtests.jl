using NCCL, CUDA
using Test

@testset "Communicator" begin
    comms = NCCL.Communicators(CUDA.devices())
    for (i,dev) in enumerate(CUDA.devices())
        @test NCCL.rank(comms[i]) == i-1
        @test NCCL.deviceid(comms[i]) == i-1
        @test NCCL.size(comms[i]) == length(CUDA.devices())
    end
    id    = NCCL.UniqueID()
    #=num_devs = length(CUDA.devices())
    comm  = Communicator(num_devs, id, 0)
    @test device(comm) == 0=#
end

@testset "Allreduce!" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    N = 512
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = cu(fill(Float64(ii), N))
        recvbuf[ii] = CUDA.zeros(Float64, N)
    end
    NCCL.groupStart()
    for ii in 1:length(devs)
        NCCL.Allreduce!(sendbuf[ii], recvbuf[ii], +, comms[ii])
    end
    NCCL.groupEnd()
    answer = sum(1:length(devs))
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "Broadcast!" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    streams = Vector{CuStream}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = (ii - 1) == root ? cu(fill(Float64(1.0), 512)) : CUDA.zeros(Float64, 512)
        recvbuf[ii] = CUDA.zeros(Float64, 512)
    end
    NCCL.groupStart()
    for ii in 1:length(devs)
        NCCL.Broadcast!(sendbuf[ii], recvbuf[ii], 512, 0, comms[ii])
    end
    NCCL.groupEnd()
    answer = 1.0
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "Reduce!" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = cu(fill(Float64(ii), 512))
        recvbuf[ii] = CUDA.zeros(Float64, 512)
    end
    NCCL.groupStart()
    for ii in 1:length(devs)
        NCCL.Reduce!(sendbuf[ii], recvbuf[ii], +, comms[ii]; root=0)
    end
    NCCL.groupEnd()
    for (ii, dev) in enumerate(devs)
        answer = (ii - 1) == root ? sum(1:length(devs)) : 0.0
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "Allgather!" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = cu(fill(Float64(ii), 512))
        recvbuf[ii] = CUDA.zeros(Float64, length(devs)*512)
    end
    NCCL.groupStart()
    for ii in 1:length(devs)
        NCCL.Allgather!(sendbuf[ii], recvbuf[ii], comms[ii])
    end
    NCCL.groupEnd()
    answer = vec(repeat(1:length(devs), inner=512))
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "ReduceScatter!" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = cu(vec(repeat(collect(1:length(devs)), inner=2)))
        recvbuf[ii] = CUDA.zeros(Float64, length(devs))
    end
    NCCL.groupStart()
    for ii in 1:length(devs)
        NCCL.ReduceScatter!(sendbuf[ii], recvbuf[ii], +, comms[ii])
    end
    NCCL.groupEnd()
    for (ii, dev) in enumerate(devs)
        answer = length(devs)*ii
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end
