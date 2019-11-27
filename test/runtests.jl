using NCCL, CUDAdrv, CuArrays, CUDAnative
using Test

@testset "NCCL.jl" begin
    @testset "Communicator" begin
        comms = Communicator(CUDAdrv.devices())
        @test device(comms[1]) == 0
        @test device(comms[2]) == 1
        @test size(comms[1])   == 2
        @test rank(comms[1])   == 0 
        @test rank(comms[2])   == 1 
        id    = UniqueID()
        #=num_devs = length(CUDAdrv.devices())
        comm  = Communicator(num_devs, id, 0)
        @test device(comm) == 0=#
    end
    @testset "Allreduce!" begin
        devs  = CUDAdrv.devices()
        comms = Communicator(devs)
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        streams = Vector{CuStream}(undef, length(devs))
        for (ii, dev) in enumerate(devs)
            CUDAnative.device!(ii - 1)
            sendbuf[ii] = cu(fill(Float64(ii), 512))
            recvbuf[ii] = CuArrays.zeros(Float64, 512)
            streams[ii] = CuStream()
        end
        groupStart()
        for ii in 1:length(devs)
            Allreduce!(sendbuf[ii], recvbuf[ii], 512, NCCL.ncclSum, comms[ii], stream=streams[ii])
        end
        groupEnd()
        answer = sum(1:length(devs)) 
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test all(crecv .== answer)
        end
    end
    @testset "Broadcast!" begin
        devs  = CUDAdrv.devices()
        comms = Communicator(devs)
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        streams = Vector{CuStream}(undef, length(devs))
        root  = 0
        for (ii, dev) in enumerate(devs)
            CUDAnative.device!(ii - 1)
            sendbuf[ii] = (ii - 1) == root ? cu(fill(Float64(1.0), 512)) : CuArrays.zeros(Float64, 512)
            recvbuf[ii] = CuArrays.zeros(Float64, 512)
            streams[ii] = CuStream()
        end
        groupStart()
        for ii in 1:length(devs)
            Broadcast!(sendbuf[ii], recvbuf[ii], 512, 0, comms[ii], stream=streams[ii])
        end
        groupEnd()
        answer = 1.0
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test all(crecv .== answer)
        end
    end
    @testset "Reduce!" begin
        devs  = CUDAdrv.devices()
        comms = Communicator(devs)
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        streams = Vector{CuStream}(undef, length(devs))
        root  = 0
        for (ii, dev) in enumerate(devs)
            CUDAnative.device!(ii - 1)
            sendbuf[ii] = cu(fill(Float64(ii), 512))
            recvbuf[ii] = CuArrays.zeros(Float64, 512)
            streams[ii] = CuStream()
        end
        groupStart()
        for ii in 1:length(devs)
            Reduce!(sendbuf[ii], recvbuf[ii], 512, NCCL.ncclSum, 0, comms[ii], stream=streams[ii])
        end
        groupEnd()
        for (ii, dev) in enumerate(devs)
            answer = (ii - 1) == root ? sum(1:length(devs)) : 0.0
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test all(crecv .== answer)
        end
    end
    @testset "Allgather!" begin
        devs  = CUDAdrv.devices()
        comms = Communicator(devs)
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        streams = Vector{CuStream}(undef, length(devs))
        for (ii, dev) in enumerate(devs)
            CUDAnative.device!(ii - 1)
            sendbuf[ii] = cu(fill(Float64(ii), 512))
            recvbuf[ii] = CuArrays.zeros(Float64, length(devs)*512)
            streams[ii] = CuStream()
        end
        groupStart()
        for ii in 1:length(devs)
            Allgather!(sendbuf[ii], recvbuf[ii], 512, comms[ii], stream=streams[ii])
        end
        groupEnd()
        answer = vec(repeat(1:length(devs), inner=512))  
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test all(crecv .== answer)
        end
    end
    @testset "ReduceScatter!" begin
        devs  = CUDAdrv.devices()
        comms = Communicator(devs)
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        streams = Vector{CuStream}(undef, length(devs))
        root  = 0
        for (ii, dev) in enumerate(devs)
            CUDAnative.device!(ii - 1)
            sendbuf[ii] = cu(vec(repeat(collect(1:length(devs)), inner=2)))
            recvbuf[ii] = CuArrays.zeros(Float64, length(devs))
            streams[ii] = CuStream()
        end
        groupStart()
        for ii in 1:length(devs)
            ReduceScatter!(sendbuf[ii], recvbuf[ii], 2, NCCL.ncclSum, comms[ii], stream=streams[ii])
        end
        groupEnd()
        for (ii, dev) in enumerate(devs)
            answer = length(devs)*ii
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test all(crecv .== answer)
        end
    end
end
