using NCCL, CUDAdrv, CuArrays, CUDAnative
using Test

@testset "NCCL.jl" begin
    @testset "Communicator" begin
        comms = Communicator(CUDAdrv.devices())
        for (i,dev) in enumerate(CUDAdrv.devices())
            @test rank(comms[i]) == i-1
            @test device(comms[i]) == i-1
            @test size(comms[i]) == length(CUDAdrv.devices())
        end
        #id    = UniqueID()
        #num_devs = length(CUDAdrv.devices())
        #comm  = Communicator(num_devs, id, 0)
        #@test device(comm) == 0
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
        # more complex example?
        recvbuf = Vector{CuMatrix{Float64}}(undef, length(devs))
        sendbuf = Vector{CuMatrix{Float64}}(undef, length(devs))
        streams = Vector{CuStream}(undef, length(devs))
        m       = 256
        k       = 512
        n       = 256
        As      = [rand(m, k) for i in 1:length(devs)]
        Bs      = [rand(k, n) for i in 1:length(devs)]
        C       = sum(As .* Bs)
        for (ii, dev) in enumerate(devs)
            CUDAnative.device!(ii - 1)
            sendbuf[ii] = cu(As[ii]) * cu(Bs[ii])
            recvbuf[ii] = CuArrays.zeros(Float64, m, n)
            streams[ii] = CuStream()
        end
        groupStart()
        for ii in 1:length(devs)
            Allreduce!(sendbuf[ii], recvbuf[ii], m*n, NCCL.ncclSum, comms[ii], stream=streams[ii])
        end
        groupEnd()
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test crecv ≈ C rtol=1e-6
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
            ReduceScatter!(sendbuf[ii], recvbuf[ii], length(devs), NCCL.ncclSum, comms[ii], stream=streams[ii])
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
