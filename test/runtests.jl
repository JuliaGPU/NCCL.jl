using Test

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using NCCL
@info "NCCL version: $(NCCL.version())"

@testset "NCCL" begin

@testset "Communicator" begin
    # clique of communicators
    comms = NCCL.Communicators(CUDA.devices())
    for (i,dev) in enumerate(CUDA.devices())
        @test NCCL.rank(comms[i]) == i-1
        @test NCCL.device(comms[i]) == dev
        @test NCCL.size(comms[i]) == length(CUDA.devices())
    end

    # single communicator (with nranks=1 or this would block)
    comm  = Communicator(1, 0)
    @test NCCL.device(comm) == CuDevice(0)
end

@testset "Allreduce!" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)

    @testset "sum" begin
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        N = 512
        for (ii, dev) in enumerate(devs)
            CUDA.device!(ii - 1)
            sendbuf[ii] = CuArray(fill(Float64(ii), N))
            recvbuf[ii] = CUDA.zeros(Float64, N)
        end
        NCCL.group() do
            for ii in 1:length(devs)
                NCCL.Allreduce!(sendbuf[ii], recvbuf[ii], +, comms[ii])
            end
        end
        answer = sum(1:length(devs))
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test all(crecv .== answer)
        end
    end

    @testset "NCCL.avg" begin
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        N = 512
        for (ii, dev) in enumerate(devs)
            CUDA.device!(ii - 1)
            sendbuf[ii] = CuArray(fill(Float64(ii), N))
            recvbuf[ii] = CUDA.zeros(Float64, N)
        end
        NCCL.group() do
            for ii in 1:length(devs)
                NCCL.Allreduce!(sendbuf[ii], recvbuf[ii], NCCL.avg, comms[ii])
            end
        end
        answer = sum(1:length(devs)) / length(devs)
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            crecv = collect(recvbuf[ii])
            @test all(crecv .≈ answer)
        end
    end
end

@testset "Broadcast!" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = (ii - 1) == root ? CuArray(fill(Float64(1.0), 512)) : CUDA.zeros(Float64, 512)
        recvbuf[ii] = CUDA.zeros(Float64, 512)
    end
    NCCL.group() do
        for ii in 1:length(devs)
            NCCL.Broadcast!(sendbuf[ii], recvbuf[ii], comms[ii]; root)
        end
    end
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
        sendbuf[ii] = CuArray(fill(Float64(ii), 512))
        recvbuf[ii] = CUDA.zeros(Float64, 512)
    end
    NCCL.group() do
        for ii in 1:length(devs)
            NCCL.Reduce!(sendbuf[ii], recvbuf[ii], +, comms[ii]; root)
        end
    end
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
        sendbuf[ii] = CuArray(fill(Float64(ii), 512))
        recvbuf[ii] = CUDA.zeros(Float64, length(devs)*512)
    end
    NCCL.group() do
        for ii in 1:length(devs)
            NCCL.Allgather!(sendbuf[ii], recvbuf[ii], comms[ii])
        end
    end
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
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = CuArray(vec(repeat(collect(1:length(devs)), inner=2)))
        recvbuf[ii] = CUDA.zeros(Float64, 2)
    end
    NCCL.group() do
        for ii in 1:length(devs)
            NCCL.ReduceScatter!(sendbuf[ii], recvbuf[ii], +, comms[ii])
        end
    end
    for (ii, dev) in enumerate(devs)
        answer = length(devs)*ii
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

@testset "Send/Recv" begin
    devs  = CUDA.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    N = 512
    for (ii, dev) in enumerate(devs)
        CUDA.device!(ii - 1)
        sendbuf[ii] = CuArray(fill(Float64(ii), N))
        recvbuf[ii] = CUDA.zeros(Float64, N)
    end

    NCCL.group() do
        for ii in 1:length(devs)
            comm = comms[ii]
            dest = mod(NCCL.rank(comm)+1, NCCL.size(comm))
            source = mod(NCCL.rank(comm)-1, NCCL.size(comm))
            NCCL.Send(sendbuf[ii], comm; dest)
            NCCL.Recv!(recvbuf[ii], comm; source)
        end
    end
    for (ii, dev) in enumerate(devs)
        answer = mod1(ii - 1, length(devs))
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end
end

end
