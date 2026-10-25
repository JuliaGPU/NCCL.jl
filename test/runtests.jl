using Test

using CUDACore
@info "CUDA driver $(CUDACore.driver_version()), runtime $(CUDACore.runtime_version())"
for dev in CUDACore.devices()
    @info "Device $(deviceid(dev)): $(CUDACore.name(dev))"
end

using NCCL_jll
@info "host_platform: $(NCCL_jll.host_platform)"
@info "is_available: $(NCCL_jll.is_available())"
@info "best_wrapper: $(NCCL_jll.best_wrapper)"

using NCCL
using BFloat16s: BFloat16
@info "NCCL version: $(NCCL.version())"

include("helpers.jl")

# an isbits type that NCCL doesn't support (with padding after `id`)
struct Particle
    position::NTuple{3,Float32}
    id::Int16
end

# a type whose values are padded in arrays: 3 bytes, stored every 4 bytes
primitive type UInt24 24 end
UInt24(x::Integer) = Core.Intrinsics.trunc_int(UInt24, UInt32(x))

@testset "NCCL" begin

@testset "Communicator" begin
    # clique of communicators
    comms = NCCL.Communicators(CUDACore.devices())
    for (i,dev) in enumerate(CUDACore.devices())
        @test NCCL.rank(comms[i]) == i-1
        @test NCCL.device(comms[i]) == dev
        @test NCCL.nranks(comms[i]) == length(CUDACore.devices())
    end

    # single communicator (with nranks=1 or this would block)
    comm = Communicator(1, 0, NCCL.UniqueID())
    @test NCCL.device(comm) == CUDACore.device()
    @test NCCL.rank(comm) == 0
    @test NCCL.nranks(comm) == 1
    @test isopen(comm)
    @test occursin("rank 0 of 1", sprint(show, comm))

    NCCL.destroy(comms)
    NCCL.destroy(comm)
end

@testset "Allreduce!" begin
    devs  = CUDACore.devices()
    comms = NCCL.Communicators(devs)

    @testset "sum" begin
        recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
        sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
        N = 512
        for (ii, dev) in enumerate(devs)
            CUDACore.device!(ii - 1)
            sendbuf[ii] = CuArray(fill(Float64(ii), N))
            recvbuf[ii] = CUDACore.zeros(Float64, N)
        end
        current_device = CUDACore.device()
        if length(devs) > 1
            @test_throws ArgumentError NCCL.Allreduce!(sendbuf[2], recvbuf[2], +, comms[1])
        end
        NCCL.group() do
            for ii in 1:length(devs)
                NCCL.Allreduce!(sendbuf[ii], recvbuf[ii], +, comms[ii])
            end
        end
        @test CUDACore.device() == current_device
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
            CUDACore.device!(ii - 1)
            sendbuf[ii] = CuArray(fill(Float64(ii), N))
            recvbuf[ii] = CUDACore.zeros(Float64, N)
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

    NCCL.destroy(comms)
end

@testset "Broadcast!" begin
    devs  = CUDACore.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
        CUDACore.device!(ii - 1)
        sendbuf[ii] = (ii - 1) == root ? CuArray(fill(Float64(1.0), 512)) : CUDACore.zeros(Float64, 512)
        recvbuf[ii] = CUDACore.zeros(Float64, 512)
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

    # the send buffer is only needed on the root
    for root in 0:length(devs)-1
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            sendbuf[ii] = CuArray(fill(Float64(root + 1), 512))
            recvbuf[ii] = CUDACore.zeros(Float64, 512)
        end
        NCCL.group() do
            for ii in 1:length(devs)
                send = ii - 1 == root ? sendbuf[ii] : nothing
                @test NCCL.Broadcast!(send, recvbuf[ii], comms[ii]; root) === recvbuf[ii]
            end
        end
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            @test collect(recvbuf[ii]) == fill(root + 1, 512)
        end
    end

    NCCL.destroy(comms)
end

@testset "Reduce!" begin
    devs  = CUDACore.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    root  = 0
    for (ii, dev) in enumerate(devs)
        CUDACore.device!(ii - 1)
        sendbuf[ii] = CuArray(fill(Float64(ii), 512))
        recvbuf[ii] = CUDACore.zeros(Float64, 512)
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

    # the receive buffer is only needed on the root
    for root in 0:length(devs)-1
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            sendbuf[ii] = CuArray(fill(Float64(ii), 512))
            recvbuf[ii] = CUDACore.zeros(Float64, 512)
        end
        NCCL.group() do
            for ii in 1:length(devs)
                recv = ii - 1 == root ? recvbuf[ii] : nothing
                @test NCCL.Reduce!(sendbuf[ii], recv, +, comms[ii]; root) === recv
            end
        end
        for (ii, dev) in enumerate(devs)
            device!(ii - 1)
            @test collect(sendbuf[ii]) == fill(ii, 512)
            if ii - 1 == root
                @test collect(recvbuf[ii]) == fill(sum(1:length(devs)), 512)
            end
        end
    end

    NCCL.destroy(comms)
end

@testset "Allgather!" begin
    devs  = CUDACore.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    for (ii, dev) in enumerate(devs)
        CUDACore.device!(ii - 1)
        sendbuf[ii] = CuArray(fill(Float64(ii), 512))
        recvbuf[ii] = CUDACore.zeros(Float64, length(devs)*512)
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

    # in place: each rank's send buffer is its own slice of the receive buffer
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        recvbuf[ii] = CuArray(fill(-1.0, length(devs)*512))
        recvbuf[ii][(ii-1)*512+1:ii*512] .= ii
    end
    NCCL.group() do
        for ii in 1:length(devs)
            NCCL.Allgather!(view(recvbuf[ii], (ii-1)*512+1:ii*512), recvbuf[ii], comms[ii])
        end
    end
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        @test collect(recvbuf[ii]) == answer
    end

    # in place, with a single buffer
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        recvbuf[ii] = CuArray(fill(-1.0, length(devs)*512))
        recvbuf[ii][(ii-1)*512+1:ii*512] .= ii
    end
    NCCL.group() do
        for ii in 1:length(devs)
            @test NCCL.Allgather!(recvbuf[ii], comms[ii]) === recvbuf[ii]
        end
    end
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        @test collect(recvbuf[ii]) == answer
    end
    if length(devs) > 1
        device!(0)
        @test_throws DimensionMismatch NCCL.Allgather!(CUDACore.zeros(length(devs) + 1),
                                                       comms[1])
    end

    NCCL.destroy(comms)
end

@testset "ReduceScatter!" begin
    devs  = CUDACore.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    for (ii, dev) in enumerate(devs)
        CUDACore.device!(ii - 1)
        sendbuf[ii] = CuArray(vec(repeat(collect(1:length(devs)), inner=2)))
        recvbuf[ii] = CUDACore.zeros(Float64, 2)
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

    # in place: each rank's receive buffer is its own block of the send buffer
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        sendbuf[ii] = CuArray(vec(repeat(collect(1.0:length(devs)), inner=2)))
    end
    NCCL.group() do
        for ii in 1:length(devs)
            NCCL.ReduceScatter!(sendbuf[ii], view(sendbuf[ii], 2ii-1:2ii), +, comms[ii])
        end
    end
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        @test collect(sendbuf[ii])[2ii-1:2ii] == fill(length(devs)*ii, 2)
    end

    # in place, with a single buffer: returns this rank's block
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        sendbuf[ii] = CuArray(vec(repeat(collect(1.0:length(devs)), inner=2)))
    end
    results = NCCL.group() do
        map(1:length(devs)) do ii
            NCCL.ReduceScatter!(sendbuf[ii], +, comms[ii])
        end
    end
    for (ii, dev) in enumerate(devs)
        device!(ii - 1)
        @test length(results[ii]) == 2
        @test collect(results[ii]) == fill(length(devs)*ii, 2)
        @test collect(sendbuf[ii])[2ii-1:2ii] == fill(length(devs)*ii, 2)
    end
    if length(devs) > 1
        device!(0)
        @test_throws DimensionMismatch NCCL.ReduceScatter!(
            CUDACore.zeros(length(devs) + 1), +, comms[1])
    end

    NCCL.destroy(comms)
end

@testset "Alltoall!" begin
    devs = collect(CUDACore.devices())
    comms = NCCL.Communicators(devs)
    n = length(devs)
    B = 3

    # the `k`-th element of the block that rank `src` sends to rank `dst`
    val(src, dst, k) = Float64(1000src + 10dst + k)
    sendbuf = [device!(() -> CuArray([val(r, j, k) for k in 1:B, j in 0:n-1]), dev)
               for (r, dev) in zip(0:n-1, devs)]
    recvbuf = [device!(() -> CUDACore.fill(-1.0, B, n), dev) for dev in devs]
    NCCL.group() do
        for i in 1:n
            @test NCCL.Alltoall!(sendbuf[i], recvbuf[i], comms[i]) === recvbuf[i]
        end
    end
    for (r, dev) in zip(0:n-1, devs)
        device!(dev) do
            @test Array(recvbuf[r+1]) == [val(i, r, k) for k in 1:B, i in 0:n-1]
        end
    end

    if n > 1
        # the buffers can't be divided over the ranks
        device!(devs[1]) do
            x = CUDACore.zeros(Float64, B * n + 1)
            y = CUDACore.zeros(Float64, B * n + 1)
            @test_throws DimensionMismatch NCCL.Alltoall!(x, y, comms[1])
        end
    end

    NCCL.destroy(comms)
end

@testset "Gather!" begin
    devs = collect(CUDACore.devices())
    comms = NCCL.Communicators(devs)
    n = length(devs)
    B = 4

    vals(r) = Float64[100r + k for k in 1:B]
    expected = reduce(vcat, vals.(0:n-1))
    for root in 0:n-1
        # the receive buffer is only needed on the root
        sendbuf = [device!(() -> CuArray(vals(r)), dev) for (r, dev) in zip(0:n-1, devs)]
        recvbuf = [r == root ? device!(() -> CUDACore.fill(-1.0, n * B), dev) : nothing
                   for (r, dev) in zip(0:n-1, devs)]
        NCCL.group() do
            for i in 1:n
                @test NCCL.Gather!(sendbuf[i], recvbuf[i], comms[i]; root) === recvbuf[i]
            end
        end
        device!(devs[root+1]) do
            @test Array(recvbuf[root+1]) == expected
        end

        # in place: the root sends the part of the receive buffer for itself
        recvbuf = device!(devs[root+1]) do
            init = fill(-1.0, n * B)
            init[root*B+1:(root+1)*B] = vals(root)
            CuArray(init)
        end
        NCCL.group() do
            for i in 1:n
                if i - 1 == root
                    NCCL.Gather!(view(recvbuf, root*B+1:(root+1)*B), recvbuf, comms[i]; root)
                else
                    NCCL.Gather!(sendbuf[i], nothing, comms[i]; root)
                end
            end
        end
        device!(devs[root+1]) do
            @test Array(recvbuf) == expected
        end
    end

    NCCL.destroy(comms)
end

@testset "Scatter!" begin
    devs = collect(CUDACore.devices())
    comms = NCCL.Communicators(devs)
    n = length(devs)
    B = 4

    vals(root) = Float64[1000root + k for k in 1:n*B]
    for root in 0:n-1
        # the send buffer is only needed on the root
        sendbuf = [r == root ? device!(() -> CuArray(vals(root)), dev) : nothing
                   for (r, dev) in zip(0:n-1, devs)]
        recvbuf = [device!(() -> CUDACore.fill(-1.0, B), dev) for dev in devs]
        NCCL.group() do
            for i in 1:n
                @test NCCL.Scatter!(sendbuf[i], recvbuf[i], comms[i]; root) === recvbuf[i]
            end
        end
        for (r, dev) in zip(0:n-1, devs)
            device!(dev) do
                @test Array(recvbuf[r+1]) == vals(root)[r*B+1:(r+1)*B]
            end
        end

        # in place: the root receives into its block of the send buffer
        sendbuf = device!(() -> CuArray(vals(root)), devs[root+1])
        recvbuf = [device!(() -> CUDACore.fill(-1.0, B), dev) for dev in devs]
        NCCL.group() do
            for i in 1:n
                if i - 1 == root
                    NCCL.Scatter!(sendbuf, view(sendbuf, root*B+1:(root+1)*B), comms[i];
                                  root)
                else
                    NCCL.Scatter!(nothing, recvbuf[i], comms[i]; root)
                end
            end
        end
        for (r, dev) in zip(0:n-1, devs)
            r == root && continue
            device!(dev) do
                @test Array(recvbuf[r+1]) == vals(root)[r*B+1:(r+1)*B]
            end
        end
        device!(devs[root+1]) do
            @test Array(sendbuf) == vals(root)
        end
    end

    NCCL.destroy(comms)
end

@testset "Send/Recv" begin
    devs  = CUDACore.devices()
    comms = NCCL.Communicators(devs)
    recvbuf = Vector{CuVector{Float64}}(undef, length(devs))
    sendbuf = Vector{CuVector{Float64}}(undef, length(devs))
    N = 512
    for (ii, dev) in enumerate(devs)
        CUDACore.device!(ii - 1)
        sendbuf[ii] = CuArray(fill(Float64(ii), N))
        recvbuf[ii] = CUDACore.zeros(Float64, N)
    end

    NCCL.group() do
        for ii in 1:length(devs)
            comm = comms[ii]
            dest = mod(NCCL.rank(comm)+1, NCCL.nranks(comm))
            source = mod(NCCL.rank(comm)-1, NCCL.nranks(comm))
            @test NCCL.Send(sendbuf[ii], comm; dest) === nothing
            @test NCCL.Recv!(recvbuf[ii], comm; source) === recvbuf[ii]
        end
    end
    for (ii, dev) in enumerate(devs)
        answer = mod1(ii - 1, length(devs))
        device!(ii - 1)
        crecv = collect(recvbuf[ii])
        @test all(crecv .== answer)
    end

    NCCL.destroy(comms)
end

@testset "element types" begin
    devs = collect(CUDACore.devices())
    comms = NCCL.Communicators(devs)
    n = length(devs)
    N = 64

    # one buffer per rank, with the contents `f(rank)`
    onranks(f) = [device!(() -> CuArray(f(r)), dev) for (r, dev) in zip(0:n-1, devs)]
    fetchall(bufs) = [device!(() -> Array(buf), dev) for (buf, dev) in zip(bufs, devs)]

    function allreduce(f, op)
        send = onranks(f)
        recv = onranks(r -> similar(f(r)))
        NCCL.group() do
            for i in 1:n
                NCCL.Allreduce!(send[i], recv[i], op, comms[i])
            end
        end
        fetchall(recv)
    end

    @testset "$T" for T in (Float16, BFloat16)
        vals(r) = T[r + i % 8 for i in 1:N]
        expected = sum(vals, 0:n-1)
        @test all(==(expected), allreduce(vals, +))
        @test all(==(expected / n), allreduce(vals, NCCL.avg))
    end

    @testset "integer average" begin
        vals(r) = Int32[(r + 1) * (isodd(i) ? -1 : 1) for i in 1:N]
        # sums are divided with truncation towards zero
        expected = div.(sum(vals, 0:n-1), n)
        @test all(==(expected), allreduce(vals, NCCL.avg))
    end

    @testset "Complex{$T}" for T in (Float16, Float32, Float64, BFloat16)
        # distinct real and imaginary parts, to catch them being mixed up
        vals(r) = [Complex{T}(r + 1, i) for i in 1:N]
        expected = sum(vals, 0:n-1)
        @test all(==(expected), allreduce(vals, +))
        @test all(==(expected / n), allreduce(vals, NCCL.avg))

        send = onranks(r -> r == 0 ? vals(r) : zero(vals(r)))
        recv = onranks(r -> zero(vals(r)))
        NCCL.group() do
            for i in 1:n
                NCCL.Broadcast!(send[i], recv[i], comms[i]; root=0)
            end
        end
        @test all(==(vals(0)), fetchall(recv))

        send = onranks(vals)
        recv = onranks(r -> zeros(Complex{T}, n * N))
        NCCL.group() do
            for i in 1:n
                NCCL.Allgather!(send[i], recv[i], comms[i])
            end
        end
        @test all(==(reduce(vcat, vals.(0:n-1))), fetchall(recv))
    end

    @testset "Bool" begin
        # every combination of values over (up to) 6 ranks
        vals(r) = Bool[(i >> r) & 1 for i in 0:N-1]
        @test all(==(reduce(.|, vals.(0:n-1))), allreduce(vals, max))
        @test all(==(reduce(.&, vals.(0:n-1))), allreduce(vals, min))

        send = onranks(vals)
        recv = onranks(r -> zeros(Bool, n * N))
        NCCL.group() do
            for i in 1:n
                NCCL.Allgather!(send[i], recv[i], comms[i])
            end
        end
        @test all(==(reduce(vcat, vals.(0:n-1))), fetchall(recv))
    end

    # other isbits types are copied as bytes
    @testset "$T" for (T, val) in ((Int16, (r, i) -> Int16(-1000r - i)),
                                   (Char, (r, i) -> 'α' + 10r + i),
                                   (NTuple{3,UInt8}, (r, i) -> (UInt8(r), UInt8(i), 0xff)),
                                   (Particle, (r, i) -> Particle((r, i, -r), i)),
                                   (UInt24, (r, i) -> UInt24(65536r + i)))
        vals(r) = T[val(r, i) for i in 1:N]

        send = onranks(r -> r == 1 % n ? vals(r) : vals(n))
        recv = onranks(r -> vals(n))
        NCCL.group() do
            for i in 1:n
                NCCL.Broadcast!(send[i], recv[i], comms[i]; root=1 % n)
            end
        end
        @test all(==(vals(1 % n)), fetchall(recv))

        send = onranks(vals)
        recv = onranks(r -> repeat(vals(n), n))
        NCCL.group() do
            for i in 1:n
                NCCL.Allgather!(send[i], recv[i], comms[i])
            end
        end
        @test all(==(reduce(vcat, vals.(0:n-1))), fetchall(recv))

        send = onranks(vals)
        recv = onranks(r -> vals(n))
        NCCL.group() do
            for i in 1:n
                NCCL.Send(send[i], comms[i]; dest=mod(i, n))
                NCCL.Recv!(recv[i], comms[i]; source=mod(i - 2, n))
            end
        end
        @test fetchall(recv) == [vals(mod(r - 1, n)) for r in 0:n-1]
    end

    NCCL.destroy(comms)
end

# keep a stream busy for a while
function spin(cycles)
    start = CUDACore.clock(UInt64)
    while CUDACore.clock(UInt64) - start < cycles
    end
    return
end

@testset "streams" begin
    devs = collect(CUDACore.devices())
    dev = last(devs)
    comm = Communicator(1, 0, NCCL.UniqueID(); device=dev)
    N = 512

    device!(dev) do
        sendbuf = CuArray(fill(42.0, N))
        recvbuf = CUDACore.zeros(Float64, N)

        s = CuStream()
        NCCL.Allreduce!(sendbuf, recvbuf, +, comm; stream=s)
        synchronize(s)
        @test CUDACore.isdone(s)
        @test Array(recvbuf) == fill(42.0, N)

        # the operation runs on `s`, after the kernel. if the buffers were considered to be
        # used by the task's stream, reading them on that stream would not wait for it.
        recvbuf .= 0
        synchronize()
        @cuda stream=s spin(UInt64(1) << 30)
        @test !CUDACore.isdone(s)
        @test NCCL.Allreduce!(sendbuf, recvbuf, +, comm; stream=s) === recvbuf
        @test Array(recvbuf) == fill(42.0, N)

        # the default streams aren't bound to a context
        recvbuf .= 0
        NCCL.Allreduce!(sendbuf, recvbuf, +, comm; stream=CUDACore.default_stream())
        @test Array(recvbuf) == fill(42.0, N)
        recvbuf .= 0
        NCCL.Allreduce!(sendbuf, recvbuf, +, comm; stream=legacy_stream())
        @test Array(recvbuf) == fill(42.0, N)

        # the per-thread default stream differs between the threads a task can run on,
        # whether it is passed explicitly or is the task's stream
        @test_throws ArgumentError NCCL.Allreduce!(sendbuf, recvbuf, +, comm;
                                                   stream=per_thread_stream())
        stream!(per_thread_stream()) do
            @test_throws ArgumentError NCCL.Allreduce!(sendbuf, recvbuf, +, comm)
        end
    end

    # an operation can be issued while another device is active
    device!(first(devs)) do
        sendbuf, recvbuf = device!(dev) do
            CuArray(fill(1.0, N)), CUDACore.zeros(Float64, N)
        end
        s = device!(() -> CuStream(), dev)
        NCCL.Allreduce!(sendbuf, recvbuf, +, comm; stream=s)
        @test CUDACore.device() == first(devs)
        device!(dev) do
            @test Array(recvbuf) == fill(1.0, N)
        end

        # a stream on another device
        if length(devs) > 1
            other = CuStream()
            @test_throws ArgumentError NCCL.Allreduce!(sendbuf, recvbuf, +, comm;
                                                       stream=other)
        end
    end

    NCCL.destroy(comm)
end

@testset "buffers" begin
    devs = collect(CUDACore.devices())
    dev = first(devs)
    # with a single rank, operations that are valid complete immediately
    comm = Communicator(1, 0, NCCL.UniqueID(); device=dev)

    device!(dev) do
        x = CuArray(Float64.(1:20))
        y = CUDACore.zeros(Float64, 20)
        m = CuArray(reshape(Float64.(1:20), 4, 5))

        @testset "contiguous views" begin
            NCCL.Allreduce!(view(x, 3:10), view(y, 11:18), +, comm)
            @test Array(y)[11:18] == 3:10
            NCCL.Broadcast!(view(m, :, 2), view(y, 1:4), comm)
            @test Array(y)[1:4] == 5:8
            NCCL.Allgather!(reshape(view(m, :, 2:3), 2, 4), view(y, 13:20), comm)
            @test Array(y)[13:20] == 5:12
            NCCL.Alltoall!(view(m, :, 3), view(y, 1:4), comm)
            @test Array(y)[1:4] == 9:12
            NCCL.Gather!(view(m, :, 4), view(y, 5:8), comm)
            @test Array(y)[5:8] == 13:16
            NCCL.Scatter!(view(m, :, 5), view(y, 9:12), comm)
            @test Array(y)[9:12] == 17:20
        end

        @testset "non-contiguous views" begin
            @test_throws ArgumentError NCCL.Allreduce!(view(x, 1:2:10), view(y, 1:5), +, comm)
            @test_throws ArgumentError NCCL.Allreduce!(view(y, 1:5), view(x, 1:2:10), +, comm)
            @test_throws ArgumentError NCCL.Allreduce!(view(x, 1:2:10), +, comm)
            @test_throws ArgumentError NCCL.Broadcast!(view(m, 1:2, 2:3), comm)
            @test_throws ArgumentError NCCL.Send(view(m, 2, :), comm; dest=0)
            @test_throws ArgumentError NCCL.Recv!(view(m, 2, :), comm; source=0)
            @test_throws ArgumentError NCCL.Alltoall!(view(m, 2, :), view(y, 1:5), comm)
            @test_throws ArgumentError NCCL.Gather!(view(m, 2, :), view(y, 1:5), comm)
            @test_throws ArgumentError NCCL.Scatter!(view(y, 1:5), view(m, 2, :), comm)
            err = try
                NCCL.Reduce!(view(m, 1:2, 2:3), +, comm)
            catch err
                err
            end
            @test err isa ArgumentError
            @test occursin("contiguous", err.msg)

            # host memory
            @test_throws ArgumentError NCCL.Allreduce!(Array(x), y, +, comm)
        end

        @testset "element types" begin
            @test_throws ArgumentError NCCL.Allreduce!(x, CUDACore.zeros(Float32, 20), +, comm)
            @test_throws ArgumentError NCCL.Broadcast!(x, CUDACore.zeros(Int64, 20), comm)

            function argument_error(f)
                try
                    f()
                catch err
                    err isa ArgumentError && return err.msg
                    rethrow()
                end
                return nothing
            end

            # operators
            @test argument_error(() -> NCCL.Allreduce!(x, -, comm)) !== nothing

            # complex numbers are reduced component-wise
            z = CUDACore.zeros(ComplexF32, 4)
            for op in (*, max, min)
                for f in (() -> NCCL.Allreduce!(z, op, comm),
                          () -> NCCL.Reduce!(z, op, comm),
                          () -> NCCL.ReduceScatter!(z, z, op, comm))
                    msg = argument_error(f)
                    @test msg !== nothing && occursin("real and imaginary parts", msg)
                end
            end
            NCCL.Allreduce!(z, +, comm)
            NCCL.Reduce!(z, NCCL.avg, comm)

            # booleans are only reduced with logical operators
            b = CuArray([true, false, true])
            for op in (+, *, NCCL.avg)
                msg = argument_error(() -> NCCL.Allreduce!(b, op, comm))
                @test msg !== nothing && occursin("logical or", msg)
            end
            NCCL.Allreduce!(b, max, comm)
            NCCL.Allreduce!(b, min, comm)
            @test Array(b) == [true, false, true]

            # isbits unions, even of supported types, store their type tags separately
            u8 = CuArray{Union{Int8,UInt8}}([Int8(1), 0x2])
            msg = argument_error(() -> NCCL.Allreduce!(u8, +, comm))
            @test msg !== nothing && occursin("isbits", msg)
            msg = argument_error(() -> NCCL.Broadcast!(u8, comm))
            @test msg !== nothing && occursin("isbits", msg)

            # other element types can't be reduced
            i16 = CUDACore.zeros(Int16, 4)
            for op in (+, max)
                @test argument_error(() -> NCCL.Allreduce!(i16, op, comm)) !== nothing
                @test argument_error(() -> NCCL.Reduce!(i16, op, comm)) !== nothing
                @test argument_error(() -> NCCL.ReduceScatter!(i16, i16, op, comm)) !== nothing
            end
            c64 = CUDACore.zeros(Complex{Int64}, 4)
            @test argument_error(() -> NCCL.Allreduce!(c64, +, comm)) !== nothing
            p = CuArray([Particle((1, 2, 3), 4)])
            msg = argument_error(() -> NCCL.Allreduce!(p, +, comm))
            @test msg !== nothing && occursin("can only be copied", msg)

            # but they can be copied
            NCCL.Broadcast!(i16, comm)
            NCCL.Allgather!(p, p, comm)
            @test Array(p) == [Particle((1, 2, 3), 4)]
            q = CuArray([Particle((0, 0, 0), 0)])
            NCCL.Alltoall!(p, q, comm)
            @test Array(q) == [Particle((1, 2, 3), 4)]
            NCCL.Gather!(p, q, comm)
            NCCL.Scatter!(p, q, comm)

            # types that aren't isbits, or have no data
            u = CuArray{Union{Int32,Nothing}}([1, nothing])
            msg = argument_error(() -> NCCL.Broadcast!(u, comm))
            @test msg !== nothing && occursin("isbits", msg)
            msg = argument_error(() -> NCCL.Alltoall!(u, similar(u), comm))
            @test msg !== nothing && occursin("isbits", msg)
            msg = argument_error(() -> NCCL.Send(CuArray([nothing, nothing]), comm; dest=0))
            @test msg !== nothing && occursin("no data", msg)
        end

        @testset "root-only buffers" begin
            # `nothing` is only accepted on other ranks than the root
            err = try
                NCCL.Broadcast!(nothing, y, comm)
            catch err
                err
            end
            @test err isa ArgumentError
            @test occursin("root", err.msg)
            @test_throws ArgumentError NCCL.Reduce!(x, nothing, +, comm)
            @test_throws ArgumentError NCCL.Gather!(x, nothing, comm)
            @test_throws ArgumentError NCCL.Scatter!(nothing, y, comm)
            # and only for the buffers that are ignored there
            @test_throws ArgumentError NCCL.Gather!(nothing, y, comm)
            @test_throws ArgumentError NCCL.Scatter!(x, nothing, comm)
            @test_throws ArgumentError NCCL.Alltoall!(nothing, y, comm)
            @test_throws ArgumentError NCCL.Broadcast!(x, nothing, comm)
            @test_throws ArgumentError NCCL.Reduce!(nothing, y, +, comm)
            @test_throws ArgumentError NCCL.Allreduce!(nothing, y, +, comm)
        end

        @testset "sizes" begin
            @test_throws DimensionMismatch NCCL.Allreduce!(x, view(y, 1:10), +, comm)
            @test_throws DimensionMismatch NCCL.Broadcast!(x, view(y, 1:10), comm)
            @test_throws DimensionMismatch NCCL.Reduce!(x, view(y, 1:10), +, comm)
            @test_throws DimensionMismatch NCCL.Allgather!(x, view(y, 1:10), comm)
            @test_throws DimensionMismatch NCCL.ReduceScatter!(view(x, 1:10), y, +, comm)
            @test_throws DimensionMismatch NCCL.Alltoall!(x, view(y, 1:10), comm)
            @test_throws DimensionMismatch NCCL.Gather!(x, view(y, 1:10), comm)
            @test_throws DimensionMismatch NCCL.Scatter!(view(x, 1:10), y, comm)
        end

        @testset "ranks" begin
            for root in (-1, 1)
                @test_throws ArgumentError NCCL.Broadcast!(x, y, comm; root)
                @test_throws ArgumentError NCCL.Reduce!(x, y, +, comm; root)
                @test_throws ArgumentError NCCL.Broadcast!(nothing, y, comm; root)
                @test_throws ArgumentError NCCL.Reduce!(x, nothing, +, comm; root)
                @test_throws ArgumentError NCCL.Gather!(x, y, comm; root)
                @test_throws ArgumentError NCCL.Scatter!(x, y, comm; root)
            end
            @test_throws ArgumentError NCCL.Send(x, comm; dest=1)
            @test_throws ArgumentError NCCL.Recv!(y, comm; source=-1)
        end

        @testset "overlap" begin
            # partially overlapping buffers
            @test_throws ArgumentError NCCL.Allreduce!(view(x, 1:10), view(x, 5:14), +, comm)
            @test_throws ArgumentError NCCL.Broadcast!(view(x, 5:14), view(x, 1:10), comm)
            @test_throws ArgumentError NCCL.Reduce!(view(x, 1:10), view(x, 2:11), +, comm)
            @test_throws ArgumentError NCCL.Allgather!(view(x, 2:11), view(x, 1:10), comm)
            @test_throws ArgumentError NCCL.ReduceScatter!(view(x, 1:10), view(x, 2:11), +,
                                                           comm)
            @test_throws ArgumentError NCCL.Gather!(view(x, 2:11), view(x, 1:10), comm)
            @test_throws ArgumentError NCCL.Scatter!(view(x, 1:10), view(x, 2:11), comm)

            # all-to-all can't be performed in place
            err = try
                NCCL.Alltoall!(x, x, comm)
            catch err
                err
            end
            @test err isa ArgumentError
            @test occursin("in place", err.msg)
            @test_throws ArgumentError NCCL.Alltoall!(view(x, 1:10), view(x, 5:14), comm)

            # adjacent buffers
            NCCL.Allreduce!(view(x, 1:10), view(x, 11:20), +, comm)
            @test Array(x) == vcat(1:10, 1:10)

            # in place
            NCCL.Allreduce!(view(x, 1:10), view(x, 1:10), +, comm)
            NCCL.Allgather!(view(x, 1:10), view(x, 1:10), comm)
            NCCL.ReduceScatter!(view(x, 1:10), view(x, 1:10), +, comm)
            NCCL.Gather!(view(x, 1:10), view(x, 1:10), comm)
            NCCL.Scatter!(view(x, 1:10), view(x, 1:10), comm)
            @test Array(x) == vcat(1:10, 1:10)
            NCCL.Alltoall!(view(x, 1:10), view(x, 11:20), comm)
            @test Array(x) == vcat(1:10, 1:10)
            @test NCCL.Allgather!(x, comm) === x
            @test NCCL.ReduceScatter!(x, +, comm) == x
            @test Array(x) == vcat(1:10, 1:10)
        end

        if length(devs) > 1
            @testset "devices" begin
                other = device!(() -> CUDACore.zeros(Float64, 20), devs[2])
                err = try
                    NCCL.Allreduce!(x, other, +, comm)
                catch err
                    err
                end
                @test err isa ArgumentError
                @test occursin("device", err.msg)
                @test_throws ArgumentError NCCL.Allreduce!(other, x, +, comm)
                @test_throws ArgumentError NCCL.Send(other, comm; dest=0)
            end
        end
    end

    NCCL.destroy(comm)
end


@testset "struct layout" begin
    # offsets and sizes from compiling a C program that prints `sizeof` and `offsetof` of
    # these structs, against the header the bindings were generated from (NCCL 2.32.3).
    layouts = [
        NCCL.LibNCCL.ncclConfig_t => (120, [0, 8, 12, 16, 20, 24, 28, 32, 40, 44, 48, 56,
                                            60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100,
                                            104, 108, 112]),
        NCCL.LibNCCL.ncclCollConfig_t => (72, [0, 8, 12, 16, 24, 28, 32, 36, 40, 48, 52,
                                               56, 64]),
        NCCL.LibNCCL.ncclSimInfo_t => (24, [0, 8, 12, 16]),
        NCCL.LibNCCL.ncclEncryptionConfig_t => (32, [0, 8, 12, 16, 24]),
        NCCL.LibNCCL.ncclWaitSignalDesc_t => (16, [0, 4, 8, 12]),
        NCCL.LibNCCL.ncclConfigExt_t => (24, nothing),
        NCCL.LibNCCL.ncclUniqueId => (128, nothing),
    ]
    for (T, (size, offsets)) in layouts
        @test sizeof(T) == size
        if offsets !== nothing
            @test fieldcount(T) == length(offsets)
            @test [fieldoffset(T, i) for i in 1:fieldcount(T)] == offsets
        end
    end
end

@testset "errors" begin
    function nccl_error(f)
        try
            f()
        catch err
            err
        end
    end

    # sending to or receiving from oneself without a matching operation is invalid, which
    # NCCL reports (with different messages, so that the second one is new to NCCL.jl)
    comm = Communicator(1, 0, NCCL.UniqueID())
    buf = CUDACore.ones(Float64, 4)
    err = nccl_error(() -> NCCL.Recv!(buf, comm; source=0))
    @test err isa NCCL.NCCLError
    err = nccl_error(() -> NCCL.Send(buf, comm; dest=0))
    @test err isa NCCL.NCCLError
    @test !isempty(err.msg)
    msg = sprint(showerror, err)
    @test occursin(string(err.code), msg)
    @test occursin(err.msg, msg)
    NCCL.destroy(comm)

    # an error NCCL didn't log must not carry the message of the previous one
    err = NCCL.NCCLError(err.code)
    @test isempty(err.msg)
    @test !occursin('\n', sprint(showerror, err))
end

@testset "lifetime" begin
    # destroying communicators synchronizes with their peers, and can deadlock
    script = raw"""
        using Test, CUDACore, NCCL

        devs = collect(CUDACore.devices())
        N = 512

        function allreduce(comms)
            bufs = map(devs) do dev
                device!(() -> CUDACore.ones(Float64, N), dev)
            end
            NCCL.group() do
                for (buf, comm) in zip(bufs, comms)
                    NCCL.Allreduce!(buf, +, comm)
                end
            end
            all(zip(devs, bufs)) do (dev, buf)
                device!(() -> all(==(length(devs)), Array(buf)), dev)
            end
        end

        @testset "lifetime" begin

        @testset "destroy" begin
            comms = NCCL.Communicators(devs)
            @test all(isopen, comms)
            @test allreduce(comms)
            NCCL.destroy(comms)
            @test !any(isopen, comms)

            # destroying again does nothing
            NCCL.destroy(comms)
            NCCL.destroy(comms[1])

            # closed communicators can't be used
            buf = device!(() -> CUDACore.ones(Float64, N), devs[1])
            @test_throws InvalidStateException NCCL.Allreduce!(buf, +, comms[1])
            @test_throws InvalidStateException NCCL.Send(buf, comms[1]; dest=0)
            @test_throws InvalidStateException NCCL.group() do
                NCCL.Allreduce!(buf, +, comms[1])
            end
            # the group was ended: another task can open one
            @test fetch(@async NCCL.group(() -> true))

            # a vector with closed and open communicators
            comms = NCCL.Communicators(devs)
            more = NCCL.Communicators(devs)
            NCCL.destroy(comms)
            NCCL.destroy([comms; more])
            @test !any(isopen, more)

            # duplicates are destroyed once
            comms = NCCL.Communicators(devs)
            NCCL.destroy([comms; reverse(comms)])
            @test !any(isopen, comms)
            comms = NCCL.Communicators(devs)
            NCCL.abort([comms; comms])
            @test !any(isopen, comms)
        end

        @testset "construction" begin
            # device ordinals
            comms = NCCL.Communicators(0:length(devs)-1)
            @test NCCL.device.(comms) == devs
            NCCL.destroy(comms)

            # with a configuration, using grouped ncclCommInitRankConfig
            config = NCCL.Config(comm_name="clique", max_ctas=4)
            comms = NCCL.Communicators(reverse(devs); config)
            @test NCCL.device.(comms) == reverse(devs)
            @test NCCL.rank.(comms) == 0:length(devs)-1
            @test all(==(length(devs)), NCCL.nranks.(comms))
            @test allreduce(reverse(comms))
            NCCL.destroy(comms)

            @test_throws ArgumentError NCCL.Communicators([devs[1], devs[1]])
            @test_throws ArgumentError NCCL.Communicators([0, 0]; config)
            @test_throws ArgumentError NCCL.Communicators(CuDevice[])
        end

        @testset "abort" begin
            comms = NCCL.Communicators(devs)
            @test allreduce(comms)
            NCCL.abort(comms)
            @test !any(isopen, comms)
            NCCL.abort(comms)
            NCCL.destroy(comms)

            comm = only(NCCL.Communicators(devs[1:1]))
            NCCL.abort(comm)
            @test !isopen(comm)
        end

        @testset "inside a group" begin
            comms = NCCL.Communicators(devs)
            @test_throws ArgumentError NCCL.group(() -> NCCL.destroy(comms))
            @test_throws ArgumentError NCCL.group(() -> NCCL.abort(comms[1]))
            @test all(isopen, comms)
            NCCL.destroy(comms)
        end

        @testset "do-block" begin
            # normal return: destroyed
            local escaped
            ret = NCCL.Communicators(devs) do comms
                escaped = comms
                allreduce(comms)
            end
            @test ret
            @test !any(isopen, escaped)

            # exception: aborted, and the exception is rethrown
            @test_throws ErrorException("oops") NCCL.Communicators(devs) do comms
                escaped = comms
                @assert allreduce(comms)
                error("oops")
            end
            @test !any(isopen, escaped)

            # the original exception is preserved if aborting fails as well (here, because
            # it happens inside a group)
            @test_throws ErrorException("inner") NCCL.group() do
                NCCL.Communicators(devs) do comms
                    escaped = comms
                    error("inner")
                end
            end
            @test all(isopen, escaped)
            @test allreduce(escaped)
            NCCL.destroy(escaped)
        end

        end
        """
    @test run_script(script; timeout=300)
end

@testset "UniqueID" begin
    id = NCCL.UniqueID()
    @test isbitstype(NCCL.UniqueID)
    bytes = Vector{UInt8}(id)
    @test length(bytes) == 128
    @test NCCL.UniqueID(bytes) == id
    @test NCCL.UniqueID(view(bytes, :)) == id
    @test NCCL.UniqueID(bytes) != NCCL.UniqueID()
    @test_throws ArgumentError NCCL.UniqueID(bytes[1:127])
    @test_throws ArgumentError NCCL.UniqueID([bytes; 0x00])
    @test occursin(bytes2hex(bytes), sprint(show, id))
end

@testset "Config" begin
    config = NCCL.Config(comm_name="test", net_name="Socket", cga_cluster_size=2,
                         min_ctas=1, max_ctas=8, nvls_ctas=16,
                         cta_policy=NCCL.LibNCCL.NCCL_CTA_POLICY_EFFICIENCY, split_share=true,
                         traffic_class=3, collnet_enable=false, n_channels_per_net_peer=4,
                         max_p2p_peers=5, graph_usage_mode=1)
    @test sprint(show, NCCL.Config(comm_name="test", min_ctas=1)) ==
          "NCCL.Config(comm_name=\"test\", min_ctas=1)"
    @test_throws InexactError NCCL.Config(min_ctas=typemax(Int))
    @test_throws ArgumentError Communicator(1, 0, NCCL.UniqueID();
                                            config=NCCL.Config(comm_name="a\0b"))

    # a configured single-rank communicator
    comm = Communicator(1, 0, NCCL.UniqueID(); config)
    buf = CUDACore.ones(Float64, 4)
    NCCL.Allreduce!(buf, +, comm)
    @test Array(buf) == ones(4)
    NCCL.destroy(comm)
end

@testset "nonblocking communicators" begin
    # values that NCCL reads as 0 (old versions also read values without digits as 0)
    for str in ["0", "0trailing", " 0", "-0", "0x0", "000", "abc", "4294967296"]
        withenv("NCCL_COMM_BLOCKING" => str) do
            @test_throws ErrorException Communicator(1, 0, NCCL.UniqueID())
        end
    end
    withenv("NCCL_COMM_BLOCKING" => "0") do
        @test_throws ErrorException NCCL.Communicators(CUDACore.devices())
    end
    # values that NCCL ignores (out of range or other than 0 and 1), or reads as 1
    for str in ["", "1", "1trailing", "2", "99999999999999999999"]
        withenv("NCCL_COMM_BLOCKING" => str) do
            @test Communicator(comm -> isopen(comm), 1, 0, NCCL.UniqueID())
        end
    end

    # NCCL caches the setting, so the following need a fresh process
    script = """
        using CUDACore, NCCL, Test
        @test_throws ErrorException Communicator(1, 0, NCCL.UniqueID())
        @test_throws ErrorException NCCL.Communicators(CUDACore.devices())
        """
    @test run_script(script; env=("NCCL_COMM_BLOCKING" => "0",))
    @test run_script(script; env=("NCCL_COMM_BLOCKING" => "0trailing",))

    # NCCL also reads the setting from configuration files, but does not let these override
    # the environment, where NCCL.jl sets it when it isn't set yet
    mktempdir() do dir
        conf = joinpath(dir, "nccl.conf")
        write(conf, "NCCL_COMM_BLOCKING=0\n")
        @test run_script("""
            using CUDACore, NCCL, Test
            @test ENV["NCCL_COMM_BLOCKING"] == "1"
            comm = Communicator(1, 0, NCCL.UniqueID())
            buf = CUDACore.ones(Float64, 4)
            NCCL.Allreduce!(buf, +, comm)
            @test Array(buf) == ones(4)
            NCCL.destroy(comm)
            """; env=("NCCL_CONF_FILE" => conf, "NCCL_COMM_BLOCKING" => nothing))

        # also when the variable is unset after loading NCCL.jl. NCCL applies the file at
        # the first call that reads a setting, so the child must not call NCCL before
        # creating its communicator: it uses an id created here.
        id_file = joinpath(dir, "id")
        write(id_file, Vector{UInt8}(NCCL.UniqueID()))
        @test run_script("""
            using CUDACore, NCCL, Test
            delete!(ENV, "NCCL_COMM_BLOCKING")
            comm = Communicator(1, 0, NCCL.UniqueID(read($(repr(id_file)))))
            @test ENV["NCCL_COMM_BLOCKING"] == "1"
            buf = CUDACore.ones(Float64, 4)
            NCCL.Allreduce!(buf, +, comm)
            @test Array(buf) == ones(4)
            NCCL.destroy(comm)
            """; env=("NCCL_CONF_FILE" => conf, "NCCL_COMM_BLOCKING" => nothing))
    end
end

@testset "failed initialization" begin
    # a network module that doesn't exist makes NCCL fail to initialize the communicators
    devs = collect(CUDACore.devices())
    config = NCCL.Config(net_name="nonexistent")
    @test_throws NCCL.NCCLError Communicator(1, 0, NCCL.UniqueID(); config)
    @test_throws NCCL.NCCLError NCCL.Communicators(devs; config)

    # this doesn't affect communicators created afterwards
    @test NCCL.Communicators(comms -> all(isopen, comms), devs)
end

if length(CUDACore.devices()) >= 2
@testset "multiple processes" begin
    nranks = min(length(CUDACore.devices()), 4)

    # construction, properties, an allreduce, and destruction
    @test run_ranks("""
        comm = Communicator(nranks, rank, id; config=NCCL.Config(comm_name="test"))
        @test isopen(comm)
        @test NCCL.rank(comm) == rank
        @test NCCL.nranks(comm) == nranks
        @test NCCL.device(comm) == CuDevice(0)
        @test length(CUDACore.devices()) == 1

        buf = CuArray(fill(Float64(rank + 1), 1024))
        NCCL.Allreduce!(buf, +, comm)
        @test all(==(nranks * (nranks + 1) / 2), Array(buf))

        NCCL.destroy(comm)
        @test !isopen(comm)
        NCCL.destroy(comm)
        @test_throws InvalidStateException NCCL.Allreduce!(buf, +, comm)
        """, nranks)

    # a struct, sent as bytes between processes
    @test run_ranks("""
        struct Particle
            position::NTuple{3,Float32}
            id::Int16
        end
        comm = Communicator(nranks, rank, id)
        particles(r) = [Particle((r, i, -i), i) for i in 1:100]
        if rank == 0
            NCCL.Send(CuArray(particles(1)), comm; dest=1)
            buf = CuArray(particles(0))
            NCCL.Recv!(buf, comm; source=1)
            @test Array(buf) == particles(2)
        elseif rank == 1
            buf = CuArray(particles(0))
            NCCL.Recv!(buf, comm; source=0)
            @test Array(buf) == particles(1)
            NCCL.Send(CuArray(particles(2)), comm; dest=0)
        end
        NCCL.destroy(comm)
        """, 2)

    # an all-to-all exchange, with distinct data for every pair of ranks
    @test run_ranks("""
        comm = Communicator(nranks, rank, id)
        val(src, dst, k) = Int32(1000src + 10dst + k)
        sendbuf = CuArray([val(rank, j, k) for k in 1:5, j in 0:nranks-1])
        recvbuf = CUDACore.zeros(Int32, 5, nranks)
        NCCL.Alltoall!(sendbuf, recvbuf, comm)
        @test Array(recvbuf) == [val(i, rank, k) for k in 1:5, i in 0:nranks-1]
        NCCL.destroy(comm)
        """, nranks)

    # buffers that are only used on the root, with NCCL checking the pointers it is passed
    # (NCCL only accepts null pointers for the ignored buffers of broadcasts and reductions)
    @test run_ranks("""
        ENV["NCCL_CHECK_POINTERS"] = "1"
        comm = Communicator(nranks, rank, id)
        root = nranks - 1
        isroot = rank == root
        data = CuArray(fill(Float64(rank + 1), 8))

        buf = CUDACore.zeros(Float64, 8)
        NCCL.Broadcast!(isroot ? data : nothing, buf, comm; root)
        @test Array(buf) == fill(nranks, 8)

        buf = isroot ? CUDACore.zeros(Float64, 8) : nothing
        NCCL.Reduce!(data, buf, +, comm; root)
        isroot && @test Array(buf) == fill(nranks * (nranks + 1) / 2, 8)

        buf = isroot ? CUDACore.zeros(Float64, 8nranks) : nothing
        NCCL.Gather!(data, buf, comm; root)
        isroot && @test Array(buf) == repeat(1.0:nranks, inner=8)

        whole = isroot ? CuArray(repeat(1.0:nranks, inner=8)) : nothing
        buf = CUDACore.zeros(Float64, 8)
        NCCL.Scatter!(whole, buf, comm; root)
        @test Array(buf) == fill(rank + 1, 8)
        @test Array(data) == fill(rank + 1, 8)
        NCCL.destroy(comm)
        """, nranks)

    # do-block: destroyed on return
    @test run_ranks("""
        escaped = Ref{Communicator}()
        ret = Communicator(nranks, rank, id) do comm
            escaped[] = comm
            buf = CuArray(fill(Float64(rank + 1), 1024))
            NCCL.Allreduce!(buf, +, comm)
            Array(buf)
        end
        @test all(==(nranks * (nranks + 1) / 2), ret)
        @test !isopen(escaped[])
        """, nranks)

    # every rank fails after a collective
    @test run_ranks("""
        escaped = Ref{Communicator}()
        @test_throws ErrorException("failure on rank \$rank") begin
            Communicator(nranks, rank, id) do comm
                escaped[] = comm
                buf = CuArray(fill(Float64(rank + 1), 1024))
                NCCL.Allreduce!(buf, +, comm)
                synchronize()
                error("failure on rank \$rank")
            end
        end
        @test !isopen(escaped[])
        """, nranks)

    # abort a communicator with a collective that never completes, because the other ranks
    # don't take part in it
    @test run_ranks("""
        comm = Communicator(nranks, rank, id)
        buf = CUDACore.ones(Float64, 1024)
        NCCL.Allreduce!(buf, +, comm)
        synchronize()
        queued = joinpath(dirname(ARGS[3]), "queued")
        if rank == 0
            NCCL.Allreduce!(buf, +, comm)
            sleep(1)
            @test !CUDACore.isdone(stream())
            touch(queued)
        else
            while !isfile(queued)
                sleep(0.1)
            end
        end
        NCCL.abort(comm)
        @test !isopen(comm)
        # the collective was cancelled (a hang here is caught by the watchdog)
        synchronize()
        @test CUDACore.isdone(stream())
        """, nranks)
end
else
    @info "Skipping the tests with multiple processes, which need 2 GPUs"
    @test_skip "multiple processes (needs 2 GPUs)"
end

@testset "large count" begin
    # counts used to be passed as `Cint`, and failed beyond `typemax(Cint)`
    script = """
        using CUDACore, NCCL, Test
        n = 2^31 + 8
        if CUDACore.free_memory() < 2n + 2^30
            @warn "Not enough GPU memory, skipping the large count test"
            exit(0)
        end
        comm = Communicator(1, 0, NCCL.UniqueID())
        sendbuf = CUDACore.fill(0x01, n)
        recvbuf = CUDACore.zeros(UInt8, n)
        NCCL.Allreduce!(sendbuf, recvbuf, +, comm)
        @test Array(recvbuf[1:8]) == fill(0x01, 8)
        @test Array(recvbuf[end-7:end]) == fill(0x01, 8)
        NCCL.destroy(comm)
        """
    @test run_script(script; timeout=300)
end

@testset "groups" begin
    # a broken group can deadlock, so run these in a separate process
    script = raw"""
        using Test, CUDACore, NCCL

        devs = collect(CUDACore.devices())
        comms = NCCL.Communicators(devs)
        N = 512

        function buffers()
            map(enumerate(devs)) do (i, dev)
                device!(dev) do
                    CuArray(fill(Float64(i), N)), CUDACore.zeros(Float64, N)
                end
            end
        end
        allreduce!(bufs, range=eachindex(devs)) =
            for i in range
                NCCL.Allreduce!(bufs[i]..., +, comms[i])
            end
        function check(bufs)
            all(enumerate(devs)) do (i, dev)
                device!(dev) do
                    all(==(sum(1:length(devs))), Array(bufs[i][2]))
                end
            end
        end
        # whether this task has no group open: if it had, another task on this thread could
        # not open one
        no_group_open() = fetch(@async try
            NCCL.group(() -> nothing)
            true
        catch err
            err isa ConcurrencyViolationError || rethrow()
            false
        end)

        # keep a stream busy for a while (using `buf`)
        function spin(buf, cycles)
            start = CUDACore.clock(UInt64)
            while CUDACore.clock(UInt64) - start < cycles
            end
            return
        end

        # an integer whose conversion calls `hook`
        struct YieldingInt <: Integer
            val::Int
            hook::Function
        end
        function Base.Int32(x::YieldingInt)
            x.hook()
            Int32(x.val)
        end
        Base.convert(::Type{Int32}, x::YieldingInt) = Int32(x)

        @testset "groups" begin

        @testset "basic" begin
            bufs = buffers()
            @test NCCL.group(() -> (allreduce!(bufs); 42)) == 42
            @test check(bufs)
            @test no_group_open()
        end

        @testset "temporaries" begin
            # operations in a group only use their buffers when the group ends, so the group
            # has to keep them alive. the send buffers (and the stream) here are only
            # referenced by the group; once freed, their memory would be reused by the
            # arrays allocated afterwards.
            function issue!(recvbufs, refs)
                for (i, dev) in enumerate(devs)
                    device!(dev) do
                        sendbuf = CuArray(fill(Float64(i), N))
                        s = CuStream()
                        NCCL.Allreduce!(sendbuf, recvbufs[i], +, comms[i]; stream=s)
                        push!(refs, WeakRef(sendbuf), WeakRef(s))
                    end
                end
            end
            recvbufs = map(dev -> device!(() -> CUDACore.zeros(Float64, N), dev), devs)
            refs = WeakRef[]
            NCCL.group() do
                issue!(recvbufs, refs)
                GC.gc(true)
                @test all(ref -> ref.value !== nothing, refs)
                for dev in devs
                    device!(dev) do
                        for _ in 1:4
                            CuArray(fill(-1.0, N))
                        end
                    end
                end
            end
            for (i, dev) in enumerate(devs)
                device!(dev) do
                    @test all(==(sum(1:length(devs))), Array(recvbufs[i]))
                end
            end
            @test no_group_open()
        end

        @testset "temporaries collected right after the group" begin
            # once the group has ended, its operations are enqueued but may not have run yet
            # (their streams are busy here). collecting the temporaries then must not let
            # their memory be reused before the operations are done with it.
            function stall(cycles)
                start = CUDACore.clock(UInt64)
                while CUDACore.clock(UInt64) - start < cycles
                end
                return
            end
            recvbufs = map(dev -> device!(() -> CUDACore.zeros(Float64, N), dev), devs)
            streams = map(dev -> device!(() -> CuStream(), dev), devs)
            function issue!()
                for (i, dev) in enumerate(devs)
                    device!(dev) do
                        sendbuf = CuArray(fill(Float64(i), N))
                        NCCL.Allreduce!(sendbuf, recvbufs[i], +, comms[i]; stream=streams[i])
                    end
                end
            end
            for (i, dev) in enumerate(devs)
                device!(dev) do
                    @cuda stream=streams[i] stall(UInt64(1) << 28)
                end
            end
            NCCL.group(issue!)
            GC.gc(true)
            for dev in devs
                device!(dev) do
                    for _ in 1:4
                        CuArray(fill(-1.0, N))
                    end
                end
            end
            for (i, dev) in enumerate(devs)
                device!(dev) do
                    @test all(==(sum(1:length(devs))), Array(recvbufs[i]))
                end
            end
        end

        @testset "ordering on the task's stream" begin
            # the operations are enqueued when the group ends, after the work that produces
            # their inputs and before the work that consumes their outputs
            function delayed_fill!(buf, val, cycles)
                start = CUDACore.clock(UInt64)
                while CUDACore.clock(UInt64) - start < cycles
                end
                for j in threadIdx().x:blockDim().x:length(buf)
                    @inbounds buf[j] = val
                end
                return
            end
            sendbufs = map(dev -> device!(() -> CUDACore.zeros(Float64, N), dev), devs)
            recvbufs = map(dev -> device!(() -> CUDACore.zeros(Float64, N), dev), devs)
            for (i, dev) in enumerate(devs)
                device!(dev) do
                    @cuda threads=256 delayed_fill!(sendbufs[i], Float64(i), UInt64(1) << 28)
                end
            end
            NCCL.group() do
                for i in eachindex(devs)
                    NCCL.Allreduce!(sendbufs[i], recvbufs[i], +, comms[i])
                end
            end
            @test !device!(() -> CUDACore.isdone(stream()), devs[1])
            for (i, dev) in enumerate(devs)
                device!(dev) do
                    recvbufs[i] .+= 1
                end
            end
            for (i, dev) in enumerate(devs)
                device!(dev) do
                    @test all(==(sum(1:length(devs)) + 1), Array(recvbufs[i]))
                end
            end
        end

        @testset "nesting" begin
            # only the outermost group submits. the last rank's operation is issued after the
            # inner group ends, so if that submitted the other ranks' operations, they would
            # be stuck waiting for it (and synchronizing would deadlock).
            bufs = buffers()
            n = length(devs)
            half = max(1, n ÷ 2)
            # the buffers are initialized asynchronously, which would confound `isdone`
            for i in 1:n
                device!(devs[i]) do
                    synchronize(stream())
                end
            end
            NCCL.group() do
                allreduce!(bufs, 1:half-1)
                NCCL.group() do
                    allreduce!(bufs, half:n-1)
                end
                @test !no_group_open()
                @test all(1:n-1) do i
                    device!(devs[i]) do
                        CUDACore.isdone(stream())
                    end
                end
                allreduce!(bufs, n:n)
            end
            @test check(bufs)
            @test no_group_open()
        end

        @testset "other task on the same thread" begin
            bufs = buffers()
            errs = NCCL.group() do
                # `@async` tasks run on the same thread, when the group's owner yields
                op = @async try
                    allreduce!(bufs)
                catch err
                    err
                end
                grp = @async try
                    NCCL.group(() -> nothing)
                catch err
                    err
                end
                fetch(op), fetch(grp)
            end
            @test errs[1] isa ConcurrencyViolationError
            @test errs[2] isa ConcurrencyViolationError
            @test no_group_open()

            NCCL.group(() -> allreduce!(bufs))
            @test check(bufs)
        end

        @testset "exceptions" begin
            # before issuing any operation
            @test_throws ErrorException("before") NCCL.group() do
                error("before")
            end
            @test no_group_open()

            # after issuing all operations, which are still submitted
            bufs = buffers()
            @test_throws ErrorException("after") NCCL.group() do
                allreduce!(bufs)
                error("after")
            end
            @test check(bufs)
            @test no_group_open()

            # in a nested group
            @test_throws ErrorException("nested") NCCL.group() do
                NCCL.group() do
                    error("nested")
                end
            end
            @test no_group_open()

            # a send to oneself without a matching receive, which is invalid, so that NCCL
            # fails to end the group
            comm = only(NCCL.Communicators(devs[1:1]))
            buf = device!(() -> CUDACore.ones(Float64, 4), devs[1])
            invalid_op() = NCCL.Send(buf, comm; dest=0)

            # an invalid operation makes the group fail
            @test_throws NCCL.NCCLError NCCL.group() do
                invalid_op()
            end
            @test no_group_open()

            # ending the group fails too: the original exception is preserved
            @test_throws ErrorException("both") NCCL.group() do
                invalid_op()
                error("both")
            end
            @test no_group_open()
            NCCL.destroy(comm)

            bufs = buffers()
            NCCL.group(() -> allreduce!(bufs))
            @test check(bufs)
        end

        @testset "stickiness" begin
            # the task is pinned to its thread, also after the group
            t = Threads.@spawn begin
                before = current_task().sticky
                inside = NCCL.group(() -> current_task().sticky)
                before, inside, current_task().sticky
            end
            @test fetch(t) == (false, true, true)

            t = Threads.@spawn begin
                try
                    NCCL.group(() -> error("sticky"))
                catch
                end
                current_task().sticky
            end
            @test fetch(t)

            # an `@async` child created in the group outlives it; parent and child must
            # remain on the same thread
            t = Threads.@spawn begin
                tid = Threads.threadid()
                started = Base.Event()
                release = Base.Event()
                local child
                NCCL.group() do
                    child = @async begin
                        notify(started)
                        wait(release)
                        Threads.threadid()
                    end
                    wait(started)
                end
                sticky = current_task().sticky
                tids = Int[]
                for _ in 1:100
                    sleep(0.001)
                    push!(tids, Threads.threadid())
                end
                notify(release)
                sticky, all(==(tid), tids), fetch(child) == tid
            end
            @test fetch(t) == (true, true, true)
        end

        @testset "conversion yields" begin
            # converting a buffer to a pointer may yield (CUDACore can synchronize), during
            # which another task can open a group on this thread. the operation must notice
            # that, rather than join the other task's group. whether converting yields is up
            # to CUDACore, so the operation must either be rejected after a yield, or be
            # submitted without one.

            # a single-rank communicator, so that joining the group would not deadlock
            comm = only(NCCL.Communicators(devs[1:1]))
            ops = [
                buf -> NCCL.Allreduce!(buf, +, comm),
                buf -> NCCL.Broadcast!(buf, comm),
                buf -> NCCL.Send(buf, comm; dest=0),
                buf -> NCCL.Recv!(buf, comm; source=0),
            ]
            # (unified memory, which CUDACore currently synchronizes on the host when it is
            # used on another stream; device memory may be handed over on the GPU instead)
            ones_unified() = fill!(CuArray{Float64,1,CUDACore.UnifiedMemory}(undef, 4), 1)
            # compile the operations, which would take longer than the kernel below
            NCCL.group() do
                for op in ops
                    op(device!(ones_unified, devs[1]))
                end
            end
            # (outside of a group too, which submits differently; sending to or receiving
            # from oneself does need a group)
            for op in ops[1:2]
                op(device!(ones_unified, devs[1]))
            end
            for op in ops
                buf = device!(devs[1]) do
                    buf = ones_unified()
                    # the buffer is last used on another stream, after a long-running kernel,
                    # so using it on the operation's stream has to wait for that stream
                    busy = CuStream()
                    @cuda stream=busy spin(buf, UInt64(1) << 29)
                    buf
                end
                release = Base.Event()
                other = @async NCCL.group(() -> wait(release))
                outcome = try
                    op(buf)
                    :submitted
                catch err
                    err isa ConcurrencyViolationError || err isa NCCL.NCCLError || rethrow()
                    # (sending to or receiving from oneself on its own is invalid, which
                    # NCCL only reports once it has been passed the operation)
                    err isa ConcurrencyViolationError ? :rejected : :submitted
                end
                # the other task can only have started if the operation yielded
                @test outcome == (istaskstarted(other) ? :rejected : :submitted)
                notify(release)
                wait(other)
                @test no_group_open()
            end
            NCCL.destroy(comm)
        end

        @testset "scalar conversion yields" begin
            # the same holds for scalar arguments, which must be converted before the check
            comm = only(NCCL.Communicators(devs[1:1]))
            ops = [
                (buf, x) -> NCCL.Broadcast!(buf, comm; root=x),
                (buf, x) -> NCCL.Reduce!(buf, +, comm; root=x),
                (buf, x) -> NCCL.Send(buf, comm; dest=x),
                (buf, x) -> NCCL.Recv!(buf, comm; source=x),
            ]
            for op in ops
                entered = Base.Event()
                release = Base.Event()
                other = Ref{Task}()
                hook = function ()
                    isassigned(other) && return
                    other[] = @async NCCL.group() do
                        notify(entered)
                        wait(release)
                    end
                    wait(entered)
                end
                buf = device!(() -> CUDACore.ones(Float64, 4), devs[1])
                @test_throws ConcurrencyViolationError op(buf, YieldingInt(0, hook))
                notify(release)
                wait(other[])
                @test no_group_open()
            end
            NCCL.destroy(comm)
        end

        @testset "other task uses a buffer during submission" begin
            # the buffers of an operation are only converted to pointers one by one. if
            # converting the receive buffer yields, another task can use the send buffer on
            # another stream in the meantime. the operation must hold on to the send buffer
            # until it has been enqueued, so that the other task's write waits for it, rather
            # than racing with it while CUDACore believes it comes after the operation.
            comm = only(NCCL.Communicators(devs[1:1]))
            device!(devs[1]) do
                s = CuStream()
                other_stream = CuStream()
                sendbuf = CuArray(fill(42.0, N))
                # (unified memory, so that converting it is likely to yield; see above)
                recvbuf = fill!(CuArray{Float64,1,CUDACore.UnifiedMemory}(undef, N), 0)
                # the send buffer is already used by `s`, so converting it doesn't yield
                NCCL.Allreduce!(sendbuf, recvbuf, +, comm; stream=s)
                stream!(() -> (sendbuf .= 42), s)
                synchronize(s)
                stream!(() -> (CUDACore.zeros(Float64, N) .= -1), other_stream)  # compile
                # the receive buffer is used by a busy stream, so converting it may yield
                busy = CuStream()
                @cuda stream=busy spin(recvbuf, UInt64(1) << 29)

                # if it doesn't, the other task only writes after the operation
                other = @async stream!(other_stream) do
                    sendbuf .= -1
                end
                NCCL.Allreduce!(sendbuf, recvbuf, +, comm; stream=s)
                wait(other)
                @test Array(recvbuf) == fill(42.0, N)
                @test Array(sendbuf) == fill(-1.0, N)
            end
            NCCL.destroy(comm)
        end

        @testset "from a spawned task" begin
            bufs = buffers()
            t = Threads.@spawn NCCL.group() do
                tid = Threads.threadid()
                for i in eachindex(devs)
                    NCCL.Allreduce!(bufs[i]..., +, comms[i])
                    # yielding must not move the task to another thread
                    sleep(0.01)
                    @assert Threads.threadid() == tid
                end
            end
            wait(t)
            @test check(bufs)
            @test no_group_open()
        end

        end

        NCCL.destroy(comms)
        """
    @test run_script(script; threads=4, timeout=300)
end

if length(CUDACore.devices()) >= 2
    @testset "garbage collection during initialization" begin
        # a task blocked in ncclCommInitRankConfig, waiting for its peer, used to prevent the GC
        # from running, while the peer was waiting for the GC to finish.
        #
        # After warming up, rank 0 enters ncclCommInitRankConfig and blocks there because rank 1
        # only joins after another task has completed a number of collections. With bindings
        # that aren't GC-safe the first collection never completes, so rank 1 never joins
        # and the watchdog kills the script.
        script = """
            using CUDACore, NCCL, Test
            for dev in 0:1
                NCCL.destroy(NCCL.Communicator(1, 0, NCCL.UniqueID(); device=dev))
            end

            id = NCCL.UniqueID()
            initializing = Threads.Atomic{Bool}(false)
            collections = Threads.Atomic{Int}(0)
            t1 = Threads.@spawn begin
                initializing[] = true
                NCCL.Communicator(2, 0, id; device=0)
            end
            Threads.@spawn begin
                while !initializing[]
                    sleep(0.01)
                end
                sleep(0.5)
                for _ in 1:10
                    GC.gc()
                    collections[] += 1
                    sleep(0.01)
                end
            end
            t2 = Threads.@spawn begin
                while collections[] < 10
                    sleep(0.01)
                end
                @test !istaskdone(t1)
                NCCL.Communicator(2, 1, id; device=1)
            end
            comms = fetch.([t1, t2])
            NCCL.destroy(comms)
            """
        @test run_script(script; threads=4, timeout=120)
    end
else
    @info "Skipping the test of garbage collection during initialization, which needs 2 GPUs"
    @test_skip "garbage collection during initialization (needs 2 GPUs)"
end

end
