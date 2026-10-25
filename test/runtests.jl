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
@info "NCCL version: $(NCCL.version())"

# Run a Julia script in a separate process, killing it if it doesn't finish in time, so
# that a deadlock fails the test instead of hanging the test suite.
function run_script(script; threads=1, timeout=300)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$threads -e $script`
    output = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output); wait=false)
    close(output.in)
    reader = @async read(output, String)
    if timedwait(() -> process_exited(proc), timeout) === :timed_out
        @error "Script did not finish within $timeout seconds, killing it"
        kill(proc, Base.SIGKILL)
    end
    wait(proc)
    log = fetch(reader)
    success(proc) || println(log)
    return success(proc)
end

@testset "NCCL" begin

@testset "Communicator" begin
    # clique of communicators
    comms = NCCL.Communicators(CUDACore.devices())
    for (i,dev) in enumerate(CUDACore.devices())
        @test NCCL.rank(comms[i]) == i-1
        @test NCCL.device(comms[i]) == dev
        @test NCCL.size(comms[i]) == length(CUDACore.devices())
    end

    # single communicator (with nranks=1 or this would block)
    comm  = Communicator(1, 0)
    @test NCCL.device(comm) == CuDevice(0)

    NCCL.destroy.(comms)
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

    NCCL.destroy.(comms)
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

    NCCL.destroy.(comms)
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

    NCCL.destroy.(comms)
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

    NCCL.destroy.(comms)
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

    NCCL.destroy.(comms)
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

    NCCL.destroy.(comms)
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
    comm = Communicator(1, 0)
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

@testset "large count" begin
    # counts used to be passed as `Cint`, and failed beyond `typemax(Cint)`
    script = """
        using CUDACore, NCCL, Test
        n = 2^31 + 8
        if CUDACore.free_memory() < 2n + 2^30
            @warn "Not enough GPU memory, skipping the large count test"
            exit(0)
        end
        comm = Communicator(1, 0)
        sendbuf = CUDACore.fill(0x01, n)
        recvbuf = CUDACore.zeros(UInt8, n)
        NCCL.Allreduce!(sendbuf, recvbuf, +, comm)
        @test Array(recvbuf[1:8]) == fill(0x01, 8)
        @test Array(recvbuf[end-7:end]) == fill(0x01, 8)
        NCCL.destroy(comm)
        """
    @test run_script(script; timeout=300)
end

if length(CUDACore.devices()) >= 2
    @testset "garbage collection during initialization" begin
        # a task blocked in ncclCommInitRank, waiting for its peer, used to prevent the GC
        # from running, while the peer was waiting for the GC to finish.
        #
        # After warming up, rank 0 enters ncclCommInitRank and blocks there because rank 1
        # only joins after another task has completed a number of collections. With bindings
        # that aren't GC-safe the first collection never completes, so rank 1 never joins
        # and the watchdog kills the script.
        script = """
            using CUDACore, NCCL, Test
            for dev in 0:1
                device!(dev)
                NCCL.destroy(NCCL.Communicator(1, 0))
            end

            id = NCCL.UniqueID()
            initializing = Threads.Atomic{Bool}(false)
            collections = Threads.Atomic{Int}(0)
            t1 = Threads.@spawn begin
                device!(0)
                initializing[] = true
                NCCL.Communicator(2, 0; unique_id=id)
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
                # NCCL uses the context current on the OS thread, and the task may have
                # migrated while sleeping
                device!(1)
                NCCL.Communicator(2, 1; unique_id=id)
            end
            comms = fetch.([t1, t2])
            NCCL.destroy.(comms)
            """
        @test run_script(script; threads=4, timeout=120)
    end
else
    @info "Skipping the test of garbage collection during initialization, which needs 2 GPUs"
    @test_skip "garbage collection during initialization (needs 2 GPUs)"
end

end
