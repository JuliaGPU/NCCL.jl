# Helpers for running code in separate processes, so that a deadlock fails a test instead of
# hanging the test suite.

# Run `cmds` concurrently, killing all of them if they don't finish within `timeout` seconds,
# or if one fails and the others don't finish within `grace` seconds after that (they are
# probably waiting for the failed one). The output of every process is printed if any of
# them fails. Returns whether all processes succeeded.
function run_processes(cmds::Vector{Cmd}; timeout=300, grace=30,
                       names=["process $i" for i in eachindex(cmds)])
    procs = map(cmds) do cmd
        output = Pipe()
        proc = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output); wait=false)
        close(output.in)
        proc, @async(read(output, String))
    end
    deadline = time() + timeout
    killed = false
    while true
        all(((proc, _),) -> process_exited(proc), procs) && break
        if any(((proc, _),) -> process_exited(proc) && !success(proc), procs)
            deadline = min(deadline, time() + grace)
        end
        if time() > deadline
            @error "Processes did not finish in time, killing them"
            for (proc, _) in procs
                process_exited(proc) || kill(proc, Base.SIGKILL)
            end
            killed = true
            break
        end
        sleep(0.1)
    end
    foreach(((proc, _),) -> wait(proc), procs)
    ok = all(((proc, _),) -> success(proc), procs)
    if !ok
        for ((proc, reader), name) in zip(procs, names)
            status = proc.termsignal != 0 ? "signal $(proc.termsignal)" :
                                            "exit code $(proc.exitcode)"
            println("--- $name ($status) ---")
            println(fetch(reader))
        end
    end
    return ok
end

julia_cmd(script, args...; threads=1) =
    `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$threads -e $script $args`

# Run a Julia script in a separate process. `env` contains pairs of environment variables
# to set, or to remove when the value is `nothing`.
function run_script(script; threads=1, timeout=300, env=())
    cmd = julia_cmd(script; threads)
    if !isempty(env)
        vars = Dict(ENV)
        for (key, value) in env
            value === nothing ? delete!(vars, key) : (vars[key] = value)
        end
        cmd = setenv(cmd, vars)
    end
    run_processes([cmd]; timeout)
end

# The `CUDA_VISIBLE_DEVICES` value that selects each of the devices of this process.
function visible_devices()
    devs = collect(CUDACore.devices())
    # reuse the parent's selection if it lists every device, e.g. by UUID, so that devices
    # are not identified by a (possibly ambiguous) ordinal
    parent = filter(!isempty, strip.(split(get(ENV, "CUDA_VISIBLE_DEVICES", ""), ',')))
    if length(parent) == length(devs)
        return String.(parent)
    end
    return ["GPU-$(CUDACore.uuid(dev))" for dev in devs]
end

# Run `script` in `nranks` processes, each with a single GPU, as the ranks of a communicator.
# The script can use the globals `rank`, `nranks` and `id`, the `NCCL.UniqueID` created by
# rank 0, and has `CUDACore`, `NCCL` and `Test` loaded.
function run_ranks(script, nranks; timeout=300)
    devices = visible_devices()
    nranks <= length(devices) || error("not enough devices for $nranks ranks")
    mktempdir() do dir
        id_file = joinpath(dir, "id")
        prelude = """
            using CUDACore, NCCL, Test
            const rank = parse(Int, ARGS[1])
            const nranks = parse(Int, ARGS[2])
            const id = let id_file = ARGS[3]
                if rank == 0
                    id = NCCL.UniqueID()
                    # write atomically, so that other ranks never read a partial file
                    write(id_file * ".tmp", Vector{UInt8}(id))
                    mv(id_file * ".tmp", id_file)
                    id
                else
                    while !isfile(id_file)
                        sleep(0.1)
                    end
                    NCCL.UniqueID(read(id_file))
                end
            end
            """
        cmds = map(0:nranks-1) do rank
            cmd = julia_cmd(prelude * script, rank, nranks, id_file)
            addenv(cmd, "CUDA_VISIBLE_DEVICES" => devices[rank+1])
        end
        run_processes(cmds; timeout, names=["rank $rank" for rank in 0:nranks-1])
    end
end
