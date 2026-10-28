# Getting started

Every rank of a communicator needs a GPU of its own, and all ranks have to create their part
of the communicator at the same time: creating a communicator blocks until every rank has
joined. How the ranks find each other depends on how the program is organized. This page
shows complete programs for the common setups. Each of them sums a small vector over all
ranks.


## Installation

Install NCCL.jl together with CUDACore.jl, which provides the arrays, devices and streams that
the examples use:

```julia
using Pkg
Pkg.add(["NCCL", "CUDACore"])
```

CUDACore.jl is the core of [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). In a project that
uses CUDA.jl, `using CUDA` can replace `using CUDACore`: CUDA.jl re-exports CUDACore.jl, so
e.g. `CUDA.device!` and `CuArray` are the same functions and types. The [MPI](@ref) example
also needs MPI.jl: `Pkg.add("MPI")`.


## One process driving all GPUs

The simplest setup uses a single process and a single task for all GPUs of one node.
[`NCCL.Communicators`](@ref) creates a communicator with one rank per device, without the
need for a [`UniqueID`](@ref NCCL.UniqueID):

```julia
using CUDACore, NCCL

devs = collect(CUDACore.devices())
bufs = map(enumerate(devs)) do (i, dev)
    CUDACore.device!(dev) do
        CuArray(fill(Float32(i), 4))
    end
end

NCCL.Communicators(devs) do comms
    NCCL.group() do
        for (comm, buf) in zip(comms, bufs)
            NCCL.Allreduce!(buf, +, comm)
        end
    end
end

for (dev, buf) in zip(devs, bufs)
    CUDACore.device!(dev) do
        println(Array(buf))
    end
end
```

Rank `i-1` uses `devs[i]`. A collective operation only starts once every rank has issued it,
and NCCL waits for that while issuing the operation. When one task drives all ranks, it can
therefore only issue an operation on all of them inside [`NCCL.group`](@ref), which issues
them together when the group ends (see [Groups](@ref)). The operations run asynchronously on
the stream of each device: `Array(buf)` copies the result on the same stream, so it waits for
the operation to complete.

The do-block form destroys the communicators when the block ends (see [Communicator
lifetime](@ref)).


## One task per GPU

The ranks can also run on different threads of one process, each in a task of its own. One
of the tasks, or the parent task, creates the [`UniqueID`](@ref NCCL.UniqueID), and each
task creates its rank with [`NCCL.Communicator(nranks, rank, id)`](@ref NCCL.Communicator):

```julia
# start Julia with at least as many threads as there are GPUs, e.g. `julia --threads=4`
using CUDACore, NCCL

devs = collect(CUDACore.devices())
nranks = length(devs)
Threads.nthreads() >= nranks || error("start Julia with at least $nranks threads")

id = NCCL.UniqueID()
tasks = map(0:nranks-1) do rank
    Threads.@spawn begin
        CUDACore.device!(devs[rank+1])
        NCCL.Communicator(nranks, rank, id) do comm
            buf = CuArray(fill(Float32(rank+1), 4))
            NCCL.Allreduce!(buf, +, comm)
            Array(buf)
        end
    end
end
foreach(println, fetch.(tasks))
```

Every task has to run on a thread of its own: creating a communicator blocks the thread
until all ranks have joined, so a task that has to wait for a free thread would never join.
The communicator uses the `device` keyword argument of `Communicator`, which defaults to the
task's current device. Each task issues the operations of its own rank, so no group is
needed.


## One process per GPU

Most multi-node programs run one process per GPU. Rank 0 then creates the
[`UniqueID`](@ref NCCL.UniqueID), and sends it to the other processes. A `UniqueID` is 128
bytes: `Vector{UInt8}(id)` returns them, and `NCCL.UniqueID(bytes)` turns them back into an
identifier. This program exchanges them through a file on a file system that all processes
can access:

```julia
# usage: julia --project allreduce.jl RANK NRANKS DEVICE ID_FILE
using CUDACore, NCCL

rank = parse(Int, ARGS[1])
nranks = parse(Int, ARGS[2])
device = parse(Int, ARGS[3])    # index of the GPU among those this process can see
id_file = ARGS[4]

if rank == 0
    id = NCCL.UniqueID()
    # write to a temporary file first, so that other ranks never read a partial file
    write(id_file * ".tmp", Vector{UInt8}(id))
    mv(id_file * ".tmp", id_file)
else
    while !isfile(id_file)
        sleep(0.1)
    end
    id = NCCL.UniqueID(read(id_file))
end

CUDACore.device!(device)

NCCL.Communicator(nranks, rank, id) do comm
    buf = CuArray(fill(Float32(rank+1), 4))
    NCCL.Allreduce!(buf, +, comm)
    println("rank $rank: ", Array(buf))
end
```

Start the processes at the same time, and pass a path that doesn't exist yet as `ID_FILE`.
On a single node with four GPUs, rank `i` can use GPU `i`:

```sh
for rank in 0 1 2 3; do
    julia --project allreduce.jl $rank 4 $rank /tmp/nccl-id &
done
wait
```

On several nodes, `RANK` is the global rank, but `DEVICE` is the index of a GPU on the node
of the process, and the processes on one node must use different GPUs. Job schedulers often
make only a single GPU visible to each process, e.g. by setting `CUDA_VISIBLE_DEVICES`; that
GPU is then device 0.

Only one rank may create the identifier: creating it also starts a listener in that process,
which the other ranks connect to. The identifier can also be sent over a socket, or with any
other library: it is a plain bits type, so for example `Serialization` and `Distributed` can
send it as is.


## MPI

In programs that use MPI, loading [MPI.jl](https://github.com/JuliaParallel/MPI.jl) enables
[`NCCL.Communicator(mpicomm)`](@ref NCCL.Communicator(::Any)), which creates a communicator
with the ranks of an MPI communicator: rank 0 creates the identifier and broadcasts it over
`mpicomm`, and every MPI rank becomes the NCCL rank with the same index.

Each process on a node has to use a different GPU. Unless each process only sees a single
GPU (e.g. because the launcher sets `CUDA_VISIBLE_DEVICES` per process), select one based on
the rank of the process among the processes on the same node:

```julia
using MPI, CUDACore, NCCL
MPI.Init()

world_rank = MPI.Comm_rank(MPI.COMM_WORLD)
node_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, world_rank)
MPI.Comm_size(node_comm) <= ndevices() ||
    error("more processes on this node than GPUs")
CUDACore.device!(MPI.Comm_rank(node_comm))

NCCL.Communicator(MPI.COMM_WORLD) do comm
    buf = CuArray(fill(Float32(world_rank+1), 4))
    NCCL.Allreduce!(buf, +, comm)
    println("rank $world_rank: ", Array(buf))
end
```

Run it with MPI.jl's launcher, e.g. `` `$(MPI.mpiexec()) -n 4 julia --project script.jl` ``,
or with the `mpiexec` of the system's MPI library if MPI.jl is configured to use that.
