# NCCL.jl

NCCL.jl is a Julia interface to the [NVIDIA Collective Communications Library
(NCCL)](https://developer.nvidia.com/nccl). NCCL implements collective operations (all-reduce,
broadcast, all-gather, ...) and point-to-point send/receive between NVIDIA GPUs. It uses
NVLink, PCIe, InfiniBand or TCP sockets, whichever connects the GPUs best, and runs the
communication as kernels on CUDA streams, so it can overlap with other GPU work.

The API resembles MPI, with some differences. Communication happens between *ranks* that
together form a *communicator*, and each rank uses one GPU. A single process can drive one
rank or several ranks, and the ranks of a communicator can be spread over many processes and
nodes. NCCL does not launch processes: the ranks have to find each other through an
identifier, an [`NCCL.UniqueID`](@ref), that one rank creates and sends to the others by
some other means (a file, a socket, MPI, ...). See [Getting started](@ref) for the common
setups. NVIDIA's documentation also [compares NCCL with
MPI](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html); the two can be used
together.

NCCL.jl provides:

- creating communicators for one process driving several GPUs, for one rank per process or
  per thread, and from MPI communicators, as well as splitting communicators;
- the collective operations `Allreduce!`, `Broadcast!`, `Reduce!`, `Allgather!`,
  `ReduceScatter!`, `Alltoall!`, `Gather!` and `Scatter!`, and point-to-point `Send` and
  `Recv!`, on `CuArray`s of NCCL's native element types and of other `isbits` types;
- [`NCCL.group`](@ref), to combine operations into one NCCL group call;
- the low-level C API in the `NCCL.LibNCCL` submodule.


## Requirements

- Julia 1.10 or later, on Linux (x86_64 or aarch64).
- One or more NVIDIA GPUs supported by CUDA 12 or 13, with a driver that provides NVML
  (`libnvidia-ml`), which NCCL uses to discover the system topology. NCCL doesn't run on the
  integrated GPUs of Jetson (Tegra) devices: JetPack 5 has no NVML, and on JetPack 7 NVML
  doesn't support the peer-to-peer query NCCL makes when creating a communicator
  (`nvmlDeviceGetP2PStatus`), so creating one fails with `ncclSystemError`.
- [CUDACore.jl](https://github.com/JuliaGPU/CUDA.jl), the core of CUDA.jl, for arrays,
  devices and streams. Arrays are `CuArray`s from CUDACore, which are the same arrays that
  CUDA.jl provides.
- The NCCL library is provided by the NCCL_jll package, which picks the build matching the
  CUDA toolkit that CUDACore uses. [`NCCL.version()`](@ref) returns the version of the
  library in use, and [`NCCL.is_available()`](@ref NCCL_jll.is_available) whether one is
  available at all: NCCL_jll has no builds for a local CUDA toolkit (CUDA_Runtime_jll's
  `local` preference), nor for platforms other than Linux.

NCCL.jl only supports blocking communicators (see [Nonblocking
communicators](@ref)), and doesn't yet wrap NCCL's device API, user buffer registration,
one-sided operations, or the variable-size collectives.


## Example

This sums a vector over all GPUs of a system, from a single process:

```julia
using CUDACore, NCCL

devs = collect(CUDACore.devices())
bufs = map(enumerate(devs)) do (i, dev)
    CUDACore.device!(dev) do
        CuArray(fill(Float32(i), 4))
    end
end

NCCL.Communicators(devs) do comms
    # issue the operations of all ranks together, or the first one waits for the others
    NCCL.group() do
        for (comm, buf) in zip(comms, bufs)
            NCCL.Allreduce!(buf, +, comm)
        end
    end
end

for (dev, buf) in zip(devs, bufs)
    CUDACore.device!(dev) do
        @show Array(buf)    # sum(1:length(devs)) everywhere
    end
end
```

The following pages explain the parts of this example, and the other ways to set up a
communicator.
