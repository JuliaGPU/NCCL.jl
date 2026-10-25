NCCL.jl
=======

[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagpu.github.io/NCCL.jl/stable)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliagpu.github.io/NCCL.jl/dev)
[![codecov](https://codecov.io/gh/JuliaGPU/NCCL.jl/graph/badge.svg?token=1qbhp59GH1)](https://codecov.io/gh/JuliaGPU/NCCL.jl)

A Julia interface to the [NVIDIA Collective Communications Library
(NCCL)](https://developer.nvidia.com/nccl), for collective and point-to-point communication
between NVIDIA GPUs, within a node and across nodes. NCCL.jl works with the `CuArray`s of
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), and supports a single process driving several
GPUs, one process or thread per GPU, and creating communicators from MPI.

Install it together with CUDACore.jl, the core of CUDA.jl (in projects that use CUDA.jl,
`using CUDA` works as well, as it re-exports CUDACore.jl):

```julia
using Pkg
Pkg.add(["NCCL", "CUDACore"])
```

For example, to sum a vector over all GPUs of a node:

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
```

See the [documentation](https://juliagpu.github.io/NCCL.jl/dev) for the other ways to create
communicators, the supported operations and element types, and the rules for groups and
streams. Coming from NCCL.jl 0.1? The documentation has a [migration
guide](https://juliagpu.github.io/NCCL.jl/dev/migration/).

NCCL.jl requires Julia 1.10 or later on Linux, and a GPU supported by CUDA 12 or 13 with a
working NVML (`libnvidia-ml`); NCCL doesn't run on Jetson (Tegra) integrated GPUs. The NCCL
library is provided by NCCL_jll.
