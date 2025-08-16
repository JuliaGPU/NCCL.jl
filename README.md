NCCL.jl
=======

[![codecov](https://codecov.io/gh/JuliaGPU/NCCL.jl/graph/badge.svg?token=1qbhp59GH1)](https://codecov.io/gh/JuliaGPU/NCCL.jl)

A Julia wrapper for the [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl). NCCL is an NVIDIA library
for multi-GPU *and* multi-node communication, optimized for NVIDIA GPUs. The API is designed to be similar to that of MPI, but there are
some differences. For example, unlike CUDA-aware MPI, NCCL can drive *multiple* devices per process. You can read more about how NCCL
compares with MPI [here](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html), and in fact the two can be used together.

NCCL is used internally in several other CUDA libraries, for example [cusolverMp](https://docs.nvidia.com/cuda/cusolvermp/index.html).
