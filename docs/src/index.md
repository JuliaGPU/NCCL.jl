# NCCL.jl

 A Julia wrapper for the [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl).

# API

Buffers passed to collective and point-to-point operations must belong to the device
associated with the communicator. Operations temporarily activate that device, and restore
the previously active device when they return. They are issued on the `stream` keyword
argument, which defaults to the task-local stream of the communicator's device
(`CUDACore.stream()` while that device is active).

```@meta
CurrentModule = NCCL
```

## Library

```@docs
NCCL_jll.is_available
version()
```

## Errors

```@docs
NCCLError
```

## Communicators

```@docs
Communicator
Communicators
UniqueID
Config
rank(comm::Communicator)
nranks(comm::Communicator)
device(comm::Communicator)
destroy
abort
```

## Groups

```@docs
group
```

## Collective communication

```@docs
Allreduce!
Broadcast!
Reduce!
Allgather!
ReduceScatter!
avg
```

## Point-to-point communication

```@docs
Send
Recv!
```
