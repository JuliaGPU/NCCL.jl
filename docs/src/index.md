# NCCL.jl

 A Julia wrapper for the [NVIDIA Collective Communications Library](https://developer.nvidia.com/nccl). 

# API

```@meta
CurrentModule = NCCL
```

## Library

```@docs
NCCL_jll.is_available
version()
```

## Communicators

```@docs
Communicators
CUDA.deviceid(comm::Communicator)
size(comm::Communicator)
rank(comm::Communicator)
abort(comm::Communicator)
default_device_stream
```

## Groups

```@docs
group
groupStart
groupEnd
```

## Collective communication

```@docs
Allreduce!
Broadcast!
Reduce!
Allgather!
ReduceScatter!
```

## Point-to-point communication

```@docs
Send
Recv!
```