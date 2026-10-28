# API reference

```@meta
CurrentModule = NCCL
```

Except for `Communicator`, the functions and types of NCCL.jl are not exported: refer to
them as `NCCL.name`.

## Library

```@docs
NCCL_jll.is_available
version
```

## Communicators

```@docs
Communicator
Communicator(::Integer, ::Integer, ::UniqueID)
Communicator(::Any)
Communicators
UniqueID
Config
rank
nranks
device
split
destroy
abort
```

## Collective communication

```@docs
Allreduce!
Broadcast!
Reduce!
Allgather!
ReduceScatter!
Alltoall!
Gather!
Scatter!
```

## Point-to-point communication

```@docs
Send
Recv!
```

## Reduction operators

```@docs
avg
PreMulSum
```

## Group calls

```@docs
group
```

## Errors

```@docs
NCCLError
```
