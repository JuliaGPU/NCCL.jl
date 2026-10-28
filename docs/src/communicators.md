# Communicators

A [`Communicator`](@ref NCCL.Communicator) object represents one rank of an NCCL
communicator. It is bound to one device, and knows its rank and the number of ranks:

```julia
NCCL.rank(comm)     # 0-based index of this rank
NCCL.nranks(comm)   # number of ranks in the communicator
NCCL.device(comm)   # the CuDevice of this rank
```


## Creating communicators

There are three ways to create communicators (see [Getting started](@ref) for complete
examples):

- [`NCCL.Communicators(devices)`](@ref NCCL.Communicators) creates all ranks of a
  communicator in the current task, one per device, and returns them as a vector.
- [`NCCL.Communicator(nranks, rank, id)`](@ref NCCL.Communicator) creates a single rank.
  Every rank calls it with the same `nranks` and [`UniqueID`](@ref NCCL.UniqueID), at the
  same time, from different processes or threads. The rank uses the current device, or the
  one passed as the `device` keyword argument.
- [`NCCL.Communicator(mpicomm)`](@ref NCCL.Communicator(::Any)) does the same for every rank
  of an MPI communicator, after exchanging the `UniqueID` with MPI. It requires loading
  MPI.jl.

If creating a communicator fails, the communicators that NCCL already allocated are aborted
before the error is thrown. The other ranks may still be waiting for the failed rank, though:
NCCL.jl can't make them fail too, except when using MPI, where the ranks check their
arguments together before creating the communicator.

### Configuration

All constructors accept a `config` keyword argument, an [`NCCL.Config`](@ref) that sets
options of the new communicator, e.g. its name in NCCL's logs and the number of CTAs (thread
blocks) its kernels use:

```julia
config = NCCL.Config(comm_name="data parallel", max_ctas=8)
comm = NCCL.Communicator(nranks, rank, id; config)
```

Options that aren't set are left to NCCL, which also reads many of them from environment
variables (see [NCCL's list of environment
variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).

### Splitting communicators

[`NCCL.split`](@ref) creates communicators for subsets of the ranks of an existing one,
without exchanging a new `UniqueID`. Ranks that pass the same color end up in the same new
communicator, and ranks that pass `nothing` in none:

```julia
# two communicators, of the even and of the odd ranks
half = NCCL.split(comm, isodd(NCCL.rank(comm)) ? 1 : 0)
```

Splitting is a collective operation: every rank of `comm` has to call `NCCL.split`, and
there must be no operations in progress on `comm`. A task that drives several ranks, as
returned by `Communicators`, splits them with a single call:
`NCCL.split(comms, colors)`.


## Communicator lifetime

Communicators are not destroyed by the garbage collector. When a program is done with a
communicator, it has to destroy it with [`NCCL.destroy`](@ref):

```julia
comm = NCCL.Communicator(nranks, rank, id)
# ... use comm ...
NCCL.destroy(comm)
```

Destroying a communicator first waits for its operations to complete, and then synchronizes
with the other ranks on the same node: every rank has to destroy its communicator, and the
ranks wait for each other. If the garbage collector did this (as NCCL.jl 0.1 did), each rank
would destroy its communicator at a different, unpredictable time, which can make a rank
block while it holds resources that others need, and deadlock the program (see
[#66](https://github.com/JuliaGPU/NCCL.jl/issues/66)). A communicator that is never
destroyed is not a correctness problem, though: its resources are only released when the
process exits.

A task that drives several ranks of a communicator, as returned by
[`NCCL.Communicators`](@ref) or `NCCL.split(comms, ...)`, has to destroy them together by
passing the vector: `NCCL.destroy(comms)`. Destroying them one by one, e.g. with
`NCCL.destroy.(comms)`, deadlocks when the first one waits for the others.

The do-block forms of the constructors and of `split` destroy the communicators when the
block returns:

```julia
NCCL.Communicator(nranks, rank, id) do comm
    # ... use comm ...
end
```

When the block throws, the communicators are aborted instead, with [`NCCL.abort`](@ref),
and the exception is rethrown. Aborting a communicator cancels its operations and frees its
resources without waiting for the other ranks. This only cleans up locally: the other ranks
do not learn about the error, and may wait forever for operations that the failed rank
never issues. A program that needs to survive failures has to tell the other ranks by other
means, e.g. MPI, so that they abort their communicators too. NCCL describes this in its
[fault tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#fault-tolerance)
documentation.

After `destroy` or `abort`, the communicator is closed: `isopen(comm)` returns `false`, and
using it throws an `InvalidStateException`. Destroying or aborting it again does nothing.
`NCCL.rank`, `NCCL.nranks` and `NCCL.device` keep working.


## Nonblocking communicators

NCCL can create nonblocking communicators, whose functions return before they complete
and require polling for the result. NCCL.jl doesn't support those yet, and always creates
blocking communicators. NCCL lets the environment variable `NCCL_COMM_BLOCKING` override
this choice, also when the variable is set in an NCCL configuration file
(`~/.nccl.conf`, `/etc/nccl.conf` or the file `NCCL_CONF_FILE` points to). To prevent that,
NCCL.jl sets `NCCL_COMM_BLOCKING=1` when it is loaded, unless the variable is already set,
and creating a communicator throws an error when it is set to 0. See [`NCCL.Config`](@ref)
for the details.
