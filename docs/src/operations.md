# Operations

NCCL.jl wraps NCCL's collective and point-to-point operations. They follow the same
conventions:

- Buffers are passed first, then the reduction operator (for reductions), then the
  communicator. The root of rooted operations, the peer of point-to-point operations, and the
  stream are keyword arguments.
- Operations write their result into a buffer passed by the caller, and return that buffer.
- Every rank of the communicator has to call a collective operation, with the same root
  and operator, and with buffers of the same element type and of matching lengths.
- Operations are asynchronous: they return once the operation has been enqueued on a stream
  (see [Streams and synchronization](@ref)).


## Collective operations

In this table, `n` is the number of ranks of the communicator, and `len` the length of a
buffer.

| Operation                                                    | Buffer lengths                 | Result                                                                        |
|:------------------------------------------------------------ |:------------------------------ |:----------------------------------------------------------------------------- |
| [`Allreduce!(sendbuf, recvbuf, op, comm)`](@ref NCCL.Allreduce!)      | equal                          | `recvbuf` on every rank holds the reduction of all `sendbuf`s                 |
| [`Broadcast!(sendbuf, recvbuf, comm; root)`](@ref NCCL.Broadcast!)    | equal                          | `recvbuf` on every rank holds `sendbuf` of `root`                             |
| [`Reduce!(sendbuf, recvbuf, op, comm; root)`](@ref NCCL.Reduce!)      | equal                          | `recvbuf` on `root` holds the reduction of all `sendbuf`s                     |
| [`Allgather!(sendbuf, recvbuf, comm)`](@ref NCCL.Allgather!)          | `len(recvbuf) = n*len(sendbuf)` | `recvbuf` on every rank holds all `sendbuf`s, in rank order                   |
| [`ReduceScatter!(sendbuf, recvbuf, op, comm)`](@ref NCCL.ReduceScatter!) | `len(sendbuf) = n*len(recvbuf)` | `recvbuf` on rank `i` holds block `i` of the reduction of all `sendbuf`s       |
| [`Alltoall!(sendbuf, recvbuf, comm)`](@ref NCCL.Alltoall!)            | equal, a multiple of `n`       | block `i` of `recvbuf` on rank `j` holds block `j` of `sendbuf` on rank `i`   |
| [`Gather!(sendbuf, recvbuf, comm; root)`](@ref NCCL.Gather!)          | `len(recvbuf) = n*len(sendbuf)` | `recvbuf` on `root` holds all `sendbuf`s, in rank order                       |
| [`Scatter!(sendbuf, recvbuf, comm; root)`](@ref NCCL.Scatter!)        | `len(sendbuf) = n*len(recvbuf)` | `recvbuf` on rank `i` holds block `i` of `sendbuf` on `root`                  |

Ranks and blocks are counted from 0, and `root` defaults to 0. Only the length of a buffer
matters, not its shape: a buffer of `n*len` elements can for example be an `len×n` matrix,
whose columns are the blocks.

### Buffers that only matter on the root

On ranks other than the root, NCCL ignores some of the buffers: the send buffer of
`Broadcast!` and `Scatter!`, and the receive buffer of `Reduce!` and `Gather!`. These ranks
can pass `nothing` instead of allocating a buffer:

```julia
NCCL.Reduce!(sendbuf, NCCL.rank(comm) == root ? recvbuf : nothing, +, comm; root)
```

### In-place operations

Operations can work in place, with the send and receive buffers sharing memory, using the
layouts that [NCCL
documents](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html).
Some operations have a method that takes a single buffer:

| Operation                                       | Layout                                                                                     |
|:----------------------------------------------- |:------------------------------------------------------------------------------------------ |
| `Allreduce!(buf, op, comm)`                     | `buf` is both the send and receive buffer                                                  |
| `Broadcast!(buf, comm; root)`                   | idem                                                                                       |
| `Reduce!(buf, op, comm; root)`                  | idem                                                                                       |
| `Allgather!(buf, comm)`                         | block `rank` of `buf` holds the data to send, `buf` receives all blocks                   |
| `ReduceScatter!(buf, op, comm)`                 | `buf` holds the data to reduce, block `rank` of `buf` receives the result, which is returned as a view; the other blocks are undefined afterwards |

The two-buffer methods also work in place when the buffers are laid out the same way: when
`sendbuf` of `Allgather!` is block `rank` of `recvbuf`, when `recvbuf` of `ReduceScatter!` is
block `rank` of `sendbuf`, when `sendbuf` of `Gather!` is block `root` of `recvbuf` on the
root, and when `recvbuf` of `Scatter!` is block `root` of `sendbuf` on the root. Such blocks
can be passed as views, e.g. `view(recvbuf, rank*len+1:(rank+1)*len)`. `Alltoall!` can't be
performed in place. Buffers that overlap in any other way throw an `ArgumentError`.


## Point-to-point operations

[`NCCL.Send(sendbuf, comm; dest)`](@ref NCCL.Send) sends a buffer to rank `dest`, which
receives it with [`NCCL.Recv!(recvbuf, comm; source)`](@ref NCCL.Recv!) into a buffer of the
same element type and length.

A send only completes when the matching receive runs, and the other way around. Operations
issued on one stream run one after the other, so a rank that both sends and receives has to
put these operations in a group, which lets NCCL run them concurrently. For example, to send
data around a ring of ranks:

```julia
rank, n = NCCL.rank(comm), NCCL.nranks(comm)
NCCL.group() do
    NCCL.Send(sendbuf, comm; dest=mod(rank+1, n))
    NCCL.Recv!(recvbuf, comm; source=mod(rank-1, n))
end
```

Without the group, every rank would wait in its send for the receive of the next rank, which
is only issued after that rank's send. Groups can contain sends and receives with many
peers, which is how NCCL implements patterns that don't fit the collective operations, such
as exchanges with neighbors, or an all-to-all with a different amount of data per pair of
ranks.


## Reduction operators

Reductions (`Allreduce!`, `Reduce!` and `ReduceScatter!`) accept these operators:

| Operator                     | Result                                                                          |
|:---------------------------- |:------------------------------------------------------------------------------- |
| `+`                          | the sum of the values of all ranks                                              |
| `*`                          | the product                                                                     |
| `max`, `min`                 | the maximum or minimum                                                          |
| [`NCCL.avg`](@ref)           | the sum divided by the number of ranks                                          |
| [`NCCL.PreMulSum(s)`](@ref NCCL.PreMulSum) | the sum of the values multiplied by `s`, e.g. `PreMulSum(1/n)` to average |

NCCL performs all arithmetic in the element type of the buffers: integer sums and products
wrap around on overflow, and `NCCL.avg` of integers truncates towards zero. Each rank can
pass a different scalar to `PreMulSum`. Other functions throw an `ArgumentError`.


## Element types

NCCL natively supports `Int8`, `UInt8`, `Int32`, `UInt32`, `Int64`, `UInt64`, `Float16`,
`Float32`, `Float64` and `BFloat16` (from [BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl),
which re-exports `Core.BFloat16` where Julia's LLVM supports it on the host architecture,
e.g. x86_64 with Julia 1.11 and later, and otherwise defines a `BFloat16` type of its own).
NCCL.jl passes some other element types to NCCL as one of these, which restricts the
reductions that are correct for them:

| Element type                                       | Passed to NCCL as         | Reduction operators                       |
|:-------------------------------------------------- |:------------------------- |:----------------------------------------- |
| native types (see above)                           | themselves                | all                                       |
| `Complex{T}`, for `T` a native floating-point type | pairs of `T`              | `+`, `NCCL.avg`, `NCCL.PreMulSum` (real scalars) |
| `Bool`                                             | `UInt8`                   | `max` (logical or), `min` (logical and)   |
| other `isbits` types, e.g. `Int16`, `Char`, structs | bytes                    | none                                      |

All of these can be used with the operations that only copy data: `Broadcast!`,
`Allgather!`, `Alltoall!`, `Gather!`, `Scatter!`, `Send` and `Recv!`. Values of other
`isbits` types are copied byte by byte, including any padding. Types that aren't `isbits`,
`isbits` unions like `Union{Int8,Nothing}`, and types of size zero are not supported.

Complex numbers are reduced by reducing their real and imaginary parts separately, which is
only correct for sums, averages, and multiplication by a real scalar. Reductions with other
operators, or of other types, throw an `ArgumentError`.


## Buffer requirements

The buffers passed to an operation have to:

- be stored contiguously in GPU memory: a `CuArray`, or a contiguous view or reshape of one.
  Non-contiguous views (e.g. `view(x, 1:2:10)`, or a row of a matrix) and host arrays throw
  an `ArgumentError`; copy them into a `CuArray` first.
- be on the device of the communicator.
- have the same element type, and lengths that fit the operation (see above). Lengths that
  don't fit throw a `DimensionMismatch`.
- not overlap, except in the in-place layouts described above.

These requirements are checked before calling NCCL. NCCL cannot check that the buffers of
different ranks match, though: mismatched lengths or element types across ranks lead to
wrong results, hangs, or crashes.

Inside a [group](@ref Groups), buffers must not be used for anything else until the group
ends.
