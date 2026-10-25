# Errors

NCCL.jl checks the arguments of its functions before calling NCCL, and throws standard Julia
exceptions when they are invalid:

- `DimensionMismatch` for buffers whose lengths don't fit the operation;
- `ArgumentError` for other invalid arguments, e.g. non-contiguous buffers, buffers on
  another device, mismatched element types, unsupported element types or reduction operators,
  ranks out of range, or overlapping buffers;
- `InvalidStateException` when using a communicator that has been destroyed or aborted;
- `ConcurrencyViolationError` when a task uses NCCL on a thread on which another task has an
  open [group](@ref Groups);
- `InexactError` when the scalar of an [`NCCL.PreMulSum`](@ref) can't be converted to the
  integer element type of the buffers.

Errors reported by NCCL itself are thrown as an [`NCCLError`](@ref NCCL.LibNCCL.NCCLError),
which holds NCCL's result code and the error message NCCL reported for it. NCCL keeps only one
such message per process, and only updates it when it logs a warning, so NCCL.jl only
includes it when it changed since the previous error; the message is empty for errors NCCL
didn't log (or that repeat the previous message exactly), and it can describe an error that
happened at the same time on another thread. To see what led to an error, enable NCCL's logging by setting
the environment variable `NCCL_DEBUG` to `WARN` (or `INFO` for more detail) before starting
Julia; see [NCCL's
documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug).

Errors on one rank are not reported to the other ranks. These may keep waiting for the
failed rank, in an operation or when creating or destroying a communicator. See
[Communicator lifetime](@ref) for cleaning up after an error.
