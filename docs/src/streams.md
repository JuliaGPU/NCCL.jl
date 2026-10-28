# Streams and synchronization

NCCL operations run on the GPU, as kernels enqueued on a CUDA stream. A call to an operation
returns once it has been enqueued, before the communication has happened.


## Which stream

By default, an operation is enqueued on the task-local stream of the communicator's device,
i.e. the stream `CUDACore.stream()` returns while that device is active. This is the same
stream that CUDACore uses for kernels and copies on that device from the same task, so the
operation is ordered with the task's other GPU work: it starts after kernels that the task
launched before, and later kernels and copies wait for it. For example, `Array(buf)` after
an operation returns the result.

The `stream` keyword argument selects another stream, which must belong to the
communicator's device:

```julia
s = CUDACore.device!(NCCL.device(comm)) do
    CuStream()
end
NCCL.Allreduce!(buf, +, comm; stream=s)
```

This allows overlapping communication with computation on the task's stream. CUDACore keeps
track of which stream last used an array: when an array that was passed to an NCCL operation
on `s` is later used on another stream, that use waits for the operation to complete. This
only works for arrays used through CUDACore (e.g. in kernels launched with `@cuda` and in
array operations), not for raw pointers.

CUDACore's per-thread default stream (`per_thread_stream()`) is a different stream on every
OS thread, so it can't be tracked like this, and passing it throws an `ArgumentError`. The
legacy default stream (`legacy_stream()`) is supported.

Operations activate the device of the communicator while they are issued, and restore the
previously active device afterwards, so they can be issued while any device is active.


## Waiting for operations

To wait for an operation on the host, synchronize its stream, e.g. with
`CUDACore.synchronize()` for the task-local stream of the current device, or
`CUDACore.synchronize(s)` for another stream `s`. Copying the result to the host with
`Array(buf)` also waits for it.

When a task drives several devices, as with [`NCCL.Communicators`](@ref), each operation runs
on the task-local stream of its own device. As usual with CUDACore, activate the device of a
buffer before using it, so that the work is ordered with the stream of that device:

```julia
for (dev, buf) in zip(devs, bufs)
    CUDACore.device!(dev) do
        println(Array(buf))
    end
end
```

Operations inside an [`NCCL.group`](@ref) are enqueued when the group ends, on their
respective streams.
