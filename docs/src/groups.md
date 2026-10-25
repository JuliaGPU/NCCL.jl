# Groups

[`NCCL.group(f)`](@ref NCCL.group) runs `f` as an NCCL group: the operations that `f`
issues are collected, and only submitted when `f` returns, all together.


## When to use groups

Groups are needed:

- when a task drives several ranks of a communicator, as with
  [`NCCL.Communicators`](@ref). Issuing a collective operation on one rank blocks until the
  other ranks have issued it too, so the operations of all ranks have to be in one group:

  ```julia
  NCCL.group() do
      for (comm, buf) in zip(comms, bufs)
          NCCL.Allreduce!(buf, +, comm)
      end
  end
  ```

- for point-to-point communication in which a rank both sends and receives, which would
  otherwise deadlock (see [Point-to-point operations](@ref)).

Groups can also improve performance: NCCL fuses the operations of a group, for example
several reductions of small buffers, into fewer kernel launches.

Destroying and splitting communicators also involve all ranks, and need the same grouping
when a task drives several ranks. These functions can't be called inside `NCCL.group`;
instead, they accept a vector of communicators, and group the calls themselves:
`NCCL.destroy(comms)` and `NCCL.split(comms, colors)`. [`NCCL.abort`](@ref) can't be called
inside a group either, but doesn't need one: aborting only cleans up locally, and
`NCCL.abort(comms)` aborts the communicators one after the other.


## Rules

Groups can be nested; the operations are submitted when the outermost group ends.

**Buffers.** An operation in a group is only enqueued on its stream when the outermost group
ends. Until then, the buffers passed to it must not be read, written or freed, and must not
be handed to another task or stream, even if the operation was issued earlier in the group.
NCCL.jl keeps the buffers, streams and communicators of the group's operations alive until
the group ends, so temporary arrays can be passed. After the group, the operations are
ordered like any other work on their streams.

**Tasks and threads.** NCCL tracks groups per OS thread, while Julia tasks can move between
threads and share them. To keep the operations of a group together:

- `NCCL.group` pins the calling task to the thread it runs on, and doesn't undo that when
  the group ends (as `@async` does to the task that calls it).
- All operations of a group have to be issued by the task that called `NCCL.group`.
  Operations issued by tasks it starts, e.g. with `Threads.@spawn`, are not part of the group.
- While a group is open, other tasks that run on the same thread (because the group's task
  yields, e.g. to wait for a lock or for I/O) can't use NCCL: their NCCL calls throw a
  `ConcurrencyViolationError` instead of silently joining the group.

**Exceptions.** When `f` throws, the group is still ended, and the exception is rethrown.
What happens to the operations that `f` issued before the exception depends on whether NCCL
recorded an error in its group state:

- NCCL records the errors of operations issued in the group, e.g. when it rejects their
  arguments (throwing an `NCCLError`). The outermost group end then discards all operations
  of the group, and submits nothing.
- Other exceptions don't affect the group: ending it submits the operations issued before
  the exception. This includes Julia exceptions, e.g. from the argument checks that NCCL.jl
  performs before calling NCCL, and `NCCLError`s from NCCL calls that don't record their
  errors in the group, e.g. creating a [`UniqueID`](@ref NCCL.UniqueID). Other ranks may
  then wait for operations that this rank never issued, and the program may hang.

Because communicators are blocking, ending the group can itself block, e.g. until the other
ranks have issued their matching operations, before the exception reaches any exception
handler of the caller. Recovery therefore can't rely on the failing task, and has to be
arranged out of band, for the whole job: for example, by having every rank abort its
communicators (with [`NCCL.abort`](@ref)) when a watchdog or another process reports a
failure, or by letting the job scheduler terminate the whole job when one process fails.
Aborting a blocking communicator is not guaranteed to interrupt a call that is blocked
inside NCCL, so terminating the job is the only reliable way out of a hang.
