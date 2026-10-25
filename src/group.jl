# Group calls
#
# NCCL keeps the state of a group (the nesting depth, the queued operations) in thread-local
# variables, so everything between ncclGroupStart and ncclGroupEnd has to happen on the same
# OS thread, and other code running on that thread must not issue NCCL calls in between
# because they would silently join the group. We pin the task that opens a group to its
# thread, and remember per thread which task owns the open group so that other tasks
# scheduled on that thread (when the owner yields) get an error instead.

mutable struct GroupState
    owner::Task
    depth::Int
    # objects used by operations in the group, which NCCL only uses when the outermost
    # group ends
    roots::Vector{Any}
end

# keyed by `Threads.threadid()`; only contains threads with an open group
const group_states = Dict{Int,GroupState}()
const group_lock = Threads.SpinLock()

function group_state(tid::Int=Threads.threadid())
    @lock group_lock get(group_states, tid, nothing)
end

# Every NCCL call that can join a group has to be preceded by this check, or a task that runs
# while the group owner yields would add its operations to the owner's group. Nothing that can
# yield may happen between the check and the native call (see `submit`).
function check_group_owner()
    state = group_state()
    if state !== nothing && state.owner !== current_task()
        throw(ConcurrencyViolationError(
            "another task has an NCCL group open on this thread; NCCL operations from other tasks would join it. Do not yield to tasks that use NCCL from within `NCCL.group`."))
    end
    return
end

# Keep `objs` alive until the open group, if any, has ended. Must be called by the owner of
# the group (see `check_group_owner`).
function retain_in_group(objs...)
    state = group_state()
    state === nothing || push!(state.roots, objs)
    return
end

function enter_group(tid::Int, task::Task)
    @lock group_lock begin
        state = get(group_states, tid, nothing)
        if state === nothing
            group_states[tid] = GroupState(task, 1, Any[])
        elseif state.owner === task
            state.depth += 1
        else
            throw(ConcurrencyViolationError(
                "another task has an NCCL group open on this thread; groups cannot be opened by multiple tasks on the same thread."))
        end
    end
    return
end

function leave_group(tid::Int)
    @lock group_lock begin
        state = group_states[tid]
        state.depth -= 1
        state.depth == 0 && delete!(group_states, tid)
    end
    return
end

"""
    NCCL.group(f)

Evaluate `f()` as an NCCL group: operations issued by `f` are only submitted when it returns,
which makes it possible to issue a collective on several communicators from a single task,
or to combine multiple point-to-point operations without deadlocking.

Groups can be nested; operations are submitted when the outermost group ends.

Because operations in a group are only enqueued on their streams when the outermost group
ends, the buffers passed to them must not be read, written, freed, or handed to another task
or stream until then. NCCL.jl keeps these buffers, as well as the streams and communicators
of the operations, alive until the group ends, so they do not need to be referenced
elsewhere. Once the group has ended, the operations are enqueued on their streams, and are
ordered like any other work on those streams: the usual CUDA.jl rules for using arrays from
other tasks and streams apply.

NCCL keeps track of groups per OS thread, so the following rules apply:

- The current task is pinned to its thread, and stays pinned after the group ends (the same
  happens when a task uses `@async`; tasks created by `f` may still be running afterwards
  and rely on it). All operations of the group must be issued from this task; operations
  issued from other tasks, e.g. `Threads.@spawn`, are not part of the group.
- `f` must not yield to another task that uses NCCL on the same thread. If that happens,
  the other task's NCCL calls throw a `ConcurrencyViolationError` rather than silently
  joining the group.

If `f` throws, the group is still ended, and the exception thrown by `f` is rethrown, even if
ending the group fails as well. What happens to the operations issued before the exception
depends on whether NCCL recorded an error in its group state:

- NCCL records the errors of operations issued in the group, e.g. when it rejects their
  arguments. Then the outermost group end discards all queued operations, and submits
  nothing.
- Other exceptions don't affect the group, and ending it submits the operations that were
  issued. This includes Julia exceptions, e.g. from the argument checks of NCCL.jl, which run
  before NCCL is called, and `NCCLError`s from NCCL calls that don't record their errors in
  the group, e.g. when creating an `NCCL.UniqueID`. If peers wait for operations that were
  never issued, this may deadlock.

Communicators are blocking, so ending the group can block, e.g. until the peers have issued
their matching operations, before the exception reaches any handler in the caller. Recovering
from such a failure, e.g. by having all ranks call `NCCL.abort`, therefore has to be arranged
out of band for the whole job, and can't rely on the failing task. Aborting a blocking
communicator is not guaranteed to interrupt a call that is blocked inside NCCL, so
terminating the job is the only reliable way out of a hang.

# Examples
```julia
NCCL.group() do
    for (comm, sendbuf, recvbuf) in zip(comms, sendbufs, recvbufs)
        NCCL.Allreduce!(sendbuf, recvbuf, +, comm)
    end
end
```

# External links
- [`ncclGroupStart`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupstart)
- [`ncclGroupEnd`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupend)
- [Group calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
"""
function group(f)
    task = current_task()
    # a running task that becomes sticky stays on the thread it is running on. this is never
    # undone: `@async` in `f` also makes this task sticky, and the child may outlive the group.
    task.sticky = true
    tid = Threads.threadid()
    enter_group(tid, task)
    try
        ncclGroupStart()
        local ret
        try
            ret = f()
        catch
            try
                ncclGroupEnd()
            catch
                # the original exception is more informative
            end
            rethrow()
        end
        ncclGroupEnd()
        return ret
    finally
        leave_group(tid)
    end
end
