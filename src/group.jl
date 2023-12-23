# Group calls

"""
    NCCL.groupStart()

Start a NCCL group call.

# External links
- [`ncclGroupStart`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupstart)
"""
groupStart() = ncclGroupStart()

"""
    NCCL.groupEnd()

End a NCCL group call

# External links
- [`ncclGroupEnd`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupend)
"""
groupEnd() = ncclGroupEnd()

"""
    NCCL.group(f)

Evaluate `f()` inside between [`NCCL.groupStart()`](@ref) and [`NCCL.groupEnd()`](@ref).
"""
function group(f)
    groupStart()
    try
        f()
    finally
        groupEnd()
    end
end