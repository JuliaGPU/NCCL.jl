# Group calls

"""
    NCCL.groupStart()

Start a NCCL group call
"""
groupStart() = ncclGroupStart()

"""
    NCCL.groupEnd()

End a NCCL group call
"""
groupEnd() = ncclGroupEnd()
