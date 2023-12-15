
export NCCLError

struct NCCLError <: Exception
    code::ncclResult_t
    msg::AbstractString
end
Base.show(io::IO, err::NCCLError) = print(io, "NCCLError(code $(err.code), $(err.msg))")

function NCCLError(code::ncclResult_t)
    msg = status_message(code)
    return NCCLError(code, msg)
end

function status_message(status)
    if status == ncclSuccess
        return "function succeeded"
    elseif status == ncclUnhandledCudaError
        return "a call to a CUDA function failed"
    elseif status == ncclSystemError
        return "a call to the system failed"
    elseif status == ncclInternalError
        return "an internal check failed. This is either a bug in NCCL or due to memory corruption"
    elseif status == ncclInvalidArgument
        return "one argument has an invalid value"
    elseif status == ncclInvalidUsage
        return "the call to NCCL is incorrect. This is usually reflecting a programming error"
    elseif status == ncclRemoteError
        return "A call failed possibly due to a network error or a remote process exiting prematurely."
    elseif status == ncclInProgress
        return "A NCCL operation on the communicator is being enqueued and is being progressed in the background."
    else
        return "unknown status"
    end
end
