export NCCLError

"""
    NCCLError(code::ncclResult_t, [msg::String])

An error returned by a NCCL function. `code` is the result code, and `msg` the error message
NCCL reported for it (see `ncclGetLastError`), or an empty string if there is none.

NCCL keeps a single, process-wide last error message, which it only updates when it logs a
warning and never clears. `msg` is therefore only set when that message changed since
NCCL.jl last read it, so that an error which NCCL didn't log doesn't repeat the message of an
earlier one. As a consequence, `msg` is empty when NCCL reports the same message twice in a
row, and may describe an error raised concurrently by another thread. Set the environment
variable `NCCL_DEBUG=WARN` to see all of NCCL's messages.
"""
struct NCCLError <: Exception
    code::ncclResult_t
    msg::String
end

const last_error_lock = ReentrantLock()
const last_error_seen = Ref("")

# NCCL's message is only meaningful if it changed since we last looked: NCCL overwrites it
# when it logs a warning, but leaves it untouched for errors it doesn't log.
function last_error_message()
    @lock last_error_lock begin
        msg = unsafe_string(ncclGetLastError(C_NULL))
        msg == last_error_seen[] && return ""
        last_error_seen[] = msg
        return msg
    end
end

NCCLError(code::ncclResult_t) = NCCLError(code, last_error_message())

function Base.showerror(io::IO, err::NCCLError)
    print(io, "NCCLError: ", description(err.code), " (code ", Int(err.code), ", ",
          err.code, ")")
    if !isempty(err.msg)
        print(io, "\n", err.msg)
    end
end

function description(code::ncclResult_t)
    if code == ncclSuccess
        return "function succeeded"
    elseif code == ncclUnhandledCudaError
        return "a call to a CUDA function failed"
    elseif code == ncclSystemError
        return "a call to the system failed"
    elseif code == ncclInternalError
        return "an internal check failed. This is either a bug in NCCL or due to memory corruption"
    elseif code == ncclInvalidArgument
        return "one argument has an invalid value"
    elseif code == ncclInvalidUsage
        return "the call to NCCL is incorrect. This is usually reflecting a programming error"
    elseif code == ncclRemoteError
        return "a call failed possibly due to a network error or a remote process exiting prematurely"
    elseif code == ncclInProgress
        return "a NCCL operation on the communicator is being enqueued and is being progressed in the background"
    elseif code == ncclTimeout
        return "a NCCL operation timed out"
    else
        return "unknown result code"
    end
end
