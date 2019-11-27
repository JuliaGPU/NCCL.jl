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
    else
        return "unknown status"
    end
end

macro check(nccl_func)
    quote
        local err::ncclResult_t
        err = $(esc(nccl_func::Expr))
        if err != ncclSuccess
            throw(NCCLError(err))
        end
        err
    end
end
