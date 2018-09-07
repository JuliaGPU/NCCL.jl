# Basic library functionality

const ncclResult_t = Cint

struct NCCLError <: Exception
    code::ncclResult_t

    NCCLError(code) = new(code)
end

# known error constants
const return_codes = Dict{ncclResult_t,Symbol}(
    0  => :SUCCESS,

    1  => :UNHANDLED_CUDA_ERROR,
    2  => :SYSTEM_ERROR,
    3  => :INTERNAL_ERROR,
    4  => :INVALID_DEVICE_POINTER,
    5  => :INVALID_RANK,
    6  => :UNSUPPORTED_DEVICE_COUNT,
    7  => :DEVICE_NOT_FOUND,
    8  => :INVALID_DEVICE_INDEX,
    9  => :LIB_WRAPPER_NOT_SET,
    10 => :CUDA_MALLOC_FAILED,
    11 => :RANK_MISMATCH,
    12 => :INVALID_ARGUMENT,
    13 => :INVALID_TYPE,
    14 => :INVALID_OPERATION
)
for code in return_codes
    @eval const $(code[2]) = NCCLError($(code[1]))
end

macro apicall(fun, argtypes, args...)
    isa(fun, QuoteNode) || error("first argument to @apicall should be a symbol")

    configured || return :(error("NCCL.jl has not been configured."))

    return quote
        status = ccall(($fun, libnccl), ncclResult_t,
                       $(esc(argtypes)), $(map(esc, args)...))

        if status != SUCCESS.code
            err = NCCLError(status)
            throw(err)
        end
    end
end

function description(err::NCCLError)
    str_ref = Ref{Cstring}()
    @apicall(:ncclGetErrorString, (ncclResult_t, Ptr{Cstring}), err.code, str_ref)
    unsafe_string(str_ref[])
end

function Base.showerror(io::IO, err::NCCLError)
    @printf(io, "NCCL error %d: %s", err.code, description(err))
end
