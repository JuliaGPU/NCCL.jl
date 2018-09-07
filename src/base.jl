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
    4  => :INVALID_ARGUMENT,
    5  => :INVALID_USAGE,
    6  => :NUM_RESULTS,
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
