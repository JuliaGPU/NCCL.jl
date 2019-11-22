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

@enum ncclRedOp_t::Cint begin
    ncclSum    = 0
    ncclProd   = 1 
    ncclMax    = 2 
    ncclMin    = 3 
    ncclNumOps = 4
end

@enum ncclDataType_t::Cint begin
    ncclInt8    = 0
    #ncclChar    = 0
    ncclUint8   = 1
    ncclInt32   = 2
    #ncclInt     = 2
    ncclUint32  = 3
    ncclInt64   = 4
    ncclUInt64  = 5
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclFloat64 = 8
    ncclNumTypes = 9 
end

function ncclDataType(T::DataType)
    if T == Float32
        return ncclFloat32 
    elseif T == Float16
        return ncclFloat16 
    elseif T == Float64
        return ncclFloat64 
    elseif T == Int8
        return ncclInt8 
    elseif T == Char 
        return ncclInt8 
    elseif T == Int32
        return ncclInt32 
    elseif T == UInt32
        return ncclUint32 
    elseif T == Int64
        return ncclInt64 
    elseif T == UInt64
        return ncclUint64 
    else
        throw(ArgumentError("ncclDataType equivalent for input type $T does not exist!"))
    end
end
