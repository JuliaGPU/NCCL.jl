function version()
    ver = Ref{Cint}()
    ncclGetVersion(ver)
    major, ver = divrem(ver[], 1000)
    minor, patch = divrem(ver, 100)

    VersionNumber(major, minor, patch)
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

function ncclReductionOp(T::DataType)
    if T == typeof(+)
        return ncclSum
    elseif T == typeof(*)
        return ncclProd
    elseif T == typeof(min)
        return ncclMin
    elseif T == typeof(max)
        return ncclMax
    else
        throw(ArgumentError("ncclReductionOp equivalent for input function type $T does not exist!"))
    end
end
