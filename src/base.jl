function version()
    ver_r = Ref{Cint}()
    ncclGetVersion(ver_r)
    ver = ver_r[]

    # nccl.h defines this as:
    #define NCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

    if ver < 2900
        major, ver = divrem(ver, 1000)
        minor, patch = divrem(ver, 100)
    else
        major, ver = divrem(ver, 10000)
        minor, patch = divrem(ver, 100)
    end
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
