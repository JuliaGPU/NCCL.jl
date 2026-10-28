# Buffers
#
# NCCL operates on ranges of device memory, so every buffer has to be stored contiguously.
# CUDACore represents most contiguous views and reshapes of a `CuArray` as a `CuArray`
# sharing its memory; the others are `SubArray`s, whose strides tell whether they are
# contiguous.

iscontiguous(buf) = false
iscontiguous(buf::DenseCuArray) = true
function iscontiguous(buf::StridedCuArray)
    expected = 1
    for (n, s) in zip(size(buf), strides(buf))
        n == 0 && return true
        n == 1 || s == expected || return false
        expected *= n
    end
    return true
end

# The allocation behind a buffer, as needed by `CUDACore.with_managed`. CUDACore has no
# public accessor for it (only an internal `managed_memory` on newer versions), so this
# relies on the internal `data` field of `CuArray`.
managed_memory(buf::CuArray) = buf.data[]
managed_memory(buf::StridedCuArray) = managed_memory(parent(buf))

buffer_device(buf::DenseCuArray) = CUDACore.device(buf)
buffer_device(buf::StridedCuArray) = buffer_device(parent(buf))

function check_buffer(comm::Communicator, buf, name)
    if !iscontiguous(buf)
        throw(ArgumentError(
            "$name must be stored contiguously in device memory (a `CuArray`, or a contiguous view or reshape of one), got a $(summary(buf)); copy it to a `CuArray` first"))
    end
    dev = buffer_device(buf)
    if dev != comm.device
        throw(ArgumentError(
            "$name is on device $(deviceid(dev)), but the communicator is on device $(deviceid(comm.device))"))
    end
    return
end

# Some operations ignore a buffer on all ranks but the root, so other ranks can pass `nothing`
# instead. Returns whether the buffer was passed.
function check_root_buffer(comm::Communicator, buf, name, root::Cint)
    if buf === nothing
        comm.rank == root && throw(ArgumentError(
            "$name is required on the root rank, but `nothing` was passed"))
        return false
    end
    check_buffer(comm, buf, name)
    return true
end

function check_eltypes(sendbuf, recvbuf)
    if eltype(sendbuf) != eltype(recvbuf)
        throw(ArgumentError(
            "sendbuf and recvbuf must have the same element type, got $(eltype(sendbuf)) and $(eltype(recvbuf))"))
    end
    return
end

function check_length(buf, len, name)
    if length(buf) != len
        throw(DimensionMismatch("$name must have $len elements, got $(length(buf))"))
    end
    return
end

# The length of the blocks that `buf` consists of, one for every rank.
function block_length(comm::Communicator, buf, name)
    len, rem = divrem(length(buf), comm.nranks)
    if rem != 0
        throw(DimensionMismatch(
            "$name must consist of a block for each of the $(comm.nranks) ranks, but its length $(length(buf)) is not a multiple of $(comm.nranks)"))
    end
    return len
end

# validate a root or peer rank, after converting it to its C type
function check_peer(comm::Communicator, peer::Cint, name)
    if !(0 <= peer < comm.nranks)
        throw(ArgumentError("$name must be in [0, $(comm.nranks)), got $peer"))
    end
    return
end

# Operations work either on disjoint buffers, or in place: NCCL then expects the send buffer
# to start `inplace_offset` bytes after the start of the receive buffer, as described by
# `inplace` (only the layouts that NCCL documents for each operation are valid; `nothing`
# for operations that can't be performed in place). This needs the addresses of the buffers,
# so it is checked during `submit`.
function check_overlap(sendptr::CuPtr, sendbuf, recvptr::CuPtr, recvbuf;
                       inplace_offset::Int=0,
                       inplace::Union{String,Nothing}="sendbuf and recvbuf must be the same")
    sendbytes = length(sendbuf) * Base.elsize(sendbuf)
    recvbytes = length(recvbuf) * Base.elsize(recvbuf)
    (sendbytes == 0 || recvbytes == 0) && return
    sendstart, recvstart = UInt(sendptr), UInt(recvptr)
    disjoint = sendstart >= recvstart + recvbytes || recvstart >= sendstart + sendbytes
    disjoint && return
    if inplace === nothing
        throw(ArgumentError(
            "sendbuf and recvbuf overlap; they must be disjoint, as this operation cannot be performed in place"))
    end
    reinterpret(Int, sendstart - recvstart) == inplace_offset && return
    throw(ArgumentError(
        "sendbuf and recvbuf overlap; they must be disjoint, or for an in-place operation, $inplace"))
end
