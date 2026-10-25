module NCCLMPIExt

using NCCL, MPI
using NCCL: Communicator, Config, UniqueID
using NCCL.LibNCCL: NCCL_UNIQUE_ID_BYTES
using CUDACore: CuDevice

# Rank 0 creates the unique ID and broadcasts it. Only one rank may create it, as creating
# an ID starts a bootstrap listener that the communicator's ranks connect to. The ID is
# preceded by a flag that tells the other ranks whether rank 0 succeeded, so that they fail
# instead of waiting for a communicator that rank 0 won't create.
function broadcast_id(mpicomm::MPI.Comm)
    buf = zeros(UInt8, 1 + NCCL_UNIQUE_ID_BYTES)
    if MPI.Comm_rank(mpicomm) == 0
        try
            buf[2:end] = Vector{UInt8}(UniqueID())
            buf[1] = 1
        catch
            MPI.Bcast!(buf, mpicomm; root=0)
            rethrow()
        end
    end
    MPI.Bcast!(buf, mpicomm; root=0)
    buf[1] == 1 ||
        error("rank 0 of the MPI communicator failed to create the NCCL unique ID")
    return UniqueID(@view buf[2:end])
end

# `device` defaults to `nothing`, i.e. the current device, rather than to `CUDACore.device()`:
# a default evaluated here could throw on one rank before the ranks agree on their arguments.
function NCCL.Communicator(mpicomm::MPI.Comm;
                           device::Union{CuDevice,Integer,Nothing}=nothing,
                           config::Union{Config,Nothing}=nothing)
    MPI.Initialized() && !MPI.Finalized() ||
        error("MPI is not initialized; call `MPI.Init()` before creating a communicator from an MPI communicator")
    nranks = MPI.Comm_size(mpicomm)
    rank = MPI.Comm_rank(mpicomm)
    prepared = prepare(mpicomm, nranks, rank, device, config)
    id = broadcast_id(mpicomm)
    return NCCL.init_prepared(prepared, id)
end

# Check the arguments on every rank before any rank joins the communicator, and agree on
# the outcome: a rank that fails has to make the others fail too, as they would otherwise
# wait for it in `ncclCommInitRank`.
function prepare(mpicomm::MPI.Comm, nranks, rank, device, config)
    prepared = try
        NCCL.prepare_init(nranks, rank, device, config)
    catch
        MPI.Allreduce(1, +, mpicomm)
        rethrow()
    end
    nfailed = MPI.Allreduce(0, +, mpicomm)
    nfailed == 0 ||
        error("$nfailed rank(s) of the MPI communicator failed before initializing the NCCL communicator")
    return prepared
end

function NCCL.Communicator(f, mpicomm::MPI.Comm; kwargs...)
    NCCL.with_communicators(f, Communicator(mpicomm; kwargs...))
end

end
