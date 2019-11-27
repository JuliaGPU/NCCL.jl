import MPI
import NCCL
using CuArrays
using CUDAdrv
using CUDAnative

MPI.Init()
comm   = MPI.COMM_WORLD
myrank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Issues:
# - Avoid allocations during allReduce

print(stdout, ENV)

@info "MPI initialized" myrank nranks

if myrank == 0
    uid = NCCL.UniqueID()
else
    uid = nothing
end
uid = MPI.bcast(uid, 0, comm)::NCCL.UniqueID

dev = CuDevice(parse(Int, first(split(ENV["CUDA_VISIBLE_DEVICES"], ","))))
@info "NCCL uid bcast" myrank uid dev
CUDAnative.device!(dev)

cuComm = NCCL.Communicator(nranks, uid, myrank)

recv = CuArray{Float32}(undef, 1024)
send = CuArray{Float32}(undef, 1024)
fill!(send, float(myrank))

# Stream to do communication on
stream = CuStream()

event = CuEvent(CUDAdrv.EVENT_DISABLE_TIMING)
NCCL.allReduce(+, send, recv, cuComm, stream)
CUDAdrv.record(event, stream) # mark communication as done

# Enqueue a marker on CuDefaultStream to wait on the communication
wait(event)
# Now do work on CuDefaultStream()
# ...

synchronize(stream)
NCCL.destroy(cuComm)
MPI.Finalize()
