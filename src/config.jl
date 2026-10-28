# Unique IDs and communicator configuration

using .LibNCCL: NCCL_UNIQUE_ID_BYTES, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_PTR,
                NCCL_API_MAGIC, NCCL_VERSION_CODE

"""
    NCCL.UniqueID()
    NCCL.UniqueID(bytes::AbstractVector{UInt8})

The identifier that all ranks of a communicator pass to [`Communicator`](@ref), so that
they find each other.

`UniqueID()` creates a new identifier (`ncclGetUniqueId`). This is done by one rank, which
then has to send it to all other ranks by some other means, e.g. MPI, a socket, or a file.
`UniqueID` is a plain bits type of 128 bytes: `Vector{UInt8}(id)` returns these bytes, and
`UniqueID(bytes)` reconstructs the identifier from them. It can also be sent as-is with
anything that handles bits types, e.g. `Serialization` or `Distributed`.

Creating an identifier also starts a bootstrap listener on the creating process, so the rank
that creates it has to take part in the communicator.

# Examples
```julia
if rank == 0
    id = NCCL.UniqueID()
    write(io, Vector{UInt8}(id))
else
    id = NCCL.UniqueID(read(io, 128))
end
```

# External links
- [`ncclGetUniqueId`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclgetuniqueid)
"""
const UniqueID = LibNCCL.ncclUniqueId

function UniqueID()
    r = Ref{UniqueID}()
    ncclGetUniqueId(r)
    return r[]
end

function UniqueID(bytes::AbstractVector{UInt8})
    length(bytes) == NCCL_UNIQUE_ID_BYTES || throw(ArgumentError(
        "a unique ID consists of $NCCL_UNIQUE_ID_BYTES bytes, got $(length(bytes))"))
    UniqueID(ntuple(i -> reinterpret(Cchar, bytes[begin+i-1]), Val(NCCL_UNIQUE_ID_BYTES)))
end

Base.Vector{UInt8}(id::UniqueID) = UInt8[reinterpret(UInt8, c) for c in id.internal]

function Base.show(io::IO, id::UniqueID)
    print(io, "NCCL.UniqueID(0x", bytes2hex(Vector{UInt8}(id)), ")")
end


"""
    NCCL.Config(; kwargs...)

Configuration for creating a communicator, passed as the `config` keyword argument of
[`Communicator`](@ref) and [`Communicators`](@ref). Every option defaults to `nothing`,
which leaves the choice to NCCL (or the corresponding environment variable, which takes
precedence over most options).

- `comm_name::String`: a name for the communicator, used by NCCL in its logs and profiling.
- `net_name::String`: the network module to use, e.g. `"IB"` or `"Socket"`, or the name of
  an external network plugin (case-insensitive).
- `cga_cluster_size::Integer`: the cooperative group array size of the kernels NCCL launches
  (0 to 8).
- `min_ctas::Integer`, `max_ctas::Integer`: the minimal and maximal number of CTAs that NCCL
  should use for each kernel (1 to 32).
- `nvls_ctas::Integer`: the total number of CTAs to use for NVLink SHARP kernels.
- `cta_policy::Integer`: a bitmask of `NCCL.LibNCCL.NCCL_CTA_POLICY_*` flags.
- `split_share::Bool`: whether communicators split from this one may share its resources.
- `traffic_class::Integer`: the traffic class to use for network operations; its meaning
  depends on the network plugin.
- `collnet_enable::Bool`: whether to use IB SHARP.
- `n_channels_per_net_peer::Integer`: the number of network channels used for each peer in
  point-to-point communication.
- `max_p2p_peers::Integer`: the maximal number of peers a rank communicates with at the same
  time using point-to-point communication.
- `graph_usage_mode::Integer`: how the communicator is used with CUDA graphs: 0 (no graphs),
  1 (one graph) or 2 (multiple graphs, or graphs mixed with regular launches; the default).

See the NCCL documentation for the details of each option. Options that the NCCL library in
use doesn't know about are ignored.

Communicators are always created in blocking mode: nonblocking communicators, which require
polling for the completion of every NCCL call, are not supported yet. NCCL lets the
environment variable `NCCL_COMM_BLOCKING` override this, and also reads that variable from
its configuration files (the file named by `NCCL_CONF_FILE`, or `~/.nccl.conf`, and
`/etc/nccl.conf`). Those files only set variables that are not in the environment yet, so
when NCCL.jl is loaded, or creates a communicator, while `NCCL_COMM_BLOCKING` is unset, it
sets it to `1`, which NCCL's configuration files can then not override. Processes started
afterwards inherit this setting. NCCL reads the variable only once per process, so this
cannot take effect if another library already initialized NCCL in the process (e.g., by
creating a communicator) before NCCL.jl was loaded. If `NCCL_COMM_BLOCKING` is set to a value that NCCL reads as 0, creating a
communicator throws an error. NCCL only parses the leading number of the value (and older
versions read a value without digits as 0), so this includes values like `0x0` or `0abc`.

# External links
- [`ncclConfig_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t)
"""
Base.@kwdef struct Config
    comm_name::Union{Nothing,String} = nothing
    net_name::Union{Nothing,String} = nothing
    cga_cluster_size::Union{Nothing,Cint} = nothing
    min_ctas::Union{Nothing,Cint} = nothing
    max_ctas::Union{Nothing,Cint} = nothing
    nvls_ctas::Union{Nothing,Cint} = nothing
    cta_policy::Union{Nothing,Cint} = nothing
    split_share::Union{Nothing,Bool} = nothing
    traffic_class::Union{Nothing,Cint} = nothing
    collnet_enable::Union{Nothing,Bool} = nothing
    n_channels_per_net_peer::Union{Nothing,Cint} = nothing
    max_p2p_peers::Union{Nothing,Cint} = nothing
    graph_usage_mode::Union{Nothing,Cint} = nothing
end

function Base.show(io::IO, config::Config)
    print(io, "NCCL.Config(")
    fields = filter(f -> getfield(config, f) !== nothing, fieldnames(Config))
    join(io, ("$f=$(repr(getfield(config, f)))" for f in fields), ", ")
    print(io, ")")
end

config_int(x::Nothing) = NCCL_CONFIG_UNDEF_INT
config_int(x) = Cint(x)

# the pointers in the result are only valid as long as `config` is preserved
config_str(x::Nothing) = Cstring(NCCL_CONFIG_UNDEF_PTR)
config_str(x::String) = Base.unsafe_convert(Cstring, x)

# The equivalent of `NCCL_CONFIG_INITIALIZER`, with the given options set. The strings in the
# result point into `config`, which has to be preserved for as long as the result is used.
function LibNCCL.ncclConfig_t(config::Config)
    ncclConfig_t(
        sizeof(ncclConfig_t),                       # size
        NCCL_API_MAGIC,                             # magic
        NCCL_VERSION_CODE,                          # version
        1,                                          # blocking
        config_int(config.cga_cluster_size),        # cgaClusterSize
        config_int(config.min_ctas),                # minCTAs
        config_int(config.max_ctas),                # maxCTAs
        config_str(config.net_name),                # netName
        config_int(config.split_share),             # splitShare
        config_int(config.traffic_class),           # trafficClass
        config_str(config.comm_name),               # commName
        config_int(config.collnet_enable),          # collnetEnable
        config_int(config.cta_policy),              # CTAPolicy
        NCCL_CONFIG_UNDEF_INT,                      # shrinkShare
        config_int(config.nvls_ctas),               # nvlsCTAs
        config_int(config.n_channels_per_net_peer), # nChannelsPerNetPeer
        NCCL_CONFIG_UNDEF_INT,                      # nvlinkCentricSched
        config_int(config.graph_usage_mode),        # graphUsageMode
        NCCL_CONFIG_UNDEF_INT,                      # numRmaCtx
        config_int(config.max_p2p_peers),           # maxP2pPeers
        NCCL_CONFIG_UNDEF_INT,                      # graphStreamOrdering
        NCCL_CONFIG_UNDEF_INT,                      # launchOrderImplicit
        NCCL_CONFIG_UNDEF_INT,                      # numRmaSig
        NCCL_CONFIG_UNDEF_INT,                      # rmaEagerInit
        NCCL_CONFIG_UNDEF_INT,                      # hostCftMode
        NCCL_CONFIG_UNDEF_INT,                      # nvlsHostMode
    )
end

# NCCL reads NCCL_COMM_BLOCKING once, from the environment or from a configuration file,
# and lets it override `config.blocking`. Configuration files only set variables that are
# not in the environment yet, so setting it before NCCL reads it keeps communicators
# blocking whatever the files say. Values set by the user are checked by `check_blocking`.
function force_blocking()
    haskey(ENV, "NCCL_COMM_BLOCKING") || (ENV["NCCL_COMM_BLOCKING"] = "1")
    return
end

# Would NCCL read `str` as a request for nonblocking communicators? NCCL parses the value
# with `strtoll` (ignoring out-of-range values), truncates it to an `int`, and only
# overrides the configuration with 0 or 1. NCCL 2.30 and older also read a value without
# any digits as 0.
function nonblocking(str::String)
    isempty(str) && return false
    Libc.errno(0)
    value = ccall(:strtoll, Clonglong, (Cstring, Ptr{Ptr{Cchar}}, Cint), str, C_NULL, 0)
    Libc.errno() == 0 || return false
    return value % Cint == 0
end

# A nonblocking communicator would make NCCL calls return before completing, which the rest
# of the package doesn't handle.
function check_blocking()
    # the user may have unset the variable since the package was loaded
    force_blocking()
    str = get(ENV, "NCCL_COMM_BLOCKING", "")
    if nonblocking(str)
        error("NCCL.jl does not support nonblocking communicators, but NCCL_COMM_BLOCKING=$(repr(str)) requests them; unset NCCL_COMM_BLOCKING or set it to 1")
    end
    return
end
