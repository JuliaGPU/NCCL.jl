using Clang.Generators, NCCL_jll

cd(@__DIR__)

include_dir = normpath(NCCL_jll.artifact_dir, "include")

# wrapper generator options
options = load_options(joinpath(@__DIR__, "generator.toml"))

# add compiler flags, e.g. "-DXXXXXXXXX"
args = get_default_args()
push!(args, "-I$include_dir")

headers = [joinpath(include_dir, "nccl.h")]
# there is also an experimental `detect_headers` function for auto-detecting top-level headers in the directory
# headers = detect_headers(clang_dir, args)

# create context
ctx = create_context(headers, args, options)

function rewriter!(ctx, options)
    for node in get_nodes(ctx.dag)
        # remove aliases for function names
        #
        # when NVIDIA changes the behavior of an API, they version the function
        # (`cuFunction_v2`), and sometimes even change function names. To maintain backwards
        # compatibility, they ship aliases with their headers such that compiled binaries
        # will keep using the old version, and newly-compiled ones will use the developer's
        # CUDA version. remove those, since we target multiple CUDA versions.
        #
        # remove this if we ever decide to support a single supported version of CUDA.
        if node isa ExprNode{<:AbstractMacroNodeType}
            isempty(node.exprs) && continue
            expr = node.exprs[1]
            if Meta.isexpr(expr, :const)
                expr = expr.args[1]
            end
            if Meta.isexpr(expr, :(=))
                lhs, rhs = expr.args
                if rhs isa Expr && rhs.head == :call
                    name = string(rhs.args[1])
                    if endswith(name, "STRUCT_SIZE")
                        rhs.head = :macrocall
                        rhs.args[1] = Symbol("@", name)
                        insert!(rhs.args, 2, nothing)
                    end
                end
                isa(lhs, Symbol) || continue
                if Meta.isexpr(rhs, :call) && rhs.args[1] in (:__CUDA_API_PTDS, :__CUDA_API_PTSZ)
                    rhs = rhs.args[2]
                end
                isa(rhs, Symbol) || continue
                lhs, rhs = String(lhs), String(rhs)
                function get_prefix(str)
                    # cuFooBar -> cu
                    isempty(str) && return nothing
                    islowercase(str[1]) || return nothing
                    for i in 2:length(str)
                        if isuppercase(str[i])
                            return str[1:i-1]
                        end
                    end
                    return nothing
                end
                lhs_prefix = get_prefix(lhs)
                lhs_prefix === nothing && continue
                rhs_prefix = get_prefix(rhs)
                if lhs_prefix == rhs_prefix
                    @debug "Removing function alias: `$expr`"
                    empty!(node.exprs)
                end
            end
        end

        if Generators.is_function(node) && !Generators.is_variadic_function(node)
            expr = node.exprs[1]
            call_expr = expr.args[2].args[1].args[3]    # assumes `@ccall`
            #=

            target_expr = call_expr.args[1].args[1]
            fn = String(target_expr.args[2].value)

            # look up API options for this function
            fn_options = Dict{String,Any}()
            if haskey(options, "api")
                names = [fn]

                # _64 aliases are used by CUBLAS with Int64 arguments. they otherwise have
                # an idential signature, so we can reuse the same type rewrites.
                if endswith(fn, "_64")
                    push!(names, fn[1:end-3])
                end

                # the exact name is always checked first, so it's always possible to
                # override the type rewrites for a specific function
                # (e.g. if a _64 function ever passes a `Ptr{Cint}` index).
                for name in names
                    if haskey(options["api"], name)
                        fn_options = options["api"][name]
                        break
                    end
                end
            end

            # rewrite pointer argument types
            arg_exprs = call_expr.args[1].args[2:end]
            argtypes = get(fn_options, "argtypes", Dict())
            for (arg, typ) in argtypes
                i = parse(Int, arg)
                arg_exprs[i].args[2] = Meta.parse(typ)
            end

            # insert `initialize_context()` before each function with a `ccall`
            if get(fn_options, "needs_context", true)
                pushfirst!(expr.args[2].args, :(initialize_context()))
            end

            # insert `@checked` before each function with a `ccall` returning a checked type`
            
            checked_types = if haskey(options, "api")
                get(options["api"], "checked_rettypes", Dict())
            else
                String[]
            end
            =#
            rettyp = call_expr.args[2]
            if rettyp isa Symbol && String(rettyp) == "ncclResult_t"
                node.exprs[1] = Expr(:macrocall, Symbol("@checked"), nothing, expr)
            end
        end
    end
end


# run generator
build!(ctx, BUILDSTAGE_NO_PRINTING)
rewriter!(ctx, options)

build!(ctx, BUILDSTAGE_PRINTING_ONLY)

