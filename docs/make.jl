using Documenter
using NCCL, CUDA

const ci = get(ENV, "CI", "") == "true"

makedocs(
    sitename = "NCCL",
    format = Documenter.HTML(),
    modules = [NCCL]
)

if ci
    deploydocs(
        repo = "github.com/JuliaGPU/NCCL.jl",
        target = "build",
        push_preview = true,
        devbranch = "main",
        forcepush = true,
    )
end