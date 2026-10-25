using Documenter
using DocumenterCodeBlocks: CodeBlocks
using NCCL, CUDA

const ci = get(ENV, "CI", "") == "true"

makedocs(
    sitename = "NCCL",
    plugins = [
        CodeBlocks(; line_counter = :named),
    ],
    format = Documenter.HTML(),
    modules = [NCCL]
)

if ci
    deploydocs(
        repo = "github.com/JuliaGPU/NCCL.jl",
        target = "build",
        push_preview = true,
        devbranch = "master",
        forcepush = true,
    )
end
