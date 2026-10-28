using Documenter
using DocumenterCodeBlocks: CodeBlocks
using NCCL

const ci = get(ENV, "CI", "") == "true"

makedocs(
    sitename = "NCCL.jl",
    plugins = [
        CodeBlocks(; line_counter = :named),
    ],
    format = Documenter.HTML(prettyurls = ci),
    modules = [NCCL],
    pages = [
        "Introduction" => "index.md",
        "Getting started" => "getting_started.md",
        "Manual" => [
            "communicators.md",
            "operations.md",
            "groups.md",
            "streams.md",
            "errors.md",
        ],
        "API reference" => "api.md",
        "Migrating from NCCL.jl 0.1" => "migration.md",
    ],
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
