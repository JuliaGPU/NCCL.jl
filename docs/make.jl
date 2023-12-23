using Documenter
using NCCL

makedocs(
    sitename = "NCCL",
    format = Documenter.HTML(),
    modules = [NCCL]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
