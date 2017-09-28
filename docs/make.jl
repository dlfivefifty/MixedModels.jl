using Documenter, MixedModels, StatsBase

makedocs(
    format = :html,
    sitename = "MixedModels.jl",
    modules = [MixedModels],
    pages = ["index.md",
             "constructors.md",
             "extractors.md",
             "fitting.md"]
)

deploydocs(
    repo    = "github.com/dmbates/MixedModels.jl.git",
    julia   = "0.6",
    osname  = "linux",
    target  = "build",
    deps    = nothing,
    make    = nothing
)
