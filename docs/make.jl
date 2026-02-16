using Documenter
using Relevent

DocMeta.setdocmeta!(Relevent, :DocTestSetup, :(using Relevent); recursive=true)

makedocs(
    sitename = "Relevent.jl",
    modules = [Relevent],
    authors = "Statistical Network Analysis with Julia",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://Statistical-network-analysis-with-Julia.github.io/Relevent.jl",
        edit_link = "main",
    ),
    repo = "https://github.com/Statistical-network-analysis-with-Julia/Relevent.jl/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Interaction History" => "guide/history.md",
            "Advanced Statistics" => "guide/statistics.md",
            "Timing Models" => "guide/timing.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Statistics" => "api/statistics.md",
            "Estimation" => "api/estimation.md",
        ],
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(
    repo = "github.com/Statistical-network-analysis-with-Julia/Relevent.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev", # serve dev docs at /stable until a release is tagged
        "dev" => "dev",
    ],
    push_preview = true,
)
