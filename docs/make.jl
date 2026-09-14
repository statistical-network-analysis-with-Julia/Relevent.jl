using Documenter
using Relevent

DocMeta.setdocmeta!(Relevent, :DocTestSetup, :(using Relevent); recursive=true)

makedocs(
    sitename = "Relevent.jl",
    modules = [Relevent],
    authors = "Statistical Network Analysis with Julia",
    format = Documenter.HTML(
        prettyurls = get(ENV, "DOCS_PRETTY_URLS", get(ENV, "CI", "false")) == "true",
        canonical = "https://statistical-network-analysis-with-Julia.github.io/Relevent.jl/dev/",
        assets = ["assets/snwj-docs.css", "assets/snwj-docs.js"],
        footer = "[Ecosystem home](/) · [Packages](/packages/) · [Get started](/getting-started/) · [Capabilities](/capabilities/) — Built with [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).",
        edit_link = "main",
    ),
    repo = Documenter.Remotes.GitHub("Statistical-network-analysis-with-Julia", "Relevent.jl"),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Interaction History" => "guide/history.md",
            "Advanced Statistics" => "guide/statistics.md",
            "R Effect Coverage" => "guide/r_catalogue.md",
            "Timing Models" => "guide/timing.md",
        ],
        "API Reference" => [
            "Types" => "api/types.md",
            "Statistics" => "api/statistics.md",
            "Estimation" => "api/estimation.md",
        ],
    ],
    # STRICT. Undefined bindings, bad cross-references, duplicate docs and
    # malformed markdown are build ERRORS, so they cannot silently accumulate
    # again (a docs build that passes while warning is one that will rot).
    #
    # `checkdocs = :exports` is the one deliberate exclusion: every *exported*
    # name must be documented, but internal machinery (materialized/private
    # types, `Base`/`Graphs` method extensions, inner constructors) need not be
    # -- filler docstrings for names a user never types are worse than none.
    warnonly = false,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/statistical-network-analysis-with-Julia/Relevent.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev", # Development alias; change to "v^" when adopting release-based stable docs.
        "dev" => "dev",
    ],
    push_preview = false, # Pull requests build docs; main/tags publish through Pages.
)
