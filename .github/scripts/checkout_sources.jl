# Reconstruct exactly the local source closure needed for this package's
# tests/docs/benchmarks. With --check, inspect existing checkouts without cloning.
using TOML

root = realpath(joinpath(@__DIR__, "..", ".."))
workspace = dirname(root)
check_only = "--check" in ARGS
queue = [(joinpath(root, "Project.toml"), true),
         (joinpath(root, "docs", "Project.toml"), false),
         (joinpath(root, "benchmark", "Project.toml"), false)]
seen = Set{String}()
repos = Set([root])
while !isempty(queue)
    project, include_tests = popfirst!(queue)
    project in seen && continue
    push!(seen, project)
    metadata = TOML.parsefile(project)
    needed = Set(keys(get(metadata, "deps", Dict())))
    if include_tests
        union!(needed, get(get(metadata, "targets", Dict()), "test", String[]))
    end
    for (name, source) in get(metadata, "sources", Dict())
        name in needed || continue
        haskey(source, "path") || continue
        target = dirname(normpath(joinpath(dirname(project), source["path"], "Project.toml")))
        target == root && continue
        dirname(target) == workspace && basename(target) == "$name.jl" ||
            error("Unexpected source layout for $name in $project: $target")
        if !isdir(target)
            check_only && error("Missing source checkout: $target")
            url = "https://github.com/statistical-network-analysis-with-Julia/$name.jl"
            run(`git clone --depth 1 --quiet $url $target`)
        end
        dependency_project = joinpath(target, "Project.toml")
        dependency = TOML.parsefile(dependency_project)
        dependency["name"] == name || error("Wrong package in $target")
        expected = get(get(metadata,"deps",Dict()), name,
                       get(get(metadata,"extras",Dict()),name,nothing))
        dependency["uuid"] == expected || error("UUID mismatch for $name")
        push!(repos, target)
        push!(queue, (dependency_project, false))
    end
end
for repo in sort!(collect(repos))
    println(basename(repo), '\t', strip(read(`git -C $repo rev-parse HEAD`, String)))
end
