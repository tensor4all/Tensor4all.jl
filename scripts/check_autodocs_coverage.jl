#!/usr/bin/env julia
# Lint: every src/**/*.jl file that contains a public, documented symbol
# (a docstring immediately preceding a non-underscored `function` / `struct` /
# `mutable struct` / `const` definition) must be listed in some `@autodocs`
# `Pages = [...]` block under docs/src/. Documenter does not warn when a file
# is omitted from a Pages list, so docstrings can silently disappear from the
# rendered site (see https://github.com/tensor4all/Tensor4all.jl/issues/...).

using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const SRC_ROOT = joinpath(REPO_ROOT, "src")
const DOCS_SRC = joinpath(REPO_ROOT, "docs", "src")

"Recursively collect every .jl file under `dir`, returned as repo-relative paths with forward slashes."
function collect_jl_files(dir::AbstractString)
    out = String[]
    for (root, _, files) in walkdir(dir)
        for file in files
            endswith(file, ".jl") || continue
            abspath = joinpath(root, file)
            relpath_unix = replace(relpath(abspath, REPO_ROOT), Base.Filesystem.path_separator => "/")
            push!(out, relpath_unix)
        end
    end
    return sort(out)
end

"Extract the union of `Pages = [...]` entries across every `@autodocs` block in any docs/src/**/*.md."
function collect_autodocs_pages(dir::AbstractString)
    pages = Set{String}()
    isdir(dir) || return pages
    for (root, _, files) in walkdir(dir)
        for file in files
            endswith(file, ".md") || continue
            text = read(joinpath(root, file), String)
            for m in eachmatch(r"@autodocs\b[^`]*?Pages\s*=\s*\[([^\]]*)\]"s, text)
                inside = m.captures[1]
                for entry in eachmatch(r"\"([^\"]+)\"", inside)
                    push!(pages, entry.captures[1])
                end
            end
        end
    end
    return pages
end

"Return true if the source file appears to define at least one publicly-documented
symbol — a triple-quoted docstring immediately followed (after blank lines) by a
top-level definition whose name does not start with an underscore."
function defines_documented_public_symbol(path::AbstractString)
    text = read(path, String)
    pat = r"\"\"\"[\s\S]*?\"\"\"\s*\n(?:#[^\n]*\n|\s*\n)*\s*(?:mutable\s+struct|struct|abstract\s+type|primitive\s+type|function|const|macro)\s+([A-Za-z][A-Za-z0-9_!]*)"
    for m in eachmatch(pat, text)
        name = m.captures[1]
        startswith(name, "_") || return true
    end
    return false
end

function main()
    documented_pages = collect_autodocs_pages(DOCS_SRC)
    src_files = collect_jl_files(SRC_ROOT)

    missing_files = String[]
    for relpath_unix in src_files
        # The Pages list entries are relative to src/, so strip the leading `src/`.
        startswith(relpath_unix, "src/") || continue
        page_key = replace(relpath_unix, r"^src/" => "")
        page_key in documented_pages && continue
        defines_documented_public_symbol(joinpath(REPO_ROOT, relpath_unix)) || continue
        push!(missing_files, page_key)
    end

    if isempty(missing_files)
        println("OK: every src/ file with a documented public symbol is reachable from some @autodocs Pages list.")
        return 0
    end

    println("ERROR: the following source files have public docstrings that no @autodocs Pages list reaches.")
    println("Add each path to the appropriate @autodocs block in docs/src/api.md (or another docs/src/*.md).")
    println()
    for path in missing_files
        @printf("  - %s\n", path)
    end
    println()
    println("If a file is intentionally internal, give its public symbols leading underscores or omit their docstrings.")
    return 1
end

exit(main())
