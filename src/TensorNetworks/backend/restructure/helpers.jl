_default_chain_edges(n::Integer) = [(i, i + 1) for i in 1:(n - 1)]

"""
    _validate_target_groups(tt, target_groups)

Validate that `target_groups` is a valid topology specification for `tt`:

- non-empty
- every site index in `target_groups` is currently a site index of `tt`
- the union of `target_groups` covers exactly the site indices of `tt`
- no site index appears twice across `target_groups`
"""
function _validate_target_groups(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    isempty(target_groups) && throw(
        ArgumentError("target_groups must contain at least one node"),
    )

    seen = Dict{Index, Int}()
    for (vertex, group) in pairs(target_groups)
        isempty(group) && throw(
            ArgumentError("target node $(vertex) must contain at least one site index"),
        )
        for site in group
            if haskey(seen, site)
                throw(
                    ArgumentError(
                        "target site index $site appears in both node $(seen[site]) and node $(vertex)",
                    ),
                )
            end
            seen[site] = vertex
        end
    end

    current_sites = _siteind_set(tt)
    for site in keys(seen)
        site in current_sites || throw(
            ArgumentError(
                "target site index $site is not a current site index of the TensorTrain",
            ),
        )
    end
    for site in current_sites
        haskey(seen, site) || throw(
            ArgumentError(
                "current site index $site is missing from target_groups; every site index must appear",
            ),
        )
    end

    return nothing
end

function _validate_target_edges(
    edges::AbstractVector{<:Tuple{<:Integer, <:Integer}},
    n_vertices::Integer,
)
    for (n, edge) in pairs(edges)
        src, dst = edge
        (1 <= src <= n_vertices) || throw(
            ArgumentError(
                "edges[$n] source $src is out of range 1:$n_vertices",
            ),
        )
        (1 <= dst <= n_vertices) || throw(
            ArgumentError(
                "edges[$n] target $dst is out of range 1:$n_vertices",
            ),
        )
        src == dst && throw(ArgumentError("edges[$n] is a self-loop ($src, $dst)"))
    end
    return nothing
end

function _flatten_target_siteinds(target_groups::AbstractVector{<:AbstractVector{<:Index}})
    flat = Index[]
    lens = Csize_t[]
    for group in target_groups
        push!(lens, Csize_t(length(group)))
        append!(flat, group)
    end
    return flat, lens
end

function _resolve_edges(
    edges::Union{Nothing, AbstractVector{<:Tuple{<:Integer, <:Integer}}},
    n_vertices::Integer,
)
    resolved = edges === nothing ? _default_chain_edges(n_vertices) : edges
    _validate_target_edges(resolved, n_vertices)
    sources = Csize_t[Csize_t(src - 1) for (src, _) in resolved]
    targets = Csize_t[Csize_t(dst - 1) for (_, dst) in resolved]
    return sources, targets
end

"""
    _build_target_args(tt, target_groups, edges)

Allocate the index handles and build the parallel arrays the four restructure
C APIs require. Returns a NamedTuple with all owned handles and arrays. The
caller MUST invoke `_release_target_args` from a `finally` block to release
the index handles.
"""
function _build_target_args(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}},
    edges::Union{Nothing, AbstractVector{<:Tuple{<:Integer, <:Integer}}},
)
    _validate_target_groups(tt, target_groups)

    n_vertices = length(target_groups)
    flat_sites, siteind_lens = _flatten_target_siteinds(target_groups)
    edge_sources, edge_targets = _resolve_edges(edges, n_vertices)

    # Vertex labels match positions 0..n-1 (matching the existing chain layout).
    vertices = Csize_t[Csize_t(i - 1) for i in 1:n_vertices]

    siteind_handles = Ptr{Cvoid}[]
    for site in flat_sites
        push!(siteind_handles, _new_index_handle(site))
    end

    return (
        vertices=vertices,
        siteind_lens=siteind_lens,
        siteind_handles=siteind_handles,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        n_vertices=Csize_t(n_vertices),
        n_edges=Csize_t(length(edge_sources)),
    )
end

function _release_target_args(args::NamedTuple)
    for handle in args.siteind_handles
        _release_index_handle(handle)
    end
    return nothing
end

function _validate_truncation_kwargs(threshold::Union{Nothing,Real}, maxdim::Union{Nothing,Integer})
    _normalize_threshold(threshold)
    _normalize_maxdim(maxdim)
    return nothing
end
