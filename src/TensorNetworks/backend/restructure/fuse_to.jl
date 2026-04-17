"""
    fuse_to(tt, target_groups; edges=nothing) -> TensorTrain

Fuse adjacent nodes of `tt` so that the result has the topology described by
`target_groups`. Each entry of `target_groups` is the site indices of one
target node, in target node order; `edges` lists the target tree edges as
`(source, target)` 1-based pairs and defaults to the chain
`(1, 2), (2, 3), ..., (n-1, n)`.

The transformation is exact (no truncation): adjacent current nodes whose
combined site indices match a target node are contracted together. Returns a
new `TensorTrain`.

Throws `ArgumentError` if `tt` is empty or if `target_groups` does not cover
the current site indices exactly once.
"""
function fuse_to(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}};
    edges::Union{Nothing, AbstractVector{<:Tuple{<:Integer, <:Integer}}}=nothing,
)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for fuse_to"))

    args = _build_target_args(tt, target_groups, edges)
    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_fuse_to),
            Cint,
            (
                Ptr{Cvoid},
                Ptr{Csize_t},
                Csize_t,
                Ptr{Ptr{Cvoid}},
                Ptr{Csize_t},
                Ptr{Csize_t},
                Ptr{Csize_t},
                Csize_t,
                Ref{Ptr{Cvoid}},
            ),
            tt_handle,
            args.vertices,
            args.n_vertices,
            args.siteind_handles,
            args.siteind_lens,
            args.edge_sources,
            args.edge_targets,
            args.n_edges,
            out,
        )
        _check_backend_status(status, "fusing TensorTrain to target topology")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(tt_handle)
        _release_target_args(args)
    end
end
