"""
    restructure_to(tt, target_groups;
                   edges=nothing,
                   split_rtol=0.0, split_cutoff=0.0, split_maxdim=0,
                   split_form=:unitary, split_final_sweep=false,
                   swap_rtol=0.0, swap_maxdim=0,
                   final_rtol=0.0, final_cutoff=0.0, final_maxdim=0,
                   final_form=:unitary) -> TensorTrain

Restructure `tt` to match the topology described by `target_groups`, combining
optional split, swap, and final-truncation phases as needed. Each entry of
`target_groups` is the site indices of one target node, in target node order;
`edges` lists target tree edges as 1-based pairs (default: chain).

The backend currently supports fuse-only, split-only, swap-only, swap-then-fuse,
and split-then-fuse plans. Mixed cases requiring split into multiple cross-node
fragments together with a subsequent swap may still be reported as unsupported.

# Phase keyword arguments

- `split_rtol`, `split_cutoff`, `split_maxdim`, `split_form`,
  `split_final_sweep`: forwarded to the split phase. See [`split_to`](@ref)
  for the meaning of each.
- `swap_rtol`, `swap_maxdim`: forwarded to the swap phase. See
  [`swap_site_indices`](@ref).
- `final_rtol`, `final_cutoff`, `final_maxdim`, `final_form`: optional final
  truncation pass on the assembled target topology. The pass runs only when
  `final_rtol`, `final_cutoff`, or `final_maxdim` is nonzero; otherwise the
  cleanup truncation is skipped.

Throws `ArgumentError` if `tt` is empty, if `target_groups` does not cover
the current site indices exactly once, or if any keyword is negative.
"""
function restructure_to(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}};
    edges::Union{Nothing, AbstractVector{<:Tuple{<:Integer, <:Integer}}}=nothing,
    split_rtol::Real=0.0,
    split_cutoff::Real=0.0,
    split_maxdim::Integer=0,
    split_form::Symbol=:unitary,
    split_final_sweep::Bool=false,
    swap_rtol::Real=0.0,
    swap_maxdim::Integer=0,
    final_rtol::Real=0.0,
    final_cutoff::Real=0.0,
    final_maxdim::Integer=0,
    final_form::Symbol=:unitary,
)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for restructure_to"))
    _validate_truncation_kwargs(split_rtol, split_cutoff, split_maxdim)
    swap_rtol >= 0 || throw(ArgumentError("swap_rtol must be nonnegative, got $swap_rtol"))
    swap_maxdim >= 0 || throw(ArgumentError("swap_maxdim must be nonnegative, got $swap_maxdim"))
    _validate_truncation_kwargs(final_rtol, final_cutoff, final_maxdim)

    args = _build_target_args(tt, target_groups, edges)
    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_restructure_to),
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
                Cdouble,
                Cdouble,
                Csize_t,
                Cint,
                Cint,
                Csize_t,
                Cdouble,
                Cdouble,
                Cdouble,
                Csize_t,
                Cint,
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
            float(split_rtol),
            float(split_cutoff),
            Csize_t(split_maxdim),
            _canonical_form_code(split_form),
            Cint(split_final_sweep ? 1 : 0),
            Csize_t(swap_maxdim),
            float(swap_rtol),
            float(final_rtol),
            float(final_cutoff),
            Csize_t(final_maxdim),
            _canonical_form_code(final_form),
            out,
        )
        _check_backend_status(status, "restructuring TensorTrain to target topology")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(tt_handle)
        _release_target_args(args)
    end
end
