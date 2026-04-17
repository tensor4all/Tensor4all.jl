"""
    split_to(tt, target_groups;
             edges=nothing, rtol=0.0, cutoff=0.0, maxdim=0,
             form=:unitary, final_sweep=false) -> TensorTrain

Split nodes of `tt` so that the result has the finer topology described by
`target_groups`. Each entry of `target_groups` is the site indices of one
target node, in target node order. `edges` lists target tree edges as
1-based pairs (default: chain).

The per-tensor split itself uses an exact QR factorization. The truncation
controls below take effect only when `final_sweep=true`, in which case a
post-split truncation sweep runs on the assembled target topology.

# Keyword arguments

- `rtol`: relative tolerance for the SVD/LU truncation sweep. `0.0` disables.
- `cutoff`: absolute cutoff fed to the same backend resolver as `rtol`.
- `maxdim`: maximum bond dimension. `0` (default) means no rank cap.
- `form`: factorization form. One of `:unitary` (SVD-based, default) or `:lu`.
- `final_sweep`: when `true`, run a global truncation sweep after the
  per-edge splits. Defaults to `false` and skips truncation entirely.

Throws `ArgumentError` if `tt` is empty, if `target_groups` does not cover
the current site indices exactly once, or if any truncation control is
negative.
"""
function split_to(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}};
    edges::Union{Nothing, AbstractVector{<:Tuple{<:Integer, <:Integer}}}=nothing,
    rtol::Real=0.0,
    cutoff::Real=0.0,
    maxdim::Integer=0,
    form::Symbol=:unitary,
    final_sweep::Bool=false,
)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for split_to"))
    _validate_truncation_kwargs(rtol, cutoff, maxdim)

    args = _build_target_args(tt, target_groups, edges)
    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_split_to),
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
            float(rtol),
            float(cutoff),
            Csize_t(maxdim),
            _canonical_form_code(form),
            Cint(final_sweep ? 1 : 0),
            out,
        )
        _check_backend_status(status, "splitting TensorTrain to target topology")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(tt_handle)
        _release_target_args(args)
    end
end
