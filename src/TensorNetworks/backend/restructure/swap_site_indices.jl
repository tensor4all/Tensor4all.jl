"""
    swap_site_indices(tt, target_assignment; rtol=0.0, maxdim=0) -> TensorTrain

Reassign site indices of `tt` to new node positions using scheduled
swap transport. `target_assignment` maps each (relocated) `Index` to its
1-based target node position; site indices not listed stay at their current
node.

# Keyword arguments

- `rtol`: relative tolerance for SVD truncation during each swap step.
  `0.0` (default) disables truncation and keeps the transport exact.
- `maxdim`: maximum bond dimension during each swap step. `0` (default) means
  no rank cap.

Throws `ArgumentError` if `tt` is empty, if any keyword is negative, or if a
listed site index does not currently belong to `tt`.
"""
function swap_site_indices(
    tt::TensorTrain,
    target_assignment::AbstractDict{<:Index, <:Integer};
    rtol::Real=0.0,
    maxdim::Integer=0,
)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for swap_site_indices"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    maxdim >= 0 || throw(ArgumentError("maxdim must be nonnegative, got $maxdim"))

    n_vertices = length(tt)
    current_sites = _siteind_set(tt)
    for (site, target) in target_assignment
        site in current_sites || throw(
            ArgumentError(
                "target_assignment refers to $site which is not a current site index",
            ),
        )
        (1 <= target <= n_vertices) || throw(
            ArgumentError(
                "target_assignment[$site] = $target is out of range 1:$n_vertices",
            ),
        )
    end

    sites = collect(keys(target_assignment))
    targets_c = Csize_t[Csize_t(target_assignment[site] - 1) for site in sites]
    siteind_handles = Ptr{Cvoid}[]
    for site in sites
        push!(siteind_handles, _new_index_handle(site))
    end

    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_swap_site_indices),
            Cint,
            (
                Ptr{Cvoid},
                Ptr{Ptr{Cvoid}},
                Ptr{Csize_t},
                Csize_t,
                Csize_t,
                Cdouble,
                Ref{Ptr{Cvoid}},
            ),
            tt_handle,
            siteind_handles,
            targets_c,
            Csize_t(length(sites)),
            Csize_t(maxdim),
            float(rtol),
            out,
        )
        _check_backend_status(status, "swapping site indices on TensorTrain")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(tt_handle)
        for handle in siteind_handles
            _release_index_handle(handle)
        end
    end
end
