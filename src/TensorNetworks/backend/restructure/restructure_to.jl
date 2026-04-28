"""
    restructure_to(tt, target_groups;
                   edges=nothing,
                   split_threshold=0.0, split_maxdim=0,
                   split_svd_policy=nothing, split_final_sweep=false,
                   swap_rtol=0.0, swap_maxdim=0,
                   final_threshold=0.0, final_maxdim=0,
                   final_svd_policy=nothing) -> TensorTrain

Restructure `tt` so that its site index grouping matches `target_groups`.
Each entry of `target_groups` is the site indices of one target node, in
target node order. `edges` lists target tree edges as 1-based pairs
(default: chain `(1, 2), (2, 3), ..., (n - 1, n)`).

Implemented as a pure Julia composition that dispatches to one of the
primitive operations:

| Case | Dispatch |
|---|---|
| Same site grouping and same node order as the current `tt` | (identity, optional final truncation) |
| Same grouping but different node ownership of indices | [`swap_site_indices`](@ref) |
| Each current node's sites are a subset of one target node | [`fuse_to`](@ref) |
| Each target node's sites are a subset of one current node | [`split_to`](@ref) |
| Mixed split/swap/fuse regrouping | `split_to` singletons, `swap_site_indices`, then `fuse_to` |

Mixed patterns are synthesized from the same primitives. Truncation controls
apply to the split phase and the optional final truncation pass; the
intermediate swap phase uses `swap_rtol` / `swap_maxdim`.

# Phase keyword arguments

- `split_threshold`, `split_maxdim`, `split_svd_policy`, `split_final_sweep`
  are forwarded to [`split_to`](@ref) when the split phase runs.
- `swap_rtol`, `swap_maxdim` are forwarded to [`swap_site_indices`](@ref)
  when the swap phase runs.
- `final_threshold`, `final_maxdim`, `final_svd_policy` describe an
  optional final [`truncate`](@ref) pass on the assembled target topology.
  The pass runs only when any of the three is nonzero / non-nothing.

Throws `ArgumentError` if `tt` is empty, if `target_groups` does not cover
the current site indices exactly once, or if any keyword is negative.
"""
function restructure_to(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}};
    edges::Union{Nothing, AbstractVector{<:Tuple{<:Integer, <:Integer}}}=nothing,
    split_threshold::Real=0.0,
    split_maxdim::Integer=0,
    split_svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
    split_final_sweep::Bool=false,
    swap_rtol::Real=0.0,
    swap_maxdim::Integer=0,
    final_threshold::Real=0.0,
    final_maxdim::Integer=0,
    final_svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for restructure_to"))
    _validate_truncation_kwargs(split_threshold, split_maxdim)
    swap_rtol >= 0 || throw(ArgumentError("swap_rtol must be nonnegative, got $swap_rtol"))
    swap_maxdim >= 0 || throw(ArgumentError("swap_maxdim must be nonnegative, got $swap_maxdim"))
    _validate_truncation_kwargs(final_threshold, final_maxdim)
    _validate_target_groups_coverage(tt, target_groups)

    current_groups = _siteinds_by_tensor(tt)
    target_sets = [Set(group) for group in target_groups]
    current_sets = [Set(group) for group in current_groups]

    if _groups_equal_as_ordered_sets(current_sets, target_sets)
        return _final_truncate(tt; final_threshold, final_maxdim, final_svd_policy)
    end

    if _is_swap_only(current_sets, target_sets)
        assignment = _build_swap_assignment(current_groups, target_groups)
        result = swap_site_indices(tt, assignment; rtol=swap_rtol, maxdim=swap_maxdim)
        return _final_truncate(result; final_threshold, final_maxdim, final_svd_policy)
    end

    if _is_fuse_only(current_sets, target_sets)
        result = fuse_to(tt, target_groups; edges)
        return _final_truncate(result; final_threshold, final_maxdim, final_svd_policy)
    end

    if _is_split_only(current_sets, target_sets)
        result = split_to(
            tt,
            target_groups;
            edges,
            threshold=split_threshold,
            maxdim=split_maxdim,
            svd_policy=split_svd_policy,
            final_sweep=split_final_sweep,
        )
        return _final_truncate(result; final_threshold, final_maxdim, final_svd_policy)
    end

    result = _mixed_restructure_to(
        tt,
        current_groups,
        target_groups;
        edges,
        split_threshold,
        split_maxdim,
        split_svd_policy,
        split_final_sweep,
        swap_rtol,
        swap_maxdim,
    )
    return _final_truncate(result; final_threshold, final_maxdim, final_svd_policy)
end

function _groups_equal_as_ordered_sets(a::Vector{Set{Index}}, b::Vector{Set{Index}})
    return length(a) == length(b) && all(a[i] == b[i] for i in eachindex(a))
end

# Site assignments share the same partition into groups, but at least one
# index sits at a different node in the target than in the current grouping.
function _is_swap_only(current_sets::Vector{Set{Index}}, target_sets::Vector{Set{Index}})
    length(current_sets) == length(target_sets) || return false
    # Target's set of partitions must equal current's set of partitions
    # (regardless of node order / assignment).
    return Set(current_sets) == Set(target_sets)
end

function _build_swap_assignment(
    current_groups::Vector{<:AbstractVector{<:Index}},
    target_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    # Map every site index to its target 1-based node position.
    assignment = Dict{Index, Int}()
    current_position = Dict{Index, Int}()
    for (pos, group) in enumerate(current_groups), index in group
        current_position[index] = pos
    end
    for (pos, group) in enumerate(target_groups), index in group
        get(current_position, index, -1) == pos && continue
        assignment[index] = pos
    end
    return assignment
end

# Every current node's site indices must fit entirely into a single target node.
# When that holds, fuse_to alone reaches the target topology.
function _is_fuse_only(current_sets::Vector{Set{Index}}, target_sets::Vector{Set{Index}})
    for current_set in current_sets
        any(target_set -> issubset(current_set, target_set), target_sets) || return false
    end
    return true
end

# Every target node's site indices must fit entirely into a single current node.
# When that holds, split_to alone reaches the target topology.
function _is_split_only(current_sets::Vector{Set{Index}}, target_sets::Vector{Set{Index}})
    for target_set in target_sets
        any(current_set -> issubset(target_set, current_set), current_sets) || return false
    end
    return true
end

function _singleton_groups(groups::AbstractVector{<:AbstractVector{<:Index}})
    return [[index] for group in groups for index in group]
end

function _mixed_restructure_to(
    tt::TensorTrain,
    current_groups::Vector{<:AbstractVector{<:Index}},
    target_groups::AbstractVector{<:AbstractVector{<:Index}};
    edges::Union{Nothing, AbstractVector{<:Tuple{<:Integer, <:Integer}}},
    split_threshold::Real,
    split_maxdim::Integer,
    split_svd_policy::Union{Nothing, SvdTruncationPolicy},
    split_final_sweep::Bool,
    swap_rtol::Real,
    swap_maxdim::Integer,
)
    current_singletons = _singleton_groups(current_groups)
    target_singletons = _singleton_groups(target_groups)

    result = split_to(
        tt,
        current_singletons;
        threshold=split_threshold,
        maxdim=split_maxdim,
        svd_policy=split_svd_policy,
        final_sweep=split_final_sweep,
    )

    result_singletons = _siteinds_by_tensor(result)
    result_sets = [Set(group) for group in result_singletons]
    target_singleton_sets = [Set(group) for group in target_singletons]

    if !_groups_equal_as_ordered_sets(result_sets, target_singleton_sets)
        assignment = _build_swap_assignment(result_singletons, target_singletons)
        result = swap_site_indices(result, assignment; rtol=swap_rtol, maxdim=swap_maxdim)
    end

    return fuse_to(result, target_groups; edges)
end

function _final_truncate(
    tt::TensorTrain;
    final_threshold::Real,
    final_maxdim::Integer,
    final_svd_policy::Union{Nothing, SvdTruncationPolicy},
)
    has_final_truncation = final_threshold > 0 || final_maxdim > 0 ||
        final_svd_policy !== nothing
    has_final_truncation || return tt
    return truncate(
        tt;
        threshold=final_threshold,
        maxdim=final_maxdim,
        svd_policy=final_svd_policy,
    )
end

function _validate_target_groups_coverage(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    isempty(target_groups) && throw(ArgumentError("target_groups must not be empty"))
    target_set = Set{Index}()
    for group in target_groups
        isempty(group) && throw(ArgumentError("target_groups entries must not be empty"))
        for index in group
            index in target_set && throw(ArgumentError(
                "target_groups: site index $index appears more than once",
            ))
            push!(target_set, index)
        end
    end
    current_set = Set(index for group in _siteinds_by_tensor(tt) for index in group)
    target_set == current_set || throw(ArgumentError(
        "target_groups must cover exactly the current site indices of tt; mismatch detected.",
    ))
    return nothing
end
