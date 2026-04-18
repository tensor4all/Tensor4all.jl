"""
    restructure_to(tt, target_groups;
                   edges=nothing,
                   split_rtol=0.0, split_cutoff=0.0, split_maxdim=0,
                   split_form=:unitary, split_final_sweep=false,
                   swap_rtol=0.0, swap_maxdim=0,
                   final_rtol=0.0, final_cutoff=0.0, final_maxdim=0,
                   final_form=:unitary) -> TensorTrain

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

Mixed patterns that need both splitting and fusing in the same call are
not yet supported and raise `ArgumentError`. Use `split_to` then `fuse_to`
manually for those cases.

# Phase keyword arguments

- `split_rtol`, `split_cutoff`, `split_maxdim`, `split_form`,
  `split_final_sweep` are forwarded to [`split_to`](@ref) when the split
  phase runs.
- `swap_rtol`, `swap_maxdim` are forwarded to [`swap_site_indices`](@ref)
  when the swap phase runs.
- `final_rtol`, `final_cutoff`, `final_maxdim`, `final_form` describe an
  optional final [`truncate`](@ref) pass on the assembled target topology.
  The pass runs only when one of `final_rtol` / `final_cutoff` /
  `final_maxdim` is nonzero.

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
    split_svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
    split_form::Symbol=:unitary,
    split_final_sweep::Bool=false,
    swap_rtol::Real=0.0,
    swap_maxdim::Integer=0,
    final_rtol::Real=0.0,
    final_cutoff::Real=0.0,
    final_maxdim::Integer=0,
    final_svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
    final_form::Symbol=:unitary,
)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for restructure_to"))
    _validate_truncation_kwargs(split_rtol, split_cutoff, split_maxdim)
    swap_rtol >= 0 || throw(ArgumentError("swap_rtol must be nonnegative, got $swap_rtol"))
    swap_maxdim >= 0 || throw(ArgumentError("swap_maxdim must be nonnegative, got $swap_maxdim"))
    _validate_truncation_kwargs(final_rtol, final_cutoff, final_maxdim)
    _validate_target_groups_coverage(tt, target_groups)

    current_groups = _siteinds_by_tensor(tt)
    # Compare by Index id rather than full Index equality: site indices that
    # round-trip through the C API can come back with the same id but a
    # different tag ordering, which would defeat Set{Index} comparison even
    # though they refer to the same physical degree of freedom.
    target_id_sets = [Set(id(index) for index in group) for group in target_groups]
    current_id_sets = [Set(id(index) for index in group) for group in current_groups]

    if _groups_equal_as_ordered_sets(current_id_sets, target_id_sets)
        return _final_truncate(tt; final_rtol, final_cutoff, final_maxdim, final_svd_policy, final_form)
    end

    if _is_swap_only(current_id_sets, target_id_sets)
        assignment = _build_swap_assignment(current_groups, target_groups)
        result = swap_site_indices(tt, assignment; rtol=swap_rtol, maxdim=swap_maxdim)
        return _final_truncate(result; final_rtol, final_cutoff, final_maxdim, final_svd_policy, final_form)
    end

    if _is_fuse_only(current_id_sets, target_id_sets)
        result = fuse_to(tt, target_groups; edges)
        return _final_truncate(result; final_rtol, final_cutoff, final_maxdim, final_svd_policy, final_form)
    end

    if _is_split_only(current_id_sets, target_id_sets)
        result = split_to(
            tt,
            target_groups;
            edges,
            rtol=split_rtol,
            cutoff=split_cutoff,
            maxdim=split_maxdim,
            svd_policy=split_svd_policy,
            form=split_form,
            final_sweep=split_final_sweep,
        )
        return _final_truncate(result; final_rtol, final_cutoff, final_maxdim, final_svd_policy, final_form)
    end

    throw(ArgumentError(
        "restructure_to: mixed split+fuse (or split+swap+fuse) restructuring is not yet supported. " *
        "Compose split_to / swap_site_indices / fuse_to manually for this case."
    ))
end

function _groups_equal_as_ordered_sets(a::Vector{Set{UInt64}}, b::Vector{Set{UInt64}})
    return length(a) == length(b) && all(a[i] == b[i] for i in eachindex(a))
end

# Site assignments share the same partition into groups, but at least one
# index sits at a different node in the target than in the current grouping.
function _is_swap_only(current_sets::Vector{Set{UInt64}}, target_sets::Vector{Set{UInt64}})
    length(current_sets) == length(target_sets) || return false
    # Target's set of partitions must equal current's set of partitions
    # (regardless of node order / assignment).
    return Set(current_sets) == Set(target_sets)
end

function _build_swap_assignment(
    current_groups::Vector{<:AbstractVector{<:Index}},
    target_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    # Map every site index to its target 1-based node position. Compare by
    # Index id so that C API round-trips (which may scramble tag order)
    # don't break the assignment lookup.
    assignment = Dict{Index, Int}()
    current_position_by_id = Dict{UInt64, Int}()
    for (pos, group) in enumerate(current_groups), index in group
        current_position_by_id[id(index)] = pos
    end
    for (pos, group) in enumerate(target_groups), index in group
        get(current_position_by_id, id(index), -1) == pos && continue
        assignment[index] = pos
    end
    return assignment
end

# Every current node's site indices must fit entirely into a single target node.
# When that holds, fuse_to alone reaches the target topology.
function _is_fuse_only(current_sets::Vector{Set{UInt64}}, target_sets::Vector{Set{UInt64}})
    for current_set in current_sets
        any(target_set -> issubset(current_set, target_set), target_sets) || return false
    end
    return true
end

# Every target node's site indices must fit entirely into a single current node.
# When that holds, split_to alone reaches the target topology.
function _is_split_only(current_sets::Vector{Set{UInt64}}, target_sets::Vector{Set{UInt64}})
    for target_set in target_sets
        any(current_set -> issubset(target_set, current_set), current_sets) || return false
    end
    return true
end

function _final_truncate(
    tt::TensorTrain;
    final_rtol::Real,
    final_cutoff::Real,
    final_maxdim::Integer,
    final_svd_policy::Union{Nothing, SvdTruncationPolicy},
    final_form::Symbol,
)
    has_final_truncation = final_rtol > 0 || final_cutoff > 0 || final_maxdim > 0 ||
        final_svd_policy !== nothing
    has_final_truncation || return tt
    return truncate(
        tt;
        rtol=final_rtol,
        cutoff=final_cutoff,
        maxdim=final_maxdim,
        svd_policy=final_svd_policy,
        form=final_form,
    )
end

function _validate_target_groups_coverage(
    tt::TensorTrain,
    target_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    isempty(target_groups) && throw(ArgumentError("target_groups must not be empty"))
    target_id_set = Set{UInt64}()
    for group in target_groups
        isempty(group) && throw(ArgumentError("target_groups entries must not be empty"))
        for index in group
            id(index) in target_id_set && throw(ArgumentError(
                "target_groups: site index $index appears more than once",
            ))
            push!(target_id_set, id(index))
        end
    end
    current_id_set = Set(id(index) for group in _siteinds_by_tensor(tt) for index in group)
    target_id_set == current_id_set || throw(ArgumentError(
        "target_groups must cover exactly the current site indices of tt; mismatch detected.",
    ))
    return nothing
end

