_query_indices(index::Index) = Index[index]
_query_indices(indices::AbstractVector{<:Index}) = collect(indices)

_index_identity_key(index::Index) = index
_same_index_identity(a::Index, b::Index) = a == b

function _index_counts(tt::TensorTrain)
    counts = Dict{Index, Int}()
    for tensor in tt
        for index in inds(tensor)
            key = _index_identity_key(index)
            counts[key] = get(counts, key, 0) + 1
        end
    end
    return counts
end

function _siteinds_by_tensor(tt::TensorTrain)
    counts = _index_counts(tt)
    siteinds = [Index[] for _ in tt.data]
    for (position, tensor) in pairs(tt.data)
        for index in inds(tensor)
            counts[_index_identity_key(index)] == 1 || continue
            hastag(index, _LINK_TAG) && continue
            push!(siteinds[position], index)
        end
    end
    return siteinds
end

function _siteinds_with_positions(tt::TensorTrain)
    siteinds = Tuple{Int, Index}[]
    for (position, indices) in pairs(_siteinds_by_tensor(tt))
        for index in indices
            push!(siteinds, (position, index))
        end
    end
    return siteinds
end

_siteind_set(tt::TensorTrain) = Set(last.(_siteinds_with_positions(tt)))
_siteind_identity_set(tt::TensorTrain) = Set(_index_identity_key(index) for (_, index) in _siteinds_with_positions(tt))

function _validate_scan_tag(tag::AbstractString)
    occursin('=', tag) && throw(
        ArgumentError("Invalid tag \"$tag\": expected a bare tag without `=`"),
    )
    return String(tag)
end

function _numbered_site_matches(tt::TensorTrain, tag::AbstractString)
    bare_tag = _validate_scan_tag(tag)
    prefix = bare_tag * "="
    numbered = Dict{Int, Tuple{Int, Index}}()

    for (position, index) in _siteinds_with_positions(tt)
        plev(index) == 0 || continue
        for index_tag in tags(index)
            startswith(index_tag, prefix) || continue
            suffix = index_tag[length(prefix)+1:end]
            number = tryparse(Int, suffix)
            number === nothing && continue
            number > 0 || continue
            haskey(numbered, number) && throw(
                ArgumentError("Duplicate numbered tag \"$index_tag\" found in TensorTrain"),
            )
            numbered[number] = (position, index)
        end
    end

    return numbered
end

function _tensor_positions_with_indices(tt::TensorTrain, query_indices::AbstractVector{<:Index})
    positions = Int[]
    query_keys = Set(_index_identity_key.(query_indices))
    for (position, tensor) in pairs(tt.data)
        tensor_indices = inds(tensor)
        any(index -> _index_identity_key(index) in query_keys, tensor_indices) || continue
        push!(positions, position)
    end
    return positions
end

function _replacement_mapping(oldsites::AbstractVector{<:Index}, newsites::AbstractVector{<:Index})
    length(oldsites) == length(newsites) || throw(
        DimensionMismatch(
            "Length mismatch: got $(length(oldsites)) old site indices and $(length(newsites)) new site indices",
        ),
    )
    old_keys = _index_identity_key.(oldsites)
    length(Set(old_keys)) == length(oldsites) || throw(
        ArgumentError("duplicate old site indices are not allowed in replace_siteinds"),
    )
    for (old, new) in zip(oldsites, newsites)
        dim(old) == dim(new) || throw(ArgumentError(
            "Cannot replace index $old (dim=$(dim(old))) with $new (dim=$(dim(new))); dimensions must match",
        ))
    end
    return Dict(old_keys .=> newsites)
end

function _ensure_replacement_targets_exist(tt::TensorTrain, oldsites::AbstractVector{<:Index})
    present = Set{Index}()
    siteinds = _siteind_identity_set(tt)
    for tensor in tt
        union!(present, _index_identity_key.(inds(tensor)))
    end
    for index in oldsites
        key = _index_identity_key(index)
        key in present || throw(ArgumentError("Not found: index $index does not occur in TensorTrain"))
        key in siteinds || throw(
            ArgumentError("Expected a site-like index in replace_siteinds, got non-site index $index"),
        )
    end
    return nothing
end

function _replacement_for_index(index::Index, replacements::Dict{Index, Index})
    key = _index_identity_key(index)
    haskey(replacements, key) || return index
    replacement = replacements[key]
    dim(index) == dim(replacement) || throw(ArgumentError(
        "Cannot replace index $index (dim=$(dim(index))) with $replacement (dim=$(dim(replacement))); dimensions must match",
    ))
    return replacement
end

function _replace_tensor_indices(tensor::Tensor, replacements::Dict{Index, Index})
    current_indices = inds(tensor)
    changed = false
    new_indices = map(current_indices) do index
        replacement = _replacement_for_index(index, replacements)
        if replacement != index
            changed = true
        end
        return replacement
    end
    return changed ? Tensor(copy_data(tensor), new_indices; structured_storage=_structured_storage_from_tensor(tensor)) : tensor
end

function _replace_tensor_indices!(tensor::Tensor, replacements::Dict{Index, Index})
    old = Index[]
    new = Index[]
    for index in inds(tensor)
        replacement = _replacement_for_index(index, replacements)
        replacement == index && continue
        push!(old, index)
        push!(new, replacement)
    end
    isempty(old) && return tensor
    return replaceinds!(tensor, old, new)
end

function _replace_tensor_indices_keep_data(tensor::Tensor, replacements::Dict{Index, Index})
    current_indices = inds(tensor)
    changed = false
    new_indices = map(current_indices) do index
        replacement = _replacement_for_index(index, replacements)
        if replacement != index
            changed = true
        end
        return replacement
    end
    if !changed
        return tensor
    end
    return Tensor(copy_data(tensor), new_indices; structured_storage=_structured_storage_from_tensor(tensor))
end

_copy_tensor(tensor::Tensor) = Tensor(copy_data(tensor), inds(tensor); structured_storage=_structured_storage_from_tensor(tensor))
_copy_train(tt::TensorTrain) = TensorTrain([_copy_tensor(tensor) for tensor in tt.data], tt.llim, tt.rlim)

"""
    findsite(tt, index_or_indices)

Return the first tensor position in `tt` containing any queried index, or
`nothing` if no queried index occurs.
"""
function findsite(tt::TensorTrain, index_or_indices::Union{Index, AbstractVector{<:Index}})
    positions = _tensor_positions_with_indices(tt, _query_indices(index_or_indices))
    return isempty(positions) ? nothing : first(positions)
end

"""
    findsites(tt, index_or_indices)

Return all tensor positions in `tt` containing any queried index, in tensor
order.
"""
function findsites(tt::TensorTrain, index_or_indices::Union{Index, AbstractVector{<:Index}})
    return _tensor_positions_with_indices(tt, _query_indices(index_or_indices))
end

"""
    findallsiteinds_by_tag(tt; tag)

Return site-like indices tagged as `tag=1`, `tag=2`, ... in numbered order,
stopping at the first missing number.
"""
function findallsiteinds_by_tag(tt::TensorTrain; tag::AbstractString)
    numbered = _numbered_site_matches(tt, tag)
    matches = Index[]
    number = 1
    while haskey(numbered, number)
        push!(matches, numbered[number][2])
        number += 1
    end
    return matches
end

"""
    findallsites_by_tag(tt; tag)

Return tensor positions for numbered site tags `tag=1`, `tag=2`, ... in scan
order, stopping at the first missing number.
"""
function findallsites_by_tag(tt::TensorTrain; tag::AbstractString)
    numbered = _numbered_site_matches(tt, tag)
    matches = Int[]
    number = 1
    while haskey(numbered, number)
        push!(matches, numbered[number][1])
        number += 1
    end
    return matches
end

"""
    replace_siteinds!(tt, oldsites, newsites)

Replace each site-like index in `oldsites` by the corresponding index in
`newsites`, mutating `tt` in place.
"""
function replace_siteinds!(
    tt::TensorTrain,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    replacements = _replacement_mapping(oldsites, newsites)
    _ensure_replacement_targets_exist(tt, oldsites)
    for index in eachindex(tt.data)
        _replace_tensor_indices!(tt.data[index], replacements)
    end
    return tt
end

"""
    replace_siteinds(tt, oldsites, newsites)

Return a non-aliasing copy of `tt` with each site-like index in `oldsites`
replaced by the corresponding index in `newsites`.
"""
function replace_siteinds(
    tt::TensorTrain,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    copied = _copy_train(tt)
    return replace_siteinds!(copied, oldsites, newsites)
end

"""
    replace_siteinds_shared(tt, oldsites, newsites)

Return a new `TensorTrain` with each site-like index in `oldsites` replaced by
the corresponding index in `newsites`. Unlike [`replace_siteinds`](@ref), tensor
data arrays are shared (not copied). Compatible with the ITensorMPS convention
where `prime(mps)` creates new index metadata but shares tensor storage.

See also [`replace_siteinds`](@ref), [`replace_siteinds!`](@ref).
"""
function replace_siteinds_shared(
    tt::TensorTrain,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    replacements = _replacement_mapping(oldsites, newsites)
    _ensure_replacement_targets_exist(tt, oldsites)
    tensors = [_replace_tensor_indices_keep_data(t, replacements) for t in tt.data]
    return TensorTrain(tensors, tt.llim, tt.rlim)
end

"""
    replace_siteinds_part!(tt, oldsites, newsites)

Replace only the listed site-like indices in `tt`, mutating the affected
tensor slots in place.
"""
function replace_siteinds_part!(
    tt::TensorTrain,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    return replace_siteinds!(tt, oldsites, newsites)
end
