_query_indices(index::Index) = Index[index]
_query_indices(indices::AbstractVector{<:Index}) = collect(indices)

function _index_counts(tt::TensorTrain)
    counts = Dict{Index, Int}()
    for tensor in tt
        for index in inds(tensor)
            counts[index] = get(counts, index, 0) + 1
        end
    end
    return counts
end

function _siteinds_by_tensor(tt::TensorTrain)
    counts = _index_counts(tt)
    siteinds = [Index[] for _ in tt.data]
    for (position, tensor) in pairs(tt.data)
        for index in inds(tensor)
            counts[index] == 1 || continue
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
    for (position, tensor) in pairs(tt.data)
        tensor_indices = inds(tensor)
        any(index -> index in tensor_indices, query_indices) || continue
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
    length(Set(oldsites)) == length(oldsites) || throw(
        ArgumentError("duplicate old site indices are not allowed in replace_siteinds"),
    )
    return Dict(oldsites .=> newsites)
end

function _ensure_replacement_targets_exist(tt::TensorTrain, oldsites::AbstractVector{<:Index})
    present = Set{Index}()
    siteinds = _siteind_set(tt)
    for tensor in tt
        union!(present, inds(tensor))
    end
    for index in oldsites
        index in present || throw(ArgumentError("Not found: index $index does not occur in TensorTrain"))
        index in siteinds || throw(
            ArgumentError("Expected a site-like index in replace_siteinds, got non-site index $index"),
        )
    end
    return nothing
end

function _replace_tensor_indices(tensor::Tensor, replacements::Dict{Index, Index})
    current_indices = inds(tensor)
    changed = false
    new_indices = map(current_indices) do index
        if haskey(replacements, index)
            changed = true
            return replacements[index]
        end
        return index
    end
    return changed ? Tensor(tensor.data, new_indices; backend_handle=tensor.backend_handle) : tensor
end

_copy_tensor(tensor::Tensor) = Tensor(tensor.data, inds(tensor); backend_handle=tensor.backend_handle)

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
        tt.data[index] = _replace_tensor_indices(tt.data[index], replacements)
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
    copied = TensorTrain([_copy_tensor(tensor) for tensor in tt.data], tt.llim, tt.llim + length(tt.data) + 1)
    return replace_siteinds!(copied, oldsites, newsites)
end
