struct OneHot{T}
    index::Index
    value::Int
end

"""
    onehot(index => value; T=Float64)

Return a lightweight one-hot selector for `index` at 1-based `value`.
"""
function onehot(replacement::Pair{Index,<:Integer}; T=Float64)
    index = first(replacement)
    value = Int(last(replacement))
    1 <= value <= dim(index) || throw(ArgumentError(
        "onehot value $value is out of range for index dimension $(dim(index))",
    ))
    return OneHot{T}(index, value)
end

inds(selector::OneHot) = Index[selector.index]

function Tensor(selector::OneHot{T}) where {T}
    data = zeros(T, dim(selector.index))
    data[selector.value] = one(T)
    return Tensor(data, [selector.index])
end

function _index_position(tensor_indices::Vector{Index}, index::Index, context::AbstractString)
    position = findfirst(==(index), tensor_indices)
    position === nothing && throw(ArgumentError(
        "$context index $index not found in tensor indices $tensor_indices",
    ))
    return position
end

function _validate_unique_indices(indices::AbstractVector{Index}, context::AbstractString)
    length(unique(indices)) == length(indices) || throw(
        ArgumentError("$context indices must not contain duplicates, got $indices"),
    )
    return nothing
end

function _tensor_from_indexed_data(data, result_inds::Vector{Index})
    data isa AbstractArray && return Tensor(copy(data), result_inds)
    return Tensor(data)
end

"""
    fixinds(t, replacements...)

Fix tensor indices to 1-based integer values and remove those indices from the
result.
"""
function fixinds(t::Tensor, replacements::Pair{Index,<:Integer}...)
    isempty(replacements) && return Tensor(t.data, inds(t); backend_handle=t.backend_handle)

    tensor_indices = inds(t)
    fixed = falses(length(tensor_indices))
    selectors = Any[Colon() for _ in 1:rank(t)]

    for replacement in replacements
        index = first(replacement)
        position = _index_position(tensor_indices, index, "fixinds")
        fixed[position] && throw(ArgumentError("fixinds index $index appears more than once"))
        value = Int(last(replacement))
        1 <= value <= dim(index) || throw(ArgumentError(
            "fixinds value $value is out of range for index dimension $(dim(index))",
        ))
        selectors[position] = value
        fixed[position] = true
    end

    result_inds = Index[tensor_indices[n] for n in eachindex(tensor_indices) if !fixed[n]]
    return _tensor_from_indexed_data(t.data[selectors...], result_inds)
end

"""
    suminds(t, indices...)

Sum over `indices` and remove them from the result.
"""
function suminds(t::Tensor, indices::Index...)
    isempty(indices) && return Tensor(t.data, inds(t); backend_handle=t.backend_handle)

    tensor_indices = inds(t)
    requested = collect(indices)
    _validate_unique_indices(requested, "suminds")
    axes = [_index_position(tensor_indices, index, "suminds") for index in requested]
    result_inds = Index[tensor_indices[n] for n in eachindex(tensor_indices) if !(n in axes)]

    summed = sum(t.data; dims=Tuple(sort(axes)))
    dropped = dropdims(summed; dims=Tuple(sort(axes)))
    return _tensor_from_indexed_data(dropped, result_inds)
end

"""
    projectinds(t, replacements...)

Project tensor indices to explicit 1-based value subsets. Projected indices
remain present with fresh dimensions equal to the subset lengths.
"""
function projectinds(
    t::Tensor,
    replacements::Pair{Index,<:AbstractVector{<:Integer}}...,
)
    isempty(replacements) && return Tensor(t.data, inds(t); backend_handle=t.backend_handle)

    tensor_indices = inds(t)
    selectors = Any[Colon() for _ in 1:rank(t)]
    result_inds = inds(t)
    seen = falses(length(tensor_indices))

    for replacement in replacements
        index = first(replacement)
        position = _index_position(tensor_indices, index, "projectinds")
        seen[position] && throw(ArgumentError("projectinds index $index appears more than once"))
        values = Int.(collect(last(replacement)))
        isempty(values) && throw(ArgumentError("projectinds values for $index must not be empty"))
        for value in values
            1 <= value <= dim(index) || throw(ArgumentError(
                "projectinds value $value is out of range for index dimension $(dim(index))",
            ))
        end

        selectors[position] = values
        result_inds[position] = Index(length(values); tags=tags(index), plev=plev(index))
        seen[position] = true
    end

    return Tensor(copy(t.data[selectors...]), result_inds)
end

contract(t::Tensor, selector::OneHot) = fixinds(t, selector.index => selector.value)
contract(selector::OneHot, t::Tensor) = fixinds(t, selector.index => selector.value)
