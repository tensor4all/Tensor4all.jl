module TensorNetworks

using ..Tensor4all: BackendUnavailableError, Index, Tensor, SkeletonNotImplemented, hastag, inds, tags

const _LINK_TAG = "Link"

export TensorTrain, LinearOperator
export set_input_space!, set_output_space!, set_iospaces!, apply
export findsite, findsites, findallsiteinds_by_tag, findallsites_by_tag
export replace_siteinds!, replace_siteinds, replace_siteinds_part!
export rearrange_siteinds, makesitediagonal, extractdiagonal, matchsiteinds
export save_as_mps, load_tt

"""
    TensorTrain(data, llim, rlim)
    TensorTrain(data)

Indexed tensor-train container backed by a Julia-owned `Vector{Tensor}`.
"""
mutable struct TensorTrain
    data::Vector{Tensor}
    llim::Int
    rlim::Int
end

Base.length(tt::TensorTrain) = length(tt.data)
Base.iterate(tt::TensorTrain, state...) = iterate(tt.data, state...)
Base.getindex(tt::TensorTrain, i::Int) = tt.data[i]
Base.setindex!(tt::TensorTrain, value::Tensor, i::Int) = (tt.data[i] = value)

TensorTrain(data::Vector{Tensor}) = TensorTrain(data, 0, length(data) + 1)

mutable struct LinearOperator
    mpo::Union{TensorTrain, Nothing}
    input_indices::Vector{Index}
    output_indices::Vector{Index}
    true_input::Vector{Union{Index, Nothing}}
    true_output::Vector{Union{Index, Nothing}}
    metadata::NamedTuple
end

function LinearOperator(;
    mpo::Union{TensorTrain, Nothing}=nothing,
    input_indices::Vector{Index}=Index[],
    output_indices::Vector{Index}=Index[],
    true_input::Vector{Union{Index, Nothing}}=Union{Index, Nothing}[],
    true_output::Vector{Union{Index, Nothing}}=Union{Index, Nothing}[],
    metadata::NamedTuple=(;),
)
    bound_input = isempty(true_input) ? fill(nothing, length(input_indices)) : copy(true_input)
    bound_output = isempty(true_output) ? fill(nothing, length(output_indices)) : copy(true_output)
    return LinearOperator(
        mpo,
        copy(input_indices),
        copy(output_indices),
        bound_input,
        bound_output,
        metadata,
    )
end

Base.length(op::LinearOperator) = length(op.input_indices)

function set_input_space!(op::LinearOperator, indices::Vector{Index})
    length(indices) == length(op.input_indices) || throw(
        ArgumentError("expected $(length(op.input_indices)) input indices, got $(length(indices))"),
    )
    op.true_input = Union{Index, Nothing}[indices...]
    return op
end

function set_output_space!(op::LinearOperator, indices::Vector{Index})
    length(indices) == length(op.output_indices) || throw(
        ArgumentError("expected $(length(op.output_indices)) output indices, got $(length(indices))"),
    )
    op.true_output = Union{Index, Nothing}[indices...]
    return op
end

set_input_space!(op::LinearOperator, ::TensorTrain; kwargs...) = throw(
    SkeletonNotImplemented(:set_input_space!, :tt),
)
set_output_space!(op::LinearOperator, ::TensorTrain; kwargs...) = throw(
    SkeletonNotImplemented(:set_output_space!, :tt),
)

function set_iospaces!(op::LinearOperator, input_indices::Vector{Index}, output_indices::Vector{Index}=input_indices)
    set_input_space!(op, input_indices)
    set_output_space!(op, output_indices)
    return op
end

set_iospaces!(op::LinearOperator, ::TensorTrain; kwargs...) = throw(
    SkeletonNotImplemented(:set_iospaces!, :tt),
)

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

function _siteinds_with_positions(tt::TensorTrain)
    counts = _index_counts(tt)
    siteinds = Tuple{Int, Index}[]
    for (position, tensor) in pairs(tt.data)
        for index in inds(tensor)
            counts[index] == 1 || continue
            hastag(index, _LINK_TAG) && continue
            push!(siteinds, (position, index))
        end
    end
    return siteinds
end

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
    return Dict(oldsites .=> newsites)
end

function _ensure_replacement_targets_exist(tt::TensorTrain, oldsites::AbstractVector{<:Index})
    present = Set{Index}()
    for tensor in tt
        union!(present, inds(tensor))
    end
    for index in oldsites
        index in present && continue
        throw(ArgumentError("Not found: index $index does not occur in TensorTrain"))
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

Replace each index in `oldsites` by the corresponding index in `newsites`,
mutating `tt` in place.
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

Return a copy of `tt` with each index in `oldsites` replaced by the
corresponding index in `newsites`.
"""
function replace_siteinds(
    tt::TensorTrain,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    copied = TensorTrain(copy(tt.data), tt.llim, tt.rlim)
    return replace_siteinds!(copied, oldsites, newsites)
end

replace_siteinds_part!(::TensorTrain, args...; kwargs...) = throw(
    SkeletonNotImplemented(Symbol("replace_siteinds_part!"), :tt),
)

rearrange_siteinds(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:rearrange_siteinds, :tt))
makesitediagonal(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:makesitediagonal, :tt))
extractdiagonal(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:extractdiagonal, :tt))
matchsiteinds(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:matchsiteinds, :tt))

apply(::LinearOperator, ::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:apply, :tt))

save_as_mps(args...; kwargs...) = throw(
    BackendUnavailableError("`save_as_mps` requires the HDF5 extension. Load `HDF5.jl` and retry."),
)
load_tt(args...; kwargs...) = throw(
    BackendUnavailableError("`load_tt` requires the HDF5 extension. Load `HDF5.jl` and retry."),
)

end
