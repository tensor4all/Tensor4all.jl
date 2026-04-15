"""
    TensorNetworks

Indexed chain-level tensor network APIs built on top of backend-backed
`Index` and `Tensor` objects.
"""
module TensorNetworks

using ..Tensor4all:
    Index,
    Tensor,
    SkeletonNotImplemented,
    _dense_array,
    dim,
    id,
    plev,
    tags

export TensorTrain, LinearOperator
export set_input_space!, set_output_space!, set_iospaces!, apply
export findsite, findsites, findallsiteinds_by_tag, findallsites_by_tag
export replace_siteinds!, replace_siteinds, replace_siteinds_part!
export rearrange_siteinds, makesitediagonal, extractdiagonal, matchsiteinds
export save_as_mps, load_tt

"""
    TensorTrain

Primary indexed chain container for the restored Julia frontend.
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

"""
    LinearOperator

Generic operator wrapper over an indexed tensor-train representation plus its
input/output index metadata.
"""
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

"""
    set_input_space!(op, indices)

Bind concrete input-space indices to `op`.
"""
function set_input_space!(op::LinearOperator, indices::Vector{Index})
    length(indices) == length(op.input_indices) || throw(
        ArgumentError("expected $(length(op.input_indices)) input indices, got $(length(indices))"),
    )
    op.true_input = Union{Index, Nothing}[indices...]
    return op
end

"""
    set_output_space!(op, indices)

Bind concrete output-space indices to `op`.
"""
function set_output_space!(op::LinearOperator, indices::Vector{Index})
    length(indices) == length(op.output_indices) || throw(
        ArgumentError("expected $(length(op.output_indices)) output indices, got $(length(indices))"),
    )
    op.true_output = Union{Index, Nothing}[indices...]
    return op
end

"""
    set_input_space!(op, tt; kwargs...)

Deferred tensor-train-based input-space binding entry point.
"""
set_input_space!(op::LinearOperator, ::TensorTrain; kwargs...) = throw(
    SkeletonNotImplemented(:set_input_space!, :tt),
)

"""
    set_output_space!(op, tt; kwargs...)

Deferred tensor-train-based output-space binding entry point.
"""
set_output_space!(op::LinearOperator, ::TensorTrain; kwargs...) = throw(
    SkeletonNotImplemented(:set_output_space!, :tt),
)

"""
    set_iospaces!(op, input_indices, output_indices=input_indices)

Bind both input and output index spaces on `op`.
"""
function set_iospaces!(op::LinearOperator, input_indices::Vector{Index}, output_indices::Vector{Index}=input_indices)
    set_input_space!(op, input_indices)
    set_output_space!(op, output_indices)
    return op
end

"""
    set_iospaces!(op, tt; kwargs...)

Deferred tensor-train-based input/output-space binding entry point.
"""
set_iospaces!(op::LinearOperator, ::TensorTrain; kwargs...) = throw(
    SkeletonNotImplemented(:set_iospaces!, :tt),
)

"""
    findsite(tt, args...; kwargs...)

Return the first site in `tt` matching the requested index query.
Currently deferred for `TensorTrain`.
"""
findsite(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:findsite, :tt))

"""
    findsites(tt, args...; kwargs...)

Return all sites in `tt` matching the requested index query.
Currently deferred for `TensorTrain`.
"""
findsites(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:findsites, :tt))

"""
    findallsiteinds_by_tag(tt; kwargs...)

Return the physical site indices in `tt` for a numbered tag family.
Currently deferred for `TensorTrain`.
"""
findallsiteinds_by_tag(::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:findallsiteinds_by_tag, :tt))

"""
    findallsites_by_tag(tt; kwargs...)

Return the tensor positions in `tt` for a numbered tag family.
Currently deferred for `TensorTrain`.
"""
findallsites_by_tag(::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:findallsites_by_tag, :tt))

"""
    replace_siteinds!(tt, args...; kwargs...)

Replace site indices in-place on `tt`.
Currently deferred for `TensorTrain`.
"""
replace_siteinds!(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(Symbol("replace_siteinds!"), :tt))

"""
    replace_siteinds(tt, args...; kwargs...)

Return a copy of `tt` with selected site indices replaced.
Currently deferred for `TensorTrain`.
"""
replace_siteinds(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:replace_siteinds, :tt))

"""
    replace_siteinds_part!(tt, args...; kwargs...)

Replace a subset of site indices in-place on `tt`.
Currently deferred for `TensorTrain`.
"""
replace_siteinds_part!(::TensorTrain, args...; kwargs...) = throw(
    SkeletonNotImplemented(Symbol("replace_siteinds_part!"), :tt),
)

"""
    rearrange_siteinds(tt, args...; kwargs...)

Rebuild `tt` with a requested physical-leg layout.
Currently deferred for `TensorTrain`.
"""
rearrange_siteinds(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:rearrange_siteinds, :tt))

"""
    makesitediagonal(tt, args...; kwargs...)

Construct a site-diagonalized chain from `tt`.
Currently deferred for `TensorTrain`.
"""
makesitediagonal(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:makesitediagonal, :tt))

"""
    extractdiagonal(tt, args...; kwargs...)

Extract a site-diagonal chain from `tt`.
Currently deferred for `TensorTrain`.
"""
extractdiagonal(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:extractdiagonal, :tt))

"""
    matchsiteinds(tt, args...; kwargs...)

Rebuild `tt` so its site-index layout matches a target description.
Currently deferred for `TensorTrain`.
"""
matchsiteinds(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:matchsiteinds, :tt))

"""
    apply(op, tt; kwargs...)

Apply `op` to a tensor train `tt`.
Currently deferred while operator materialization stays metadata-only.
"""
apply(::LinearOperator, ::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:apply, :tt))

include("TensorNetworks/HDF5.jl")

end
