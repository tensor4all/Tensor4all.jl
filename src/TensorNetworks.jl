module TensorNetworks

using ..Tensor4all: BackendUnavailableError, Index, Tensor, SkeletonNotImplemented

export TensorTrain, LinearOperator
export set_input_space!, set_output_space!, set_iospaces!, apply
export findsite, findsites, findallsiteinds_by_tag, findallsites_by_tag
export replace_siteinds!, replace_siteinds, replace_siteinds_part!
export rearrange_siteinds, makesitediagonal, extractdiagonal, matchsiteinds
export save_as_mps, load_tt

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

findsite(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:findsite, :tt))
findsites(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:findsites, :tt))
findallsiteinds_by_tag(::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:findallsiteinds_by_tag, :tt))
findallsites_by_tag(::TensorTrain; kwargs...) = throw(SkeletonNotImplemented(:findallsites_by_tag, :tt))

replace_siteinds!(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(Symbol("replace_siteinds!"), :tt))
replace_siteinds(::TensorTrain, args...; kwargs...) = throw(SkeletonNotImplemented(:replace_siteinds, :tt))
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
