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

"""
    invalidate_canonical!(tt)

Conservatively mark the canonical window of `tt` invalid after tensor
replacement or topology mutation.
"""
function invalidate_canonical!(tt::TensorTrain)
    tt.llim = 0
    tt.rlim = length(tt) + 1
    return tt
end

function Base.setindex!(tt::TensorTrain, value::Tensor, i::Int)
    tt.data[i] = value
    invalidate_canonical!(tt, i)
    return value
end

"""
    invalidate_canonical!(tt, i)

Mark tensor position `i` and its neighboring bonds as non-canonical while
preserving canonical bounds outside the touched block.
"""
function invalidate_canonical!(tt::TensorTrain, i::Integer)
    position = Int(i)
    1 <= position <= length(tt) || throw(BoundsError(tt.data, i))
    tt.llim = min(tt.llim, position - 1)
    tt.rlim = max(tt.rlim, position + 1)
    return tt
end

"""
    replaceblock!(tt, i, tensor)

Replace tensor block `i` and locally invalidate canonical bounds.
"""
function replaceblock!(tt::TensorTrain, i::Integer, tensor::Tensor)
    position = Int(i)
    1 <= position <= length(tt) || throw(BoundsError(tt.data, i))
    tt.data[position] = tensor
    invalidate_canonical!(tt, position)
    return tt
end

"""
    insert_site!(tt, position, tensor)

Insert `tensor` at `position` and fully invalidate canonical bounds.
"""
function insert_site!(tt::TensorTrain, position::Integer, tensor::Tensor)
    pos = Int(position)
    1 <= pos <= length(tt) + 1 || throw(BoundsError(tt.data, position))
    insert!(tt.data, pos, tensor)
    invalidate_canonical!(tt)
    return tt
end

"""
    delete_site!(tt, position)

Delete tensor block `position` and fully invalidate canonical bounds.
"""
function delete_site!(tt::TensorTrain, position::Integer)
    pos = Int(position)
    1 <= pos <= length(tt) || throw(BoundsError(tt.data, position))
    deleteat!(tt.data, pos)
    invalidate_canonical!(tt)
    return tt
end

function Base.insert!(tt::TensorTrain, i::Integer, value::Tensor)
    return insert_site!(tt, i, value)
end

function Base.deleteat!(tt::TensorTrain, i)
    return delete_site!(tt, i)
end

function Base.push!(tt::TensorTrain, value::Tensor)
    push!(tt.data, value)
    invalidate_canonical!(tt)
    return tt
end

function Base.pushfirst!(tt::TensorTrain, value::Tensor)
    pushfirst!(tt.data, value)
    invalidate_canonical!(tt)
    return tt
end

TensorTrain(data::AbstractVector{<:Tensor}) = TensorTrain(Tensor[tensor for tensor in data], 0, length(data) + 1)

"""
    LinearOperator(; mpo=nothing, input_indices=Index[], output_indices=Index[], ...)

Metadata wrapper for TensorTrain-backed linear operators.
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
    length(input_indices) == length(output_indices) || throw(
        DimensionMismatch(
            "LinearOperator input/output metadata length mismatch: got $(length(input_indices)) input indices and $(length(output_indices)) output indices",
        ),
    )
    if mpo !== nothing
        length(input_indices) == length(mpo) || throw(
            ArgumentError("LinearOperator.mpo has $(length(mpo)) sites but input_indices has $(length(input_indices))"),
        )
    end
    isempty(true_input) || length(true_input) == length(input_indices) || throw(
        DimensionMismatch(
            "LinearOperator true_input length mismatch: expected $(length(input_indices)), got $(length(true_input))",
        ),
    )
    isempty(true_output) || length(true_output) == length(output_indices) || throw(
        DimensionMismatch(
            "LinearOperator true_output length mismatch: expected $(length(output_indices)), got $(length(true_output))",
        ),
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
