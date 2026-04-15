"""
    set_input_space!(op, indices)

Bind explicit input indices onto `op`.
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

Bind explicit output indices onto `op`.
"""
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

"""
    set_iospaces!(op, input_indices, output_indices=input_indices)

Bind explicit input and output indices onto `op`.
"""
function set_iospaces!(op::LinearOperator, input_indices::Vector{Index}, output_indices::Vector{Index}=input_indices)
    set_input_space!(op, input_indices)
    set_output_space!(op, output_indices)
    return op
end

set_iospaces!(op::LinearOperator, ::TensorTrain; kwargs...) = throw(
    SkeletonNotImplemented(:set_iospaces!, :tt),
)
