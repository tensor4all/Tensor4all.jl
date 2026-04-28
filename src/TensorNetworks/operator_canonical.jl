function _validate_operator_metadata(op::LinearOperator, action::AbstractString)
    mpo = op.mpo
    mpo === nothing && throw(ArgumentError("LinearOperator.mpo must be set before $action"))
    nsites = length(mpo)
    length(op.input_indices) == nsites || throw(
        ArgumentError("$action requires $(nsites) input indices, got $(length(op.input_indices))"),
    )
    length(op.output_indices) == nsites || throw(
        ArgumentError("$action requires $(nsites) output indices, got $(length(op.output_indices))"),
    )
    length(op.true_input) == nsites || throw(
        ArgumentError("$action requires $(nsites) true_input entries, got $(length(op.true_input))"),
    )
    length(op.true_output) == nsites || throw(
        ArgumentError("$action requires $(nsites) true_output entries, got $(length(op.true_output))"),
    )
    return mpo
end

function _invalidate_operator_mpo!(op::LinearOperator)
    op.mpo === nothing && return op
    op.mpo.llim = 0
    op.mpo.rlim = length(op.mpo) + 1
    return op
end

function _shared_index_with_neighbor(
    tensor_indices::Vector{Index},
    neighbor::Tensor,
    forbidden::Set{Index},
)
    neighbor_indices = Set(inds(neighbor))
    matches = [index for index in tensor_indices if index in neighbor_indices && index ∉ forbidden]
    length(matches) <= 1 || throw(
        ArgumentError("Expected at most one shared link index between neighboring operator tensors, got $matches"),
    )
    return isempty(matches) ? nothing : only(matches)
end

function _operator_boundary_link(position::Int, side::Symbol)
    return Index(1; tags=["Link", "operator-boundary=$position", "side=$(String(side))"])
end

function _operator_link_indices(tt::TensorTrain, position::Int, input_index::Index, output_index::Index)
    tensor_indices = inds(tt[position])
    forbidden = Set([input_index, output_index])
    left = position > 1 ? _shared_index_with_neighbor(tensor_indices, tt[position - 1], forbidden) : nothing
    right = position < length(tt) ? _shared_index_with_neighbor(tensor_indices, tt[position + 1], forbidden) : nothing

    leftovers = [
        index for index in tensor_indices
        if index ∉ forbidden &&
            (left === nothing || index != left) &&
            (right === nothing || index != right)
    ]

    if position == 1 && left === nothing
        left = isempty(leftovers) ? _operator_boundary_link(position, :left) : popfirst!(leftovers)
    end
    if position == length(tt) && right === nothing
        right = isempty(leftovers) ? _operator_boundary_link(position, :right) : pop!(leftovers)
    end
    if left === nothing
        left = _operator_boundary_link(position, :left)
    end
    if right === nothing
        right = _operator_boundary_link(position, :right)
    end
    isempty(leftovers) || throw(
        ArgumentError("Unexpected extra operator link indices at tensor $position: $leftovers"),
    )
    return left, right
end

function _axis_for_index(tensor_indices::Vector{Index}, index::Index)
    return findfirst(==(index), tensor_indices)
end

function _operator_canonical_tensor(
    tt::TensorTrain,
    position::Int,
    input_index::Index,
    output_index::Index,
)
    tensor = tt[position]
    tensor_indices = inds(tensor)
    input_axis = _axis_for_index(tensor_indices, input_index)
    output_axis = _axis_for_index(tensor_indices, output_index)
    input_axis === nothing && throw(
        ArgumentError("operator input index $input_index is not attached to tensor $position"),
    )
    output_axis === nothing && throw(
        ArgumentError("operator output index $output_index is not attached to tensor $position"),
    )

    left, right = _operator_link_indices(tt, position, input_index, output_index)
    left_axis = _axis_for_index(tensor_indices, left)
    right_axis = _axis_for_index(tensor_indices, right)

    source_axes = Int[]
    left_axis === nothing || push!(source_axes, left_axis)
    push!(source_axes, input_axis, output_axis)
    right_axis === nothing || push!(source_axes, right_axis)
    length(Set(source_axes)) == length(source_axes) || throw(
        ArgumentError("operator tensor $position has overlapping link/input/output axes"),
    )

    data = length(source_axes) == ndims(tensor.data) ? permutedims(tensor.data, Tuple(source_axes)) :
        tensor.data
    if length(source_axes) != ndims(tensor.data)
        existing_axes = Set(source_axes)
        expected_missing = count(isnothing, (left_axis, right_axis))
        ndims(tensor.data) - length(existing_axes) == 0 || throw(
            ArgumentError("operator tensor $position has unsupported extra axes"),
        )
        expected_missing in (1, 2) || throw(
            ArgumentError("operator tensor $position is missing an unexpected link axis"),
        )
    end

    return reshape(copy(data), dim(left), dim(input_index), dim(output_index), dim(right))
end

function _operator_canonical_tensors(op::LinearOperator)
    mpo = _validate_operator_metadata(op, "canonicalizing LinearOperator")
    return [
        _operator_canonical_tensor(mpo, position, op.input_indices[position], op.output_indices[position])
        for position in eachindex(mpo.data)
    ]
end

function _identity_operator_data(::Type{T}, left::Index, input::Index, output::Index, right::Index) where {T}
    dim(left) == dim(right) || throw(
        DimensionMismatch("Identity operator insertion requires matching left/right link dimensions, got $(dim(left)) and $(dim(right))"),
    )
    dim(input) == dim(output) || throw(
        DimensionMismatch("Identity operator insertion requires matching input/output dimensions, got $(dim(input)) and $(dim(output))"),
    )
    data = zeros(T, dim(left), dim(input), dim(output), dim(right))
    for link in 1:dim(left), value in 1:dim(input)
        data[link, value, value, link] = one(T)
    end
    return data
end

function _operator_tensor_from_canonical(
    data::Array,
    left::Index,
    input::Index,
    output::Index,
    right::Index,
    position::Int,
    nsites::Int,
)
    if nsites == 1
        dim(left) == 1 || throw(DimensionMismatch("single-site operator left boundary dimension must be 1"))
        dim(right) == 1 || throw(DimensionMismatch("single-site operator right boundary dimension must be 1"))
        return Tensor(reshape(data, dim(input), dim(output)), [input, output])
    elseif position == 1
        dim(left) == 1 || throw(DimensionMismatch("left boundary dimension must be 1 at operator site 1"))
        return Tensor(reshape(data, dim(input), dim(output), dim(right)), [input, output, right])
    elseif position == nsites
        dim(right) == 1 || throw(DimensionMismatch("right boundary dimension must be 1 at final operator site"))
        return Tensor(reshape(data, dim(left), dim(input), dim(output)), [left, input, output])
    end
    return Tensor(data, [left, input, output, right])
end

function _operator_matchsiteinds(
    canonical_tensors::Vector{<:Array},
    current_input::Vector{Index},
    current_output::Vector{Index},
    target_input::Vector{Index},
    target_output::Vector{Index},
)
    length(target_input) == length(target_output) || throw(
        DimensionMismatch(
            "Input and output target site sets must have the same length, got $(length(target_input)) and $(length(target_output))",
        ),
    )
    positions = _validate_target_positions(current_input, target_input)
    for (n, position) in pairs(positions)
        current_output[n] == target_output[position] || throw(
            ArgumentError("Input/output target site positions must agree for each LinearOperator site"),
        )
    end

    links = _link_indices_for_embedding(canonical_tensors, positions, length(target_input))
    sparse_by_position = Dict(positions[n] => canonical_tensors[n] for n in eachindex(positions))
    value_type = mapreduce(eltype, promote_type, canonical_tensors)

    tensors = Tensor[]
    for position in eachindex(target_input)
        left = links[position]
        right = links[position + 1]
        data = if haskey(sparse_by_position, position)
            sparse_by_position[position]
        else
            _identity_operator_data(value_type, left, target_input[position], target_output[position], right)
        end
        push!(
            tensors,
            _operator_tensor_from_canonical(
                data,
                left,
                target_input[position],
                target_output[position],
                right,
                position,
                length(target_input),
            ),
        )
    end

    return TensorTrain(tensors, 0, length(tensors) + 1)
end
