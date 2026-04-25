"""
    insert_operator_identity!(op, position, input_index, output_index;
                              true_input=nothing, true_output=nothing)

Insert an identity MPO site into `op` and update all `LinearOperator`
metadata arrays at the same position.
"""
function insert_operator_identity!(
    op::LinearOperator,
    position::Integer,
    input_index::Index,
    output_index::Index;
    true_input::Union{Index, Nothing}=nothing,
    true_output::Union{Index, Nothing}=nothing,
)
    _validate_operator_metadata(op, "insert_operator_identity!")
    1 <= position <= length(op.input_indices) + 1 || throw(
        ArgumentError("insert position must be in 1:$(length(op.input_indices) + 1), got $position"),
    )
    dim(input_index) == dim(output_index) || throw(
        DimensionMismatch("identity input/output dimensions must match, got $(dim(input_index)) and $(dim(output_index))"),
    )

    canonical = _operator_canonical_tensors(op)
    new_input = copy(op.input_indices)
    new_output = copy(op.output_indices)
    insert!(new_input, Int(position), input_index)
    insert!(new_output, Int(position), output_index)

    op.mpo = _operator_matchsiteinds(
        canonical,
        copy(op.input_indices),
        copy(op.output_indices),
        new_input,
        new_output,
    )
    op.input_indices = new_input
    op.output_indices = new_output
    insert!(op.true_input, Int(position), true_input)
    insert!(op.true_output, Int(position), true_output)
    return _invalidate_operator_mpo!(op)
end

"""
    delete_operator_site!(op, position)

Delete one operator site and its input/output and bound-space metadata. The
neighboring bond dimensions must be compatible after the site is removed.
"""
function delete_operator_site!(op::LinearOperator, position::Integer)
    return delete_operator_sites!(op, [position])
end

"""
    delete_operator_sites!(op, positions)

Delete multiple operator sites and their corresponding metadata entries.
"""
function delete_operator_sites!(op::LinearOperator, positions::AbstractVector{<:Integer})
    _validate_operator_metadata(op, "delete_operator_sites!")
    isempty(positions) && return op
    nsites = length(op.input_indices)
    sorted_positions = sort(unique(Int.(positions)))
    all(position -> 1 <= position <= nsites, sorted_positions) || throw(
        ArgumentError("delete positions must be in 1:$nsites, got $positions"),
    )
    length(sorted_positions) < nsites || throw(
        ArgumentError("delete_operator_sites! cannot delete every operator site"),
    )

    canonical = _operator_canonical_tensors(op)
    keep = [position for position in 1:nsites if position ∉ sorted_positions]
    kept_tensors = canonical[keep]
    new_input = op.input_indices[keep]
    new_output = op.output_indices[keep]

    op.mpo = _operator_matchsiteinds(
        kept_tensors,
        new_input,
        new_output,
        new_input,
        new_output,
    )
    op.input_indices = copy(new_input)
    op.output_indices = copy(new_output)
    deleteat!(op.true_input, sorted_positions)
    deleteat!(op.true_output, sorted_positions)
    return _invalidate_operator_mpo!(op)
end

function _validate_permutation(order::AbstractVector{<:Integer}, nsites::Int)
    length(order) == nsites || throw(
        DimensionMismatch("permutation length must be $nsites, got $(length(order))"),
    )
    Set(Int.(order)) == Set(1:nsites) || throw(
        ArgumentError("order must be a permutation of 1:$nsites, got $order"),
    )
    return Int.(order)
end

"""
    permute_operator_sites!(op, order; kwargs...)

Permute `op`'s MPO topology and all per-site metadata arrays together.
Keyword arguments are forwarded to [`restructure_to`](@ref).
"""
function permute_operator_sites!(op::LinearOperator, order::AbstractVector{<:Integer}; kwargs...)
    mpo = _validate_operator_metadata(op, "permute_operator_sites!")
    permutation = _validate_permutation(order, length(op.input_indices))
    current_groups = _siteinds_by_tensor(mpo)
    target_groups = [current_groups[position] for position in permutation]

    op.mpo = restructure_to(mpo, target_groups; kwargs...)
    op.input_indices = op.input_indices[permutation]
    op.output_indices = op.output_indices[permutation]
    op.true_input = op.true_input[permutation]
    op.true_output = op.true_output[permutation]
    return _invalidate_operator_mpo!(op)
end

function _replace_operator_indices!(
    op::LinearOperator,
    field::Symbol,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    _validate_operator_metadata(op, "replace_operator_$(field)_indices!")
    replacements = _replacement_mapping(oldsites, newsites)
    metadata = getfield(op, field)
    for index in oldsites
        any(existing -> existing == index, metadata) || throw(
            ArgumentError("operator $(field) index $index does not occur in LinearOperator metadata"),
        )
    end
    replace_siteinds!(op.mpo, oldsites, newsites)
    setfield!(op, field, [get(replacements, index, index) for index in metadata])
    return _invalidate_operator_mpo!(op)
end

"""
    replace_operator_input_indices!(op, oldsites, newsites)

Rename internal operator input site indices in both `op.mpo` and
`op.input_indices`. Bound `true_input` spaces are preserved.
"""
function replace_operator_input_indices!(
    op::LinearOperator,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    return _replace_operator_indices!(op, :input_indices, oldsites, newsites)
end

"""
    replace_operator_output_indices!(op, oldsites, newsites)

Rename internal operator output site indices in both `op.mpo` and
`op.output_indices`. Bound `true_output` spaces are preserved.
"""
function replace_operator_output_indices!(
    op::LinearOperator,
    oldsites::AbstractVector{<:Index},
    newsites::AbstractVector{<:Index},
)
    return _replace_operator_indices!(op, :output_indices, oldsites, newsites)
end

"""
    transpose(op::LinearOperator)

Return the transposed operator by swapping internal input/output spaces,
bound true spaces, and the corresponding physical tensor axes.
"""
function Base.transpose(op::LinearOperator)
    mpo = _validate_operator_metadata(op, "transpose")
    tensors = Tensor[]
    for position in eachindex(mpo.data)
        left, right = _operator_link_indices(
            mpo,
            position,
            op.input_indices[position],
            op.output_indices[position],
        )
        canonical = _operator_canonical_tensor(
            mpo,
            position,
            op.input_indices[position],
            op.output_indices[position],
        )
        transposed = permutedims(canonical, (1, 3, 2, 4))
        push!(
            tensors,
            _operator_tensor_from_canonical(
                transposed,
                left,
                op.output_indices[position],
                op.input_indices[position],
                right,
                position,
                length(mpo),
            ),
        )
    end

    return LinearOperator(;
        mpo=TensorTrain(tensors, 0, length(tensors) + 1),
        input_indices=copy(op.output_indices),
        output_indices=copy(op.input_indices),
        true_input=copy(op.true_output),
        true_output=copy(op.true_input),
        metadata=op.metadata,
    )
end
