function _positions_for_index(tt::TensorTrain, index::Index, context::AbstractString)
    positions = findsites(tt, index)
    isempty(positions) && throw(ArgumentError("$context index $index not found in TensorTrain"))
    return positions
end

function _copy_and_apply_tensor_op(op::Function, tt::TensorTrain, context::AbstractString, specs)
    copied = _copy_train(tt)
    for spec in specs
        index = spec isa Pair ? first(spec) : spec
        for position in _positions_for_index(copied, index, context)
            replaceblock!(copied, position, op(copied[position], spec))
        end
    end
    return copied
end

"""
    fixinds(tt, replacements...)

Return a copy of `tt` with site indices fixed to 1-based values.
"""
function fixinds(tt::TensorTrain, replacements::Pair{Index,<:Integer}...)
    return _copy_and_apply_tensor_op(tt, "fixinds", replacements) do tensor, replacement
        fixinds(tensor, replacement)
    end
end

"""
    suminds(tt, indices...)

Return a copy of `tt` with `indices` summed out of their local tensors.
"""
function suminds(tt::TensorTrain, indices::Index...)
    return _copy_and_apply_tensor_op(tt, "suminds", indices) do tensor, index
        suminds(tensor, index)
    end
end

"""
    projectinds(tt, replacements...)

Return a copy of `tt` with site indices projected to explicit value subsets.
"""
function projectinds(
    tt::TensorTrain,
    replacements::Pair{Index,<:AbstractVector{<:Integer}}...,
)
    return _copy_and_apply_tensor_op(tt, "projectinds", replacements) do tensor, replacement
        projectinds(tensor, replacement)
    end
end
