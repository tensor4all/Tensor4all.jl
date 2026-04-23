"""
    invalidate_canonical!(tt)

Mark every bond in `tt` as non-canonical.
"""
function invalidate_canonical!(tt::TensorTrain)
    tt.llim = 0
    tt.rlim = length(tt) + 1
    return tt
end

"""
    invalidate_canonical!(tt, i)

Mark tensor position `i` and its neighboring bonds as non-canonical.
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

function Base.setindex!(tt::TensorTrain, value::Tensor, i::Int)
    replaceblock!(tt, i, value)
    return value
end

"""
    insert_site!(tt, position, tensor)

Insert `tensor` at `position` and fully invalidate canonical bounds.
"""
function insert_site!(tt::TensorTrain, position::Integer, tensor::Tensor)
    pos = Int(position)
    1 <= pos <= length(tt) + 1 || throw(BoundsError(tt.data, position))
    Base.insert!(tt.data, pos, tensor)
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
    Base.deleteat!(tt.data, pos)
    invalidate_canonical!(tt)
    return tt
end

Base.insert!(tt::TensorTrain, position::Integer, tensor::Tensor) =
    insert_site!(tt, position, tensor)

Base.deleteat!(tt::TensorTrain, position::Integer) =
    delete_site!(tt, position)

function Base.push!(tt::TensorTrain, tensor::Tensor)
    push!(tt.data, tensor)
    invalidate_canonical!(tt)
    return tt
end

function Base.pushfirst!(tt::TensorTrain, tensor::Tensor)
    pushfirst!(tt.data, tensor)
    invalidate_canonical!(tt)
    return tt
end
