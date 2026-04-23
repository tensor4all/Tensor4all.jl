function _insert_identity_eltype(tt::TensorTrain, position::Integer, T)
    T === nothing || return T
    isempty(tt.data) && return Float64
    if position == 0
        return eltype(first(tt.data))
    elseif position == length(tt)
        return eltype(last(tt.data))
    end
    return promote_type(eltype(tt[position]), eltype(tt[position + 1]))
end

function _fresh_link_like(link::Index)
    return Index(dim(link); tags=tags(link), plev=plev(link))
end

function _boundary_link(position::Integer)
    return Index(1; tags=[_LINK_TAG, "insert=$position"])
end

function _prepend_link_axis(tensor::Tensor, link::Index)
    data = reshape(tensor.data, 1, size(tensor.data)...)
    return Tensor(data, Index[link, inds(tensor)...])
end

function _append_link_axis(tensor::Tensor, link::Index)
    data = reshape(tensor.data, size(tensor.data)..., 1)
    return Tensor(data, Index[inds(tensor)..., link])
end

"""
    identity_link_tensor(left, right, site; T=Float64)

Create a three-index copy tensor that preserves a chain when `site` is summed
out.
"""
function identity_link_tensor(left::Index, right::Index, site::Index; T=Float64)
    return diagtensor(ones(T, dim(left)), Index[left, site, right])
end

function _insert_left_boundary_identity!(tt::TensorTrain, newsite::Index, T)
    link = _boundary_link(0)
    identity = Tensor(ones(T, dim(newsite), 1), [newsite, link])
    if !isempty(tt.data)
        replaceblock!(tt, 1, _prepend_link_axis(tt[1], link))
    end
    return insert_site!(tt, 1, identity)
end

function _insert_right_boundary_identity!(tt::TensorTrain, newsite::Index, T)
    link = _boundary_link(length(tt))
    identity = Tensor(ones(T, 1, dim(newsite)), [link, newsite])
    if !isempty(tt.data)
        replaceblock!(tt, length(tt), _append_link_axis(tt[end], link))
    end
    return insert_site!(tt, length(tt) + 1, identity)
end

"""
    insert_identity!(tt, newsite, position; T=nothing)

Insert an identity/copy site after `position`, where `position == 0` inserts
before the first site and `position == length(tt)` inserts after the last site.
"""
function insert_identity!(
    tt::TensorTrain,
    newsite::Index,
    position::Integer;
    T=nothing,
)
    pos = Int(position)
    0 <= pos <= length(tt) || throw(ArgumentError(
        "insert_identity! position must be in 0:$(length(tt)), got $position",
    ))

    value_type = _insert_identity_eltype(tt, pos, T)
    if pos == 0
        return _insert_left_boundary_identity!(tt, newsite, value_type)
    elseif pos == length(tt)
        return _insert_right_boundary_identity!(tt, newsite, value_type)
    end

    oldlink = linkinds(tt, pos)
    newlink = _fresh_link_like(oldlink)
    replaceinds!(tt[pos + 1], [oldlink], [newlink])
    identity = identity_link_tensor(oldlink, newlink, newsite; T=value_type)
    return insert_site!(tt, pos + 1, identity)
end
