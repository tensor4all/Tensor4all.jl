"""
    storage_kind(t)

Return the compact storage kind for `t` as `:dense`, `:diagonal`, or
`:structured`.
"""
function storage_kind(t::Tensor)
    if _has_backend_tensor_handle(t)
        tn = _tensor_networks_module()
        return _storage_kind_symbol(tn._storage_kind_from_handle(backend_handle_ptr(t)))
    end
    return :dense
end

function _storage_kind_symbol(kind::Integer)
    tn = _tensor_networks_module()
    kind == tn._T4A_STORAGE_KIND_DENSE && return :dense
    kind == tn._T4A_STORAGE_KIND_DIAGONAL && return :diagonal
    kind == tn._T4A_STORAGE_KIND_STRUCTURED && return :structured
    throw(ArgumentError("Unsupported backend tensor storage kind $kind"))
end

_has_backend_tensor_handle(t::Tensor) =
    t.backend_handle isa BackendTensorHandle && backend_handle_ptr(t) != C_NULL

"""
    payload_rank(t)

Return the rank of `t`'s compact payload storage.
"""
function payload_rank(t::Tensor)
    if _has_backend_tensor_handle(t)
        return _tensor_networks_module()._payload_rank_from_handle(backend_handle_ptr(t))
    end
    return rank(t)
end

"""
    payload_len(t)

Return the number of scalar elements in `t`'s compact payload storage.
"""
function payload_len(t::Tensor)
    if _has_backend_tensor_handle(t)
        return _tensor_networks_module()._payload_len_from_handle(backend_handle_ptr(t))
    end
    return length(t.data)
end

"""
    payload_dims(t)

Return compact payload dimensions for `t`.
"""
function payload_dims(t::Tensor)
    if _has_backend_tensor_handle(t)
        return _tensor_networks_module()._payload_dims_from_handle(backend_handle_ptr(t))
    end
    return collect(size(t.data))
end

"""
    payload_strides(t)

Return compact payload strides in scalar elements.
"""
function payload_strides(t::Tensor)
    if _has_backend_tensor_handle(t)
        return _tensor_networks_module()._payload_strides_from_handle(backend_handle_ptr(t))
    end
    return collect(strides(t.data))
end

"""
    axis_classes(t)

Return the zero-based mapping from logical tensor axes to compact payload axes.
"""
function axis_classes(t::Tensor)
    if _has_backend_tensor_handle(t)
        return _tensor_networks_module()._axis_classes_from_handle(backend_handle_ptr(t))
    end
    return collect(0:(rank(t)-1))
end

"""
    payload(t)

Return a copy of `t`'s compact payload values in column-major payload order.
"""
function payload(t::Tensor)
    if _has_backend_tensor_handle(t)
        return _tensor_networks_module()._payload_from_handle(backend_handle_ptr(t))
    end
    return vec(copy(t.data))
end

function _storage_payload(values::AbstractVector)
    T = eltype(values)
    if T <: Real
        return Vector{Float64}(values), :f64
    elseif T <: Complex
        return Vector{ComplexF64}(values), :c64
    end
    throw(ArgumentError("structured tensor storage supports only real or complex values, got eltype $T"))
end

function _validate_diagonal_payload(values::AbstractVector, siteinds::Vector{Index})
    isempty(siteinds) && return nothing

    diag_len = length(values)
    for index in siteinds
        dim(index) == diag_len || throw(DimensionMismatch(
            "diagtensor requires every index dimension to match payload length $diag_len; got $(dim(index)) for $index",
        ))
    end
    return nothing
end

function _diagonal_dense(payload_values::AbstractVector, siteinds::Vector{Index})
    if isempty(siteinds)
        length(payload_values) == 1 || throw(DimensionMismatch(
            "rank-0 diagtensor requires exactly one payload value, got $(length(payload_values))",
        ))
        data = Array{eltype(payload_values),0}(undef)
        data[] = only(payload_values)
        return data
    end

    data = zeros(eltype(payload_values), Tuple(dim.(siteinds)))
    for n in eachindex(payload_values)
        data[ntuple(_ -> n, length(siteinds))...] = payload_values[n]
    end
    return data
end

"""
    diagtensor(values, inds)

Create a tensor with diagonal compact backend storage and dense Julia data.

Every logical index dimension must equal `length(values)`.
"""
function diagtensor(values::AbstractVector, siteinds::AbstractVector{<:Index})
    indices = collect(siteinds)
    payload_values, scalar_kind = _storage_payload(values)
    _validate_diagonal_payload(payload_values, indices)
    data = _diagonal_dense(payload_values, indices)

    isempty(indices) && return Tensor(data, Index[])

    tn = _tensor_networks_module()
    handle = tn._new_diag_tensor_handle(payload_values, indices, scalar_kind)
    try
        tensor = Tensor(data, indices; backend_handle=_owned_backend_tensor_handle(handle))
        handle = C_NULL
        return tensor
    finally
        tn._release_tensor_handle(handle)
    end
end

"""
    delta(i, j, more...; T=Float64)

Create a diagonal identity-like tensor over two or more matching indices.
"""
function delta(i::Index, j::Index, more::Index...; T=Float64)
    return diagtensor(ones(T, dim(i)), Index[i, j, more...])
end

"""
    identity_tensor(i, j; T=Float64)

Alias for `delta(i, j; T)`.
"""
identity_tensor(i::Index, j::Index; T=Float64) = delta(i, j; T)
