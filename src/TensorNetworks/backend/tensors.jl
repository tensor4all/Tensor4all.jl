function _promoted_scalar_kind(tts::TensorTrain...)
    any_complex = false
    for tt in tts
        for tensor in tt.data
            T = eltype(tensor.data)
            if T <: Real
                continue
            elseif T <: Complex
                any_complex = true
            else
                throw(ArgumentError("backend apply supports only real or complex tensors, got eltype $T"))
            end
        end
    end
    return any_complex ? :c64 : :f64
end

function _interleaved_complex_data(data)
    raw = Vector{Float64}(undef, 2 * length(data))
    for (n, value) in enumerate(data)
        z = ComplexF64(value)
        raw[2n-1] = real(z)
        raw[2n] = imag(z)
    end
    return raw
end

_scalar_kind_code(::Val{:f64}) = _T4A_SCALAR_KIND_F64
_scalar_kind_code(::Val{:c64}) = _T4A_SCALAR_KIND_C64

function _scalar_kind_code(scalar_kind::Symbol)
    scalar_kind === :f64 && return _T4A_SCALAR_KIND_F64
    scalar_kind === :c64 && return _T4A_SCALAR_KIND_C64
    throw(ArgumentError("Unknown scalar kind $scalar_kind"))
end

function _clone_tensor_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && return C_NULL
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        _t4a(:t4a_tensor_clone),
        Cint,
        (Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
        ptr,
        out,
    )
    _check_backend_status(status, "cloning backend tensor")
    return out[]
end

function _backend_handle_for_clone(tensor::Tensor, scalar_kind::Symbol)
    tensor.backend_handle isa BackendTensorHandle || return C_NULL

    ptr = backend_handle_ptr(tensor)
    ptr == C_NULL && return C_NULL
    _tensor_scalar_kind_from_handle(ptr) == _scalar_kind_code(scalar_kind) || return C_NULL
    _tensor_indices_from_handle(ptr) == inds(tensor) || return C_NULL
    return ptr
end

function _new_tensor_handle(tensor::Tensor, scalar_kind::Symbol)
    clone_source = _backend_handle_for_clone(tensor, scalar_kind)
    clone_source == C_NULL || return _clone_tensor_handle(clone_source)

    index_handles = Ptr{Cvoid}[]
    try
        for index in inds(tensor)
            push!(index_handles, _new_index_handle(index))
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        if scalar_kind === :f64
            # convert preserves array shape; broadcasting Float64.(arr) collapses
            # rank-0 arrays to a scalar, which then breaks the Ptr{Float64} ccall.
            dense = convert(Array{Float64,ndims(tensor.data)}, tensor.data)
            status = ccall(
                _t4a(:t4a_tensor_new_dense_f64),
                Cint,
                (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t, Ref{Ptr{Cvoid}}),
                rank(tensor),
                index_handles,
                dense,
                length(dense),
                out,
            )
        elseif scalar_kind === :c64
            dense = _interleaved_complex_data(tensor.data)
            status = ccall(
                _t4a(:t4a_tensor_new_dense_c64),
                Cint,
                (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t, Ref{Ptr{Cvoid}}),
                rank(tensor),
                index_handles,
                dense,
                length(tensor.data),
                out,
            )
        else
            throw(ArgumentError("Unknown scalar kind $scalar_kind"))
        end

        _check_backend_status(status, "creating backend tensor")
        return out[]
    finally
        for handle in index_handles
            _release_index_handle(handle)
        end
    end
end

function _new_diag_tensor_handle(
    values::AbstractVector,
    indices::AbstractVector{Index},
    scalar_kind::Symbol,
)
    index_handles = Ptr{Cvoid}[]
    try
        for index in indices
            push!(index_handles, _new_index_handle(index))
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        if scalar_kind === :f64
            payload = Vector{Float64}(values)
            status = ccall(
                _t4a(:t4a_tensor_new_diag_f64),
                Cint,
                (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t, Ref{Ptr{Cvoid}}),
                length(indices),
                index_handles,
                payload,
                length(payload),
                out,
            )
        elseif scalar_kind === :c64
            payload = _interleaved_complex_data(values)
            status = ccall(
                _t4a(:t4a_tensor_new_diag_c64),
                Cint,
                (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t, Ref{Ptr{Cvoid}}),
                length(indices),
                index_handles,
                payload,
                length(values),
                out,
            )
        else
            throw(ArgumentError("Unknown scalar kind $scalar_kind"))
        end

        _check_backend_status(status, "creating backend diagonal tensor")
        return out[]
    finally
        for handle in index_handles
            _release_index_handle(handle)
        end
    end
end

function _new_structured_tensor_handle(
    values::AbstractVector,
    indices::AbstractVector{Index},
    payload_dims::AbstractVector{<:Integer},
    payload_strides::AbstractVector{<:Integer},
    axis_classes::AbstractVector{<:Integer},
    scalar_kind::Symbol,
)
    index_handles = Ptr{Cvoid}[]
    payload_dims_c = Csize_t[Csize_t(dim) for dim in payload_dims]
    payload_strides_c = Cptrdiff_t[Cptrdiff_t(stride) for stride in payload_strides]
    axis_classes_c = Csize_t[Csize_t(class) for class in axis_classes]

    try
        for index in indices
            push!(index_handles, _new_index_handle(index))
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        if scalar_kind === :f64
            payload = Vector{Float64}(values)
            status = ccall(
                _t4a(:t4a_tensor_new_structured_f64),
                Cint,
                (
                    Csize_t,
                    Ptr{Ptr{Cvoid}},
                    Ptr{Float64},
                    Csize_t,
                    Ptr{Csize_t},
                    Csize_t,
                    Ptr{Cptrdiff_t},
                    Csize_t,
                    Ptr{Csize_t},
                    Csize_t,
                    Ref{Ptr{Cvoid}},
                ),
                length(indices),
                index_handles,
                payload,
                length(payload),
                payload_dims_c,
                length(payload_dims_c),
                payload_strides_c,
                length(payload_strides_c),
                axis_classes_c,
                length(axis_classes_c),
                out,
            )
        elseif scalar_kind === :c64
            payload = _interleaved_complex_data(values)
            status = ccall(
                _t4a(:t4a_tensor_new_structured_c64),
                Cint,
                (
                    Csize_t,
                    Ptr{Ptr{Cvoid}},
                    Ptr{Float64},
                    Csize_t,
                    Ptr{Csize_t},
                    Csize_t,
                    Ptr{Cptrdiff_t},
                    Csize_t,
                    Ptr{Csize_t},
                    Csize_t,
                    Ref{Ptr{Cvoid}},
                ),
                length(indices),
                index_handles,
                payload,
                length(values),
                payload_dims_c,
                length(payload_dims_c),
                payload_strides_c,
                length(payload_strides_c),
                axis_classes_c,
                length(axis_classes_c),
                out,
            )
        else
            throw(ArgumentError("Unknown scalar kind $scalar_kind"))
        end

        _check_backend_status(status, "creating backend structured tensor")
        return out[]
    finally
        for handle in index_handles
            _release_index_handle(handle)
        end
    end
end

function _tensor_indices_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_tensor_indices),
        Cint,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend tensor indices")

    handles = Vector{Ptr{Cvoid}}(undef, Int(out_len[]))
    status = ccall(
        _t4a(:t4a_tensor_indices),
        Cint,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, Ref{Csize_t}),
        ptr,
        handles,
        length(handles),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor indices")

    indices = Index[]
    try
        for handle in handles
            push!(indices, _index_from_handle(handle))
        end
    finally
        for handle in handles
            _release_index_handle(handle)
        end
    end
    return indices
end

function _tensor_dims_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_tensor_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend tensor dimensions")

    dims = Vector{Csize_t}(undef, Int(out_len[]))
    status = ccall(
        _t4a(:t4a_tensor_dims),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        dims,
        length(dims),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor dimensions")
    return Int.(dims)
end

function _tensor_scalar_kind_from_handle(ptr::Ptr{Cvoid})
    out_kind = Ref{Cint}(0)
    status = ccall(_t4a(:t4a_tensor_scalar_kind), Cint, (Ptr{Cvoid}, Ref{Cint}), ptr, out_kind)
    _check_backend_status(status, "querying backend tensor scalar kind")
    return out_kind[]
end

function _storage_kind_from_handle(ptr::Ptr{Cvoid})
    out_kind = Ref{Cint}(0)
    status = ccall(_t4a(:t4a_tensor_storage_kind), Cint, (Ptr{Cvoid}, Ref{Cint}), ptr, out_kind)
    _check_backend_status(status, "querying backend tensor storage kind")
    return out_kind[]
end

function _payload_rank_from_handle(ptr::Ptr{Cvoid})
    out_rank = Ref{Csize_t}(0)
    status = ccall(_t4a(:t4a_tensor_payload_rank), Cint, (Ptr{Cvoid}, Ref{Csize_t}), ptr, out_rank)
    _check_backend_status(status, "querying backend tensor payload rank")
    return Int(out_rank[])
end

function _payload_len_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(_t4a(:t4a_tensor_payload_len), Cint, (Ptr{Cvoid}, Ref{Csize_t}), ptr, out_len)
    _check_backend_status(status, "querying backend tensor payload length")
    return Int(out_len[])
end

function _copy_csize_t_tensor_buffer(ptr::Ptr{Cvoid}, symbol::Symbol, context::AbstractString)
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(symbol),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend tensor $context")

    len = Int(out_len[])
    len == 0 && return Int[]
    buffer = Vector{Csize_t}(undef, len)
    status = ccall(
        _t4a(symbol),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        buffer,
        length(buffer),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor $context")
    return Int.(buffer)
end

function _copy_cptrdiff_t_tensor_buffer(ptr::Ptr{Cvoid}, symbol::Symbol, context::AbstractString)
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(symbol),
        Cint,
        (Ptr{Cvoid}, Ptr{Cptrdiff_t}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend tensor $context")

    len = Int(out_len[])
    len == 0 && return Int[]
    buffer = Vector{Cptrdiff_t}(undef, len)
    status = ccall(
        _t4a(symbol),
        Cint,
        (Ptr{Cvoid}, Ptr{Cptrdiff_t}, Csize_t, Ref{Csize_t}),
        ptr,
        buffer,
        length(buffer),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor $context")
    return Int.(buffer)
end

_payload_dims_from_handle(ptr::Ptr{Cvoid}) =
    _copy_csize_t_tensor_buffer(ptr, :t4a_tensor_payload_dims, "payload dimensions")

_payload_strides_from_handle(ptr::Ptr{Cvoid}) =
    _copy_cptrdiff_t_tensor_buffer(ptr, :t4a_tensor_payload_strides, "payload strides")

_axis_classes_from_handle(ptr::Ptr{Cvoid}) =
    _copy_csize_t_tensor_buffer(ptr, :t4a_tensor_axis_classes, "axis classes")

function _read_dense_f64_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_tensor_copy_dense_f64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend dense tensor data")

    dense = Vector{Float64}(undef, Int(out_len[]))
    status = ccall(
        _t4a(:t4a_tensor_copy_dense_f64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        dense,
        length(dense),
        out_len,
    )
    _check_backend_status(status, "copying backend dense tensor data")
    return dense
end

function _read_dense_c64_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_tensor_copy_dense_c64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend dense tensor data")

    raw = Vector{Float64}(undef, 2 * Int(out_len[]))
    status = ccall(
        _t4a(:t4a_tensor_copy_dense_c64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        raw,
        Int(out_len[]),
        out_len,
    )
    _check_backend_status(status, "copying backend dense tensor data")

    dense = Vector{ComplexF64}(undef, Int(out_len[]))
    for n in eachindex(dense)
        dense[n] = ComplexF64(raw[2n-1], raw[2n])
    end
    return dense
end

function _read_payload_f64_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_tensor_copy_payload_f64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend tensor payload data")

    payload = Vector{Float64}(undef, Int(out_len[]))
    status = ccall(
        _t4a(:t4a_tensor_copy_payload_f64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        payload,
        length(payload),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor payload data")
    return payload
end

function _read_payload_c64_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_tensor_copy_payload_c64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend tensor payload data")

    raw = Vector{Float64}(undef, 2 * Int(out_len[]))
    status = ccall(
        _t4a(:t4a_tensor_copy_payload_c64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        raw,
        Int(out_len[]),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor payload data")

    payload = Vector{ComplexF64}(undef, Int(out_len[]))
    for n in eachindex(payload)
        payload[n] = ComplexF64(raw[2n-1], raw[2n])
    end
    return payload
end

function _payload_from_handle(ptr::Ptr{Cvoid})
    scalar_kind = _tensor_scalar_kind_from_handle(ptr)
    if scalar_kind == _T4A_SCALAR_KIND_F64
        return _read_payload_f64_from_handle(ptr)
    elseif scalar_kind == _T4A_SCALAR_KIND_C64
        return _read_payload_c64_from_handle(ptr)
    else
        throw(ArgumentError("Unsupported backend scalar kind $scalar_kind"))
    end
end

function _tensor_from_handle(ptr::Ptr{Cvoid})
    dims = _tensor_dims_from_handle(ptr)
    indices = _tensor_indices_from_handle(ptr)
    scalar_kind = _tensor_scalar_kind_from_handle(ptr)

    if scalar_kind == _T4A_SCALAR_KIND_F64
        dense = _read_dense_f64_from_handle(ptr)
    elseif scalar_kind == _T4A_SCALAR_KIND_C64
        dense = _read_dense_c64_from_handle(ptr)
    else
        throw(ArgumentError("Unsupported backend scalar kind $scalar_kind"))
    end

    data = copy(reshape(dense, Tuple(dims)))
    handle = _clone_tensor_handle(ptr)
    try
        tensor = Tensor(data, indices; backend_handle=_owned_backend_tensor_handle(handle))
        handle = C_NULL
        return tensor
    finally
        _release_tensor_handle(handle)
    end
end
