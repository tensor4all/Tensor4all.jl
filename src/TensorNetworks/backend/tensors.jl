function _promoted_scalar_kind(tts::TensorTrain...)
    any_complex = false
    for tt in tts
        for tensor in tt.data
            T = eltype(tensor)
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

function _new_dense_tensor_handle(tensor::Tensor, scalar_kind::Symbol, index_handles, out)
    if scalar_kind === :f64
        # convert preserves array shape; broadcasting Float64.(arr) collapses
        # rank-0 arrays to a scalar, which then breaks the Ptr{Float64} ccall.
        tensor_data = copy_data(tensor)
        dense = convert(Array{Float64,ndims(tensor_data)}, tensor_data)
        return ccall(
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
        tensor_data = copy_data(tensor)
        dense = _interleaved_complex_data(tensor_data)
        return ccall(
            _t4a(:t4a_tensor_new_dense_c64),
            Cint,
            (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t, Ref{Ptr{Cvoid}}),
            rank(tensor),
            index_handles,
            dense,
            length(tensor_data),
            out,
        )
    end
    throw(ArgumentError("Unknown scalar kind $scalar_kind"))
end

function _new_diagonal_tensor_handle(
    tensor::Tensor,
    storage::StructuredTensorStorage,
    scalar_kind::Symbol,
    index_handles,
    out,
)
    if scalar_kind === :f64
        payload = Float64.(storage.payload)
        return ccall(
            _t4a(:t4a_tensor_new_diag_f64),
            Cint,
            (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t, Ref{Ptr{Cvoid}}),
            rank(tensor),
            index_handles,
            payload,
            length(payload),
            out,
        )
    elseif scalar_kind === :c64
        payload = _interleaved_complex_data(storage.payload)
        return ccall(
            _t4a(:t4a_tensor_new_diag_c64),
            Cint,
            (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Float64}, Csize_t, Ref{Ptr{Cvoid}}),
            rank(tensor),
            index_handles,
            payload,
            length(storage.payload),
            out,
        )
    end
    throw(ArgumentError("Unknown scalar kind $scalar_kind"))
end

function _new_explicit_structured_tensor_handle(
    tensor::Tensor,
    storage::StructuredTensorStorage,
    scalar_kind::Symbol,
    index_handles,
    out,
)
    payload_dims = Csize_t.(storage.payload_dims)
    payload_strides = Cptrdiff_t.(storage.payload_strides)
    axis_classes = Csize_t.(storage.axis_classes .- 1)

    if scalar_kind === :f64
        payload = Float64.(storage.payload)
        return ccall(
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
            rank(tensor),
            index_handles,
            payload,
            length(payload),
            payload_dims,
            length(payload_dims),
            payload_strides,
            length(payload_strides),
            axis_classes,
            length(axis_classes),
            out,
        )
    elseif scalar_kind === :c64
        payload = _interleaved_complex_data(storage.payload)
        return ccall(
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
            rank(tensor),
            index_handles,
            payload,
            length(storage.payload),
            payload_dims,
            length(payload_dims),
            payload_strides,
            length(payload_strides),
            axis_classes,
            length(axis_classes),
            out,
        )
    end
    throw(ArgumentError("Unknown scalar kind $scalar_kind"))
end

function _new_structured_tensor_handle(
    tensor::Tensor,
    storage::StructuredTensorStorage,
    scalar_kind::Symbol,
    index_handles,
    out,
)
    storage.kind === :diagonal &&
        return _new_diagonal_tensor_handle(tensor, storage, scalar_kind, index_handles, out)
    storage.kind === :structured &&
        return _new_explicit_structured_tensor_handle(tensor, storage, scalar_kind, index_handles, out)
    throw(ArgumentError("Unknown structured storage kind $(storage.kind)"))
end

function _new_tensor_handle(tensor::Tensor, scalar_kind::Symbol)
    existing_handle = _backend_handle_ptr(tensor.backend_handle)
    if existing_handle != C_NULL
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_tensor_clone),
            Cint,
            (Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
            existing_handle,
            out,
        )
        _check_backend_status(status, "cloning backend tensor")
        return out[]
    end

    index_handles = Ptr{Cvoid}[]
    try
        for index in inds(tensor)
            push!(index_handles, _new_index_handle(index))
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        storage = _structured_storage_from_tensor(tensor)
        if storage === nothing
            status = _new_dense_tensor_handle(tensor, scalar_kind, index_handles, out)
        else
            status = _new_structured_tensor_handle(tensor, storage, scalar_kind, index_handles, out)
        end

        _check_backend_status(status, "creating backend tensor")
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

function _tensor_storage_kind_from_handle(ptr::Ptr{Cvoid})
    out_kind = Ref{Cint}(0)
    status = ccall(_t4a(:t4a_tensor_storage_kind), Cint, (Ptr{Cvoid}, Ref{Cint}), ptr, out_kind)
    _check_backend_status(status, "querying backend tensor storage kind")
    return out_kind[]
end

function _storage_kind_symbol(kind::Cint)
    kind == _T4A_STORAGE_KIND_DENSE && return :dense
    kind == _T4A_STORAGE_KIND_DIAGONAL && return :diagonal
    kind == _T4A_STORAGE_KIND_STRUCTURED && return :structured
    throw(ArgumentError("Unsupported backend storage kind $kind"))
end

function _read_csize_vector_from_handle(ptr::Ptr{Cvoid}, symbol::Symbol, context::AbstractString)
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

    values = Vector{Csize_t}(undef, Int(out_len[]))
    status = ccall(
        _t4a(symbol),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        values,
        length(values),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor $context")
    return Int.(values)
end

function _tensor_payload_dims_from_handle(ptr::Ptr{Cvoid})
    return _read_csize_vector_from_handle(ptr, :t4a_tensor_payload_dims, "payload dimensions")
end

function _tensor_axis_classes_from_handle(ptr::Ptr{Cvoid})
    return _read_csize_vector_from_handle(ptr, :t4a_tensor_axis_classes, "axis classes")
end

function _tensor_payload_strides_from_handle(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_tensor_payload_strides),
        Cint,
        (Ptr{Cvoid}, Ptr{Cptrdiff_t}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    _check_backend_status(status, "querying backend tensor payload strides")

    strides = Vector{Cptrdiff_t}(undef, Int(out_len[]))
    status = ccall(
        _t4a(:t4a_tensor_payload_strides),
        Cint,
        (Ptr{Cvoid}, Ptr{Cptrdiff_t}, Csize_t, Ref{Csize_t}),
        ptr,
        strides,
        length(strides),
        out_len,
    )
    _check_backend_status(status, "copying backend tensor payload strides")
    return Int.(strides)
end

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

function _structured_storage_from_handle(ptr::Ptr{Cvoid}, scalar_kind::Cint)
    storage_kind = _tensor_storage_kind_from_handle(ptr)
    storage_kind == _T4A_STORAGE_KIND_DENSE && return nothing

    kind = _storage_kind_symbol(storage_kind)
    payload_dims = _tensor_payload_dims_from_handle(ptr)
    payload_strides = _tensor_payload_strides_from_handle(ptr)
    axis_classes = _tensor_axis_classes_from_handle(ptr) .+ 1
    if scalar_kind == _T4A_SCALAR_KIND_F64
        payload = _read_payload_f64_from_handle(ptr)
    elseif scalar_kind == _T4A_SCALAR_KIND_C64
        payload = _read_payload_c64_from_handle(ptr)
    else
        throw(ArgumentError("Unsupported backend scalar kind $scalar_kind"))
    end
    return StructuredTensorStorage{eltype(payload)}(
        kind,
        payload,
        payload_dims,
        payload_strides,
        axis_classes,
    )
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
    return Tensor(data, indices; structured_storage=_structured_storage_from_handle(ptr, scalar_kind))
end

function _lazy_tensor_from_owned_handle(ptr::Ptr{Cvoid})
    dims = _tensor_dims_from_handle(ptr)
    indices = _tensor_indices_from_handle(ptr)
    scalar_kind = _tensor_scalar_kind_from_handle(ptr)
    T = if scalar_kind == _T4A_SCALAR_KIND_F64
        Float64
    elseif scalar_kind == _T4A_SCALAR_KIND_C64
        ComplexF64
    else
        throw(ArgumentError("Unsupported backend scalar kind $scalar_kind"))
    end
    return _tensor_from_backend_handle(
        TensorHandle(ptr),
        indices,
        T,
        Val(length(dims));
        structured_storage=_structured_storage_from_handle(ptr, scalar_kind),
    )
end
