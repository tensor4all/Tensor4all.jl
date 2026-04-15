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

function _new_tensor_handle(tensor::Tensor, scalar_kind::Symbol)
    index_handles = Ptr{Cvoid}[]
    try
        for index in inds(tensor)
            push!(index_handles, _new_index_handle(index))
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        if scalar_kind === :f64
            dense = Float64.(tensor.data)
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
    return Tensor(data, indices)
end
