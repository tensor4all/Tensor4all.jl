function _new_treetn_handle(tt::TensorTrain, scalar_kind::Symbol)
    tensor_handles = Ptr{Cvoid}[]
    try
        for tensor in tt.data
            push!(tensor_handles, _new_tensor_handle(tensor, scalar_kind))
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_new),
            Cint,
            (Ptr{Ptr{Cvoid}}, Csize_t, Ref{Ptr{Cvoid}}),
            tensor_handles,
            length(tensor_handles),
            out,
        )
        _check_backend_status(status, "creating backend TensorTrain")
        return out[]
    finally
        for handle in tensor_handles
            _release_tensor_handle(handle)
        end
    end
end

function _treetn_num_vertices(ptr::Ptr{Cvoid})
    out_n = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_treetn_num_vertices),
        Cint,
        (Ptr{Cvoid}, Ref{Csize_t}),
        ptr,
        out_n,
    )
    _check_backend_status(status, "querying backend TensorTrain length")
    return Int(out_n[])
end

function _treetn_tensor_handle(ptr::Ptr{Cvoid}, vertex::Integer)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(
        _t4a(:t4a_treetn_tensor),
        Cint,
        (Ptr{Cvoid}, Csize_t, Ref{Ptr{Cvoid}}),
        ptr,
        vertex,
        out,
    )
    _check_backend_status(status, "querying backend tensor at vertex $(vertex + 1)")
    return out[]
end

function _treetn_canonical_region(ptr::Ptr{Cvoid})
    out_len = Ref{Csize_t}(0)
    status = ccall(
        _t4a(:t4a_treetn_canonical_region),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        Csize_t(0),
        out_len,
    )
    _check_backend_status(status, "querying backend canonical region length")

    nvertices = Int(out_len[])
    nvertices == 0 && return Int[]

    vertices = Vector{Csize_t}(undef, nvertices)
    status = ccall(
        _t4a(:t4a_treetn_canonical_region),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        vertices,
        Csize_t(nvertices),
        out_len,
    )
    _check_backend_status(status, "copying backend canonical region")
    return Int.(vertices)
end

function _derive_llim_rlim(canonical_region::Vector{Int}, ntensors::Int)
    if length(canonical_region) == 1
        center = only(canonical_region)
        return center, center + 2
    end
    return 0, ntensors + 1
end

function _treetn_scale(tt::TensorTrain, re::Float64, im::Float64)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for scalar multiply"))

    scalar_kind = im == 0.0 ? _promoted_scalar_kind(tt) : :c64
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_scale),
            Cint,
            (Ptr{Cvoid}, Cdouble, Cdouble, Ref{Ptr{Cvoid}}),
            tt_handle,
            re,
            im,
            out,
        )
        _check_backend_status(status, "scaling TensorTrain")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_treetn_handle(tt_handle)
    end
end

function _treetn_from_handle(ptr::Ptr{Cvoid})
    ntensors = _treetn_num_vertices(ptr)
    tensors = Tensor[]
    for vertex in 0:(ntensors - 1)
        tensor_handle = _treetn_tensor_handle(ptr, vertex)
        try
            push!(tensors, _tensor_from_handle(tensor_handle))
        finally
            _release_tensor_handle(tensor_handle)
        end
    end

    canonical_region = _treetn_canonical_region(ptr)
    llim, rlim = _derive_llim_rlim(canonical_region, ntensors)
    return TensorTrain(tensors, llim, rlim)
end

Base.:*(α::Number, tt::TensorTrain) = _treetn_scale(tt, Float64(real(α)), Float64(imag(α)))
Base.:*(tt::TensorTrain, α::Number) = α * tt
Base.:/(tt::TensorTrain, α::Number) = tt * inv(α)
Base.:-(tt::TensorTrain) = _treetn_scale(tt, -1.0, 0.0)
