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

function _treetn_from_handle(ptr::Ptr{Cvoid}; llim::Int=0, rlim::Union{Int, Nothing}=nothing)
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

    final_rlim = isnothing(rlim) ? ntensors + 1 : rlim
    return TensorTrain(tensors, llim, final_rlim)
end
