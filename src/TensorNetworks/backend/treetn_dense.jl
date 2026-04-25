"""
    to_dense(tt::TensorTrain) -> Tensor

Contract every bond of `tt` and return the result as a single dense `Tensor`
whose indices are the site (non-bond) indices.

For a `TensorTrain` with `n` site tensors and a total of `s` site indices,
the resulting `Tensor` has rank `s` (the union of all non-bond indices). The
underlying contraction is performed by `t4a_treetn_to_dense`; bond indices
are summed out.

Throws `ArgumentError` if `tt` is empty.
"""
function to_dense(tt::TensorTrain)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for to_dense"))

    scalar_kind = _promoted_scalar_kind(tt)
    tt_handle = _new_treetn_handle(tt, scalar_kind)
    result_handle = C_NULL
    try
        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            _t4a(:t4a_treetn_to_dense),
            Cint,
            (Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
            tt_handle,
            out,
        )
        _check_backend_status(status, "materializing TensorTrain to dense Tensor")
        result_handle = out[]
        return _restore_dense_site_metadata(tt, _tensor_from_handle(result_handle))
    finally
        _release_tensor_handle(result_handle)
        _release_treetn_handle(tt_handle)
    end
end

function _restore_dense_site_metadata(tt::TensorTrain, tensor::Tensor)
    requested_by_id = Dict{UInt64, Index}()
    for group in _siteinds_by_tensor(tt), index in group
        requested_by_id[id(index)] = index
    end

    restored = map(inds(tensor)) do index
        return get(requested_by_id, id(index), index)
    end
    return restored == inds(tensor) ? tensor : Tensor(tensor.data, restored)
end
