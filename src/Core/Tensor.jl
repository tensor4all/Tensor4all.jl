"""
    Tensor(data, inds)

Create a backend-backed dense tensor from Julia array data and index metadata.

The constructor validates shape consistency in Julia, then hands ownership of
the tensor storage to the `tensor4all-capi` backend.

# Examples
```jldoctest
julia> using Tensor4all

julia> i = Index(2; tags=["i"]);

julia> j = Index(3; tags=["j"]);

julia> t = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j]);

julia> (rank(t), dims(t))
(2, (2, 3))
```
"""
mutable struct Tensor
    ptr::Ptr{Cvoid}
end

function _adopt_tensor(ptr::Ptr{Cvoid}, context::AbstractString)
    ptr = _check_ptr(ptr, context)
    tensor = Tensor(ptr)
    finalizer(t -> _release_handle!(:t4a_tensor_release, t, :ptr), tensor)
    return tensor
end

function _clone_tensor(t::Tensor)
    ptr = ccall(_capi_symbol(:t4a_tensor_clone), Ptr{Cvoid}, (Ptr{Cvoid},), t.ptr)
    return _adopt_tensor(ptr, "t4a_tensor_clone")
end

function _storage_kind(t::Tensor)
    out = Ref{Cint}(0)
    _check_status(
        ccall(
            _capi_symbol(:t4a_tensor_get_storage_kind),
            _StatusCode,
            (Ptr{Cvoid}, Ref{Cint}),
            t.ptr,
            out,
        ),
        "t4a_tensor_get_storage_kind",
    )
    return out[]
end

function Tensor(data::Array{Float64,N}, inds::AbstractVector{Index}) where {N}
    length(inds) == N || throw(DimensionMismatch(
        "Tensor rank $N requires $N indices, got $(length(inds))",
    ))
    expected_dims = Tuple(dim.(inds))
    expected_dims == size(data) || throw(DimensionMismatch(
        "Tensor dimensions $expected_dims do not match data size $(size(data))",
    ))
    dims = collect(Csize_t, size(data))
    index_ptrs = Ptr{Cvoid}[i.ptr for i in inds]
    ptr = ccall(
        _capi_symbol(:t4a_tensor_new_dense_f64),
        Ptr{Cvoid},
        (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Csize_t),
        N,
        index_ptrs,
        dims,
        data,
        length(data),
    )
    return _adopt_tensor(ptr, "t4a_tensor_new_dense_f64")
end

function Tensor(data::Array{ComplexF64,N}, inds::AbstractVector{Index}) where {N}
    length(inds) == N || throw(DimensionMismatch(
        "Tensor rank $N requires $N indices, got $(length(inds))",
    ))
    expected_dims = Tuple(dim.(inds))
    expected_dims == size(data) || throw(DimensionMismatch(
        "Tensor dimensions $expected_dims do not match data size $(size(data))",
    ))
    dims = collect(Csize_t, size(data))
    index_ptrs = Ptr{Cvoid}[i.ptr for i in inds]
    raw = Vector{Float64}(undef, 2 * length(data))
    for n in eachindex(data)
        raw[2n - 1] = real(data[n])
        raw[2n] = imag(data[n])
    end
    ptr = ccall(
        _capi_symbol(:t4a_tensor_new_dense_c64),
        Ptr{Cvoid},
        (Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{Cdouble}, Csize_t),
        N,
        index_ptrs,
        dims,
        raw,
        length(data),
    )
    return _adopt_tensor(ptr, "t4a_tensor_new_dense_c64")
end

function Tensor(data::AbstractArray, inds::AbstractVector{Index})
    throw(ArgumentError(
        "Array must be contiguous in memory for C API. Got $(typeof(data)) with strides $(strides(data)). Use collect() to make a contiguous copy.",
    ))
end

"""
    inds(t)

Return a copy of the index metadata attached to `t`.
"""
function inds(t::Tensor)
    out_rank = Ref{Csize_t}(0)
    _check_status(
        ccall(
            _capi_symbol(:t4a_tensor_get_rank),
            _StatusCode,
            (Ptr{Cvoid}, Ref{Csize_t}),
            t.ptr,
            out_rank,
        ),
        "t4a_tensor_get_rank",
    )
    handles = Vector{Ptr{Cvoid}}(undef, out_rank[])
    _check_status(
        ccall(
            _capi_symbol(:t4a_tensor_get_indices),
            _StatusCode,
            (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t),
            t.ptr,
            handles,
            length(handles),
        ),
        "t4a_tensor_get_indices",
    )
    return [_adopt_index(handle, "t4a_tensor_get_indices") for handle in handles]
end

"""
    rank(t)

Return the tensor rank of `t`.
"""
function rank(t::Tensor)
    out_rank = Ref{Csize_t}(0)
    _check_status(
        ccall(
            _capi_symbol(:t4a_tensor_get_rank),
            _StatusCode,
            (Ptr{Cvoid}, Ref{Csize_t}),
            t.ptr,
            out_rank,
        ),
        "t4a_tensor_get_rank",
    )
    return Int(out_rank[])
end

"""
    dims(t)

Return the dense array dimensions of `t`.
"""
function dims(t::Tensor)
    n = rank(t)
    out_dims = Vector{Csize_t}(undef, n)
    _check_status(
        ccall(
            _capi_symbol(:t4a_tensor_get_dims),
            _StatusCode,
            (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t),
            t.ptr,
            out_dims,
            length(out_dims),
        ),
        "t4a_tensor_get_dims",
    )
    return Tuple(Int.(out_dims))
end

function _dense_array(t::Tensor)
    shape = dims(t)
    tensor_inds = inds(t)
    len = prod(shape)
    kind = _storage_kind(t)
    if kind == _T4A_STORAGE_DENSE_F64 || kind == _T4A_STORAGE_DIAG_F64
        out_len = Ref{Csize_t}(0)
        buf = Vector{Float64}(undef, len)
        _check_status(
            ccall(
                _capi_symbol(:t4a_tensor_get_data_f64),
                _StatusCode,
                (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ref{Csize_t}),
                t.ptr,
                buf,
                len,
                out_len,
            ),
            "t4a_tensor_get_data_f64",
        )
        return reshape(buf, shape...), tensor_inds
    end

    out_len = Ref{Csize_t}(0)
    raw = Vector{Float64}(undef, 2 * len)
    _check_status(
        ccall(
            _capi_symbol(:t4a_tensor_get_data_c64),
            _StatusCode,
            (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t, Ref{Csize_t}),
            t.ptr,
            raw,
            len,
            out_len,
        ),
        "t4a_tensor_get_data_c64",
    )
    data = ComplexF64[ComplexF64(raw[2n - 1], raw[2n]) for n in 1:len]
    return reshape(data, shape...), tensor_inds
end

"""
    prime(t, n=1)

Return `t` with all attached indices primed by `n`.
"""
function prime(t::Tensor, n::Integer=1)
    data, tensor_inds = _dense_array(t)
    return Tensor(copy(data), prime.(tensor_inds, Ref(n)))
end

"""
    swapinds(t, a, b)

Swap index metadata `a` and `b` on `t`.
"""
function swapinds(t::Tensor, a::Index, b::Index)
    data, tensor_inds = _dense_array(t)
    pa = findall(==(a), tensor_inds)
    pb = findall(==(b), tensor_inds)
    length(pa) == 1 || throw(ArgumentError("Index $(a) must appear exactly once"))
    length(pb) == 1 || throw(ArgumentError("Index $(b) must appear exactly once"))
    perm = collect(1:length(tensor_inds))
    perm[pa[1]], perm[pb[1]] = perm[pb[1]], perm[pa[1]]
    return Tensor(permutedims(data, Tuple(perm)), tensor_inds[perm])
end

"""
    contract(a, b)

Contract two tensors along their shared indices.
"""
function contract(a::Tensor, b::Tensor)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check_status(
        ccall(
            _capi_symbol(:t4a_tensor_contract),
            _StatusCode,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
            a.ptr,
            b.ptr,
            out,
        ),
        "t4a_tensor_contract",
    )
    return _adopt_tensor(out[], "t4a_tensor_contract")
end
