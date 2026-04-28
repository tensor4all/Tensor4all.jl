import Base: +, -, *, /
import LinearAlgebra: norm, qr, svd

struct StructuredTensorStorage{T}
    kind::Symbol
    payload::Vector{T}
    payload_dims::Vector{Int}
    payload_strides::Vector{Int}
    axis_classes::Vector{Int}
end

mutable struct TensorHandle
    ptr::Ptr{Cvoid}
    function TensorHandle(ptr::Ptr{Cvoid}; owned::Bool=true)
        ptr == C_NULL && throw(ArgumentError("TensorHandle cannot wrap C_NULL"))
        handle = new(ptr)
        owned && finalizer(_release_owned_tensor_handle, handle)
        return handle
    end
end

function _release_owned_tensor_handle(handle::TensorHandle)
    ptr = handle.ptr
    ptr == C_NULL && return nothing
    handle.ptr = C_NULL
    try
        ccall(Libdl.dlsym(require_backend(), :t4a_tensor_release), Cvoid, (Ptr{Cvoid},), ptr)
    catch
        return nothing
    end
    return nothing
end

_backend_handle_ptr(handle::Nothing) = C_NULL
_backend_handle_ptr(handle::Ptr{Cvoid}) = handle
_backend_handle_ptr(handle::TensorHandle) = handle.ptr

const _CORE_T4A_SUCCESS = Cint(0)
const _CORE_T4A_SCALAR_KIND_F64 = Cint(0)
const _CORE_T4A_SCALAR_KIND_C64 = Cint(1)

"""
    Tensor(data, inds; backend_handle=nothing)

Create a tensor from Julia-owned dense array data and index metadata.

This constructor validates metadata and shape consistency. Tensors keep a dense
Julia snapshot when constructed from arrays. Backend operations may return
handle-backed tensors that materialize dense Julia data lazily for indexing and
`Array` extraction; selected constructors may also attach compact storage
metadata used by backend calls.

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
mutable struct Tensor{T,N}
    data::Union{Nothing,Array{T,N}}
    inds::Vector{Index}
    backend_handle::Union{Nothing,Ptr{Cvoid},TensorHandle}
    structured_storage::Union{Nothing,StructuredTensorStorage{T}}
end

"""
    ITensor

Compatibility alias for [`Tensor`](@ref). Constructor calls such as
`ITensor(data, i, j)` use Tensor4all's Julia-owned tensor object model.
"""
const ITensor = Tensor

function Tensor(
    data::Array{T,N},
    inds::AbstractVector{Index};
    backend_handle::Union{Nothing,Ptr{Cvoid},TensorHandle}=nothing,
    structured_storage::Union{Nothing,StructuredTensorStorage{T}}=nothing,
) where {T,N}
    length(inds) == N || throw(DimensionMismatch(
        "Tensor rank $N requires $N indices, got $(length(inds))",
    ))
    expected_dims = Tuple(dim.(inds))
    expected_dims == size(data) || throw(DimensionMismatch(
        "Tensor dimensions $expected_dims do not match data size $(size(data))",
    ))
    _validate_structured_storage(structured_storage, expected_dims)
    return Tensor{T,N}(copy(data), collect(inds), backend_handle, structured_storage)
end

function _tensor_from_backend_handle(
    handle::TensorHandle,
    indices::AbstractVector{Index},
    ::Type{T},
    ::Val{N};
    structured_storage::Union{Nothing,StructuredTensorStorage{T}}=nothing,
) where {T,N}
    length(indices) == N || throw(DimensionMismatch(
        "Tensor rank $N requires $N indices, got $(length(indices))",
    ))
    return Tensor{T,N}(nothing, collect(indices), handle, structured_storage)
end

function Tensor(
    data::AbstractArray,
    inds::AbstractVector{Index};
    backend_handle=nothing,
    structured_storage=nothing,
)
    throw(ArgumentError(
        "Array must be contiguous in memory for C API. Got $(typeof(data)). Use collect(data) to make a contiguous copy.",
    ))
end

Tensor(data::Array, inds::Index...) = Tensor(data, collect(inds))
Tensor(value::Number) = Tensor(fill(value), Index[])

"""
    Tensor(value, inds...)
    ITensor(value, inds...)

Create a dense tensor filled with scalar `value` over the supplied indices.
This is an ITensors-compatible convenience constructor; `ITensor` is an alias
of `Tensor`, so both spellings share the same implementation.
"""
function Tensor(value::Number, first_index::Index, rest_indices::Index...)
    indices = Index[first_index, rest_indices...]
    return Tensor(fill(value, dim.(indices)...), indices)
end

Base.eltype(::Tensor{T}) where {T} = T

function _read_core_csize_vector_from_handle(ptr::Ptr{Cvoid}, symbol::Symbol, context::AbstractString)
    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(require_backend(), symbol),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    status == _CORE_T4A_SUCCESS || throw(ArgumentError("querying backend tensor $context failed"))

    values = Vector{Csize_t}(undef, Int(out_len[]))
    status = ccall(
        Libdl.dlsym(require_backend(), symbol),
        Cint,
        (Ptr{Cvoid}, Ptr{Csize_t}, Csize_t, Ref{Csize_t}),
        ptr,
        values,
        length(values),
        out_len,
    )
    status == _CORE_T4A_SUCCESS || throw(ArgumentError("copying backend tensor $context failed"))
    return Int.(values)
end

function _tensor_dims_from_owned_handle(handle)
    ptr = _backend_handle_ptr(handle)
    ptr == C_NULL && throw(ArgumentError("Tensor has no backend handle"))
    return Tuple(_read_core_csize_vector_from_handle(ptr, :t4a_tensor_dims, "dimensions"))
end

function _tensor_scalar_kind_from_owned_handle(handle)
    ptr = _backend_handle_ptr(handle)
    ptr == C_NULL && throw(ArgumentError("Tensor has no backend handle"))
    out_kind = Ref{Cint}(0)
    status = ccall(
        Libdl.dlsym(require_backend(), :t4a_tensor_scalar_kind),
        Cint,
        (Ptr{Cvoid}, Ref{Cint}),
        ptr,
        out_kind,
    )
    status == _CORE_T4A_SUCCESS || throw(ArgumentError("querying backend tensor scalar kind failed"))
    return out_kind[]
end

function _read_dense_f64_from_owned_handle(handle)
    ptr = _backend_handle_ptr(handle)
    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(require_backend(), :t4a_tensor_copy_dense_f64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    status == _CORE_T4A_SUCCESS || throw(ArgumentError("querying backend dense tensor data failed"))

    dense = Vector{Float64}(undef, Int(out_len[]))
    status = ccall(
        Libdl.dlsym(require_backend(), :t4a_tensor_copy_dense_f64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        dense,
        length(dense),
        out_len,
    )
    status == _CORE_T4A_SUCCESS || throw(ArgumentError("copying backend dense tensor data failed"))
    return dense
end

function _read_dense_c64_from_owned_handle(handle)
    ptr = _backend_handle_ptr(handle)
    out_len = Ref{Csize_t}(0)
    status = ccall(
        Libdl.dlsym(require_backend(), :t4a_tensor_copy_dense_c64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        C_NULL,
        0,
        out_len,
    )
    status == _CORE_T4A_SUCCESS || throw(ArgumentError("querying backend dense tensor data failed"))

    raw = Vector{Float64}(undef, 2 * Int(out_len[]))
    status = ccall(
        Libdl.dlsym(require_backend(), :t4a_tensor_copy_dense_c64),
        Cint,
        (Ptr{Cvoid}, Ptr{Float64}, Csize_t, Ref{Csize_t}),
        ptr,
        raw,
        Int(out_len[]),
        out_len,
    )
    status == _CORE_T4A_SUCCESS || throw(ArgumentError("copying backend dense tensor data failed"))

    dense = Vector{ComplexF64}(undef, Int(out_len[]))
    for n in eachindex(dense)
        dense[n] = ComplexF64(raw[2n-1], raw[2n])
    end
    return dense
end

function _materialize_tensor_data!(t::Tensor{T,N}) where {T,N}
    data = getfield(t, :data)
    data !== nothing && return data
    handle = getfield(t, :backend_handle)
    handle === nothing && throw(ArgumentError("Tensor has neither dense data nor backend handle"))
    scalar_kind = _tensor_scalar_kind_from_owned_handle(handle)
    dense = if scalar_kind == _CORE_T4A_SCALAR_KIND_F64
        _read_dense_f64_from_owned_handle(handle)
    elseif scalar_kind == _CORE_T4A_SCALAR_KIND_C64
        _read_dense_c64_from_owned_handle(handle)
    else
        throw(ArgumentError("Unsupported backend tensor scalar kind $scalar_kind"))
    end
    materialized = convert(Array{T,N}, reshape(dense, dims(t)))
    setfield!(t, :data, materialized)
    return materialized
end

function Base.getproperty(t::Tensor, name::Symbol)
    name === :data && return _materialize_tensor_data!(t)
    return getfield(t, name)
end

function _validate_structured_storage(
    storage::Union{Nothing,StructuredTensorStorage},
    logical_dims::Tuple,
)
    storage === nothing && return nothing
    storage.kind in (:diagonal, :structured) || throw(
        ArgumentError("structured storage kind must be :diagonal or :structured, got $(storage.kind)"),
    )
    all(>(0), storage.payload_dims) || throw(
        ArgumentError("payload dimensions must be positive, got $(storage.payload_dims)"),
    )
    length(storage.payload) == prod(storage.payload_dims; init=1) || throw(DimensionMismatch(
        "payload length $(length(storage.payload)) does not match payload dimensions $(storage.payload_dims)",
    ))
    length(storage.payload_dims) == length(storage.payload_strides) || throw(DimensionMismatch(
        "payload_dims length $(length(storage.payload_dims)) does not match payload_strides length $(length(storage.payload_strides))",
    ))
    length(storage.axis_classes) == length(logical_dims) || throw(DimensionMismatch(
        "axis_classes length $(length(storage.axis_classes)) does not match tensor rank $(length(logical_dims))",
    ))
    if !isempty(storage.axis_classes)
        minimum(storage.axis_classes) >= 1 || throw(
            ArgumentError("axis_classes use 1-based payload axes, got $(storage.axis_classes)"),
        )
        maximum(storage.axis_classes) <= length(storage.payload_dims) || throw(
            ArgumentError(
                "axis_classes $(storage.axis_classes) refer to payload rank $(length(storage.payload_dims))",
            ),
        )
    end
    for (axis, payload_axis) in enumerate(storage.axis_classes)
        expected_dim = storage.payload_dims[payload_axis]
        actual_dim = logical_dims[axis]
        actual_dim == expected_dim || throw(DimensionMismatch(
            "logical axis $axis has dimension $actual_dim but payload axis $payload_axis has dimension $expected_dim",
        ))
    end
    if storage.kind === :diagonal
        length(storage.payload_dims) == 1 || throw(DimensionMismatch(
            "diagonal storage requires payload rank 1, got $(length(storage.payload_dims))",
        ))
        all(==(1), storage.axis_classes) || throw(
            ArgumentError("diagonal storage axis_classes must all be 1, got $(storage.axis_classes)"),
        )
    end
    return nothing
end

"""
    inds(t)

Return a copy of the index metadata attached to `t`.
"""
inds(t::Tensor) = copy(t.inds)

"""
    rank(t)

Return the tensor rank of `t`.
"""
rank(t::Tensor) = length(t.inds)

"""
    dims(t)

Return the logical dimensions of `t`.
"""
dims(t::Tensor) = getfield(t, :data) === nothing ? _tensor_dims_from_owned_handle(getfield(t, :backend_handle)) : size(getfield(t, :data))

"""
    prime(t, n=1)

Return `t` with all attached indices primed by `n`.
"""
function prime(t::Tensor, n::Integer=1)
    return Tensor(
        copy(t.data),
        prime.(inds(t), Ref(n));
        backend_handle=t.backend_handle,
        structured_storage=_copy_structured_storage(t.structured_storage),
    )
end

"""
    dag(t)

Return the elementwise complex-conjugated tensor with the same index metadata.
"""
dag(t::Tensor) = Tensor(
    conj(t.data),
    inds(t);
    backend_handle=nothing,
    structured_storage=_map_structured_payload(conj, t.structured_storage),
)

"""
    swapinds(t, a, b)

Swap index metadata `a` and `b` on `t`.
"""
function swapinds(t::Tensor, a::Index, b::Index)
    swapped = map(inds(t)) do idx
        idx == a ? b : idx == b ? a : idx
    end
    return Tensor(
        copy(t.data),
        swapped;
        backend_handle=t.backend_handle,
        structured_storage=_copy_structured_storage(t.structured_storage),
    )
end

"""
    replaceind(t, old, new)

Return a copy of `t` with index metadata `old` replaced by `new`.
"""
function replaceind(t::Tensor, old::Index, new::Index)
    return Tensor(
        t.data,
        replaceind(inds(t), old, new);
        backend_handle=t.backend_handle,
        structured_storage=_copy_structured_storage(t.structured_storage),
    )
end

replaceind(t::Tensor, replacement::Pair{Index,Index}) = replaceind(
    t,
    first(replacement),
    last(replacement),
)

"""
    replaceinds(t, replacements...)

Return a copy of `t` with multiple index metadata replacements applied.
"""
function replaceinds(t::Tensor, replacements::Pair{Index,Index}...)
    return Tensor(
        t.data,
        replaceinds(inds(t), replacements...);
        backend_handle=t.backend_handle,
        structured_storage=_copy_structured_storage(t.structured_storage),
    )
end

function replaceinds(
    t::Tensor,
    oldinds::AbstractVector{Index},
    newinds::AbstractVector{Index},
)
    return Tensor(
        t.data,
        replaceinds(inds(t), oldinds, newinds);
        backend_handle=t.backend_handle,
        structured_storage=_copy_structured_storage(t.structured_storage),
    )
end

function replaceinds(
    t::Tensor,
    oldinds::Tuple{Vararg{Index}},
    newinds::Tuple{Vararg{Index}},
)
    return Tensor(
        t.data,
        replaceinds(inds(t), oldinds, newinds);
        backend_handle=t.backend_handle,
        structured_storage=_copy_structured_storage(t.structured_storage),
    )
end

"""
    replaceind!(t, old, new)

Replace index metadata `old` by `new` in `t`.
"""
function replaceind!(t::Tensor, old::Index, new::Index)
    t.inds .= replaceind(t.inds, old, new)
    return t
end

replaceind!(t::Tensor, replacement::Pair{Index,Index}) = replaceind!(
    t,
    first(replacement),
    last(replacement),
)

"""
    replaceinds!(t, replacements...)

Apply multiple index metadata replacements in place to `t`.
"""
function replaceinds!(t::Tensor, replacements::Pair{Index,Index}...)
    t.inds .= replaceinds(t.inds, replacements...)
    return t
end

function replaceinds!(
    t::Tensor,
    oldinds::AbstractVector{Index},
    newinds::AbstractVector{Index},
)
    t.inds .= replaceinds(t.inds, oldinds, newinds)
    return t
end

function replaceinds!(
    t::Tensor,
    oldinds::Tuple{Vararg{Index}},
    newinds::Tuple{Vararg{Index}},
)
    t.inds .= replaceinds(t.inds, oldinds, newinds)
    return t
end

"""
    _match_index_permutation(source_inds, target_inds)

Return the permutation that reorders `source_inds` to match `target_inds`.
Throws `ArgumentError` if the index sets do not match.
"""
function _match_index_permutation(source_inds::Vector{Index}, target_inds::Vector{Index})
    length(source_inds) == length(target_inds) || throw(
        DimensionMismatch("Tensor ranks differ: $(length(source_inds)) vs $(length(target_inds))"),
    )

    perm = Int[]
    for target_idx in target_inds
        pos = findfirst(==(target_idx), source_inds)
        pos === nothing && throw(ArgumentError(
            "Index $target_idx not found in source indices $source_inds",
        ))
        push!(perm, pos)
    end

    length(Set(perm)) == length(perm) || throw(ArgumentError(
        "Duplicate index match: source=$source_inds target=$target_inds",
    ))
    return Tuple(perm)
end

function _permute_to_match(a::Tensor, b::Tensor)
    perm = _match_index_permutation(inds(b), inds(a))
    return perm == Tuple(1:rank(a)) ? b.data : permutedims(b.data, perm)
end

function Base.:+(a::Tensor, b::Tensor)
    b_data = _permute_to_match(a, b)
    return Tensor(a.data .+ b_data, inds(a); backend_handle=nothing)
end

function Base.:-(a::Tensor, b::Tensor)
    b_data = _permute_to_match(a, b)
    return Tensor(a.data .- b_data, inds(a); backend_handle=nothing)
end

Base.:-(t::Tensor) = Tensor(
    -t.data,
    inds(t);
    backend_handle=nothing,
    structured_storage=_map_structured_payload(-, t.structured_storage),
)

function Base.:*(α::Number, t::Tensor)
    return Tensor(
        α .* t.data,
        inds(t);
        backend_handle=nothing,
        structured_storage=_map_structured_payload(x -> α * x, t.structured_storage),
    )
end

Base.:*(t::Tensor, α::Number) = α * t
Base.:*(a::Tensor, b::Tensor) = contract(a, b)

function Base.:/(t::Tensor, α::Number)
    return Tensor(
        t.data ./ α,
        inds(t);
        backend_handle=nothing,
        structured_storage=_map_structured_payload(x -> x / α, t.structured_storage),
    )
end

norm(t::Tensor) = norm(t.data)

function Base.isapprox(
    a::Tensor,
    b::Tensor;
    atol::Real=0,
    rtol::Real=Base.rtoldefault(eltype(a.data), eltype(b.data), atol),
)
    b_data = _permute_to_match(a, b)
    return isapprox(a.data, b_data; atol=atol, rtol=rtol)
end

function Base.Array(t::Tensor, requested_inds::Index...)
    isempty(requested_inds) && rank(t) == 0 && return copy(t.data)
    perm = _match_index_permutation(inds(t), collect(requested_inds))
    return perm == Tuple(1:rank(t)) ? copy(t.data) : permutedims(t.data, perm)
end

function _copy_structured_storage(storage::Nothing)
    return nothing
end

function _copy_structured_storage(storage::StructuredTensorStorage{T}) where {T}
    return StructuredTensorStorage{T}(
        storage.kind,
        copy(storage.payload),
        copy(storage.payload_dims),
        copy(storage.payload_strides),
        copy(storage.axis_classes),
    )
end

function _map_structured_payload(f::Function, storage::Nothing)
    return nothing
end

function _map_structured_payload(f::Function, storage::StructuredTensorStorage)
    payload = f.(storage.payload)
    return StructuredTensorStorage{eltype(payload)}(
        storage.kind,
        payload,
        copy(storage.payload_dims),
        copy(storage.payload_strides),
        copy(storage.axis_classes),
    )
end

"""
    delta(i, j, inds...; T=Float64)

Create a diagonal tensor with payload `ones(T, dim(i))` over equal-dimension
indices. The returned tensor keeps dense `Array` extraction compatible with the
rest of the Julia object model, while backend calls use compact diagonal
payload metadata.
"""
function delta(first_index::Index, second_index::Index, rest::Index...; T::Type=Float64)
    indices = Index[first_index, second_index, rest...]
    diagonal_dim = dim(first_index)
    index_dims = dim.(indices)
    all(==(diagonal_dim), index_dims) || throw(DimensionMismatch(
        "delta indices must have equal dimensions, got $index_dims",
    ))

    payload = ones(T, diagonal_dim)
    data = zeros(T, Tuple(index_dims))
    for n in 1:diagonal_dim
        data[ntuple(Returns(n), length(indices))...] = one(T)
    end
    storage = StructuredTensorStorage{T}(
        :diagonal,
        payload,
        [diagonal_dim],
        [1],
        ones(Int, length(indices)),
    )
    return Tensor(data, indices; structured_storage=storage)
end

"""
    isdiag(t)

Return `true` when `t` is backed by compact diagonal storage metadata.
"""
isdiag(t::Tensor) = t.structured_storage !== nothing && t.structured_storage.kind === :diagonal

"""
    structured_storage_info(t)

Return storage metadata for `t` as a named tuple. `axis_classes` are reported
with Julia's 1-based axis numbering.
"""
function structured_storage_info(t::Tensor)
    storage = t.structured_storage
    if storage === nothing
        return (;
            kind=:dense,
            dtype=eltype(t.data),
            logical_dims=dims(t),
            payload_dims=dims(t),
            payload_strides=Tuple(strides(t.data)),
            payload_length=length(t.data),
            axis_classes=Tuple(1:rank(t)),
        )
    end
    return (;
        kind=storage.kind,
        dtype=eltype(t.data),
        logical_dims=dims(t),
        payload_dims=Tuple(storage.payload_dims),
        payload_strides=Tuple(storage.payload_strides),
        payload_length=length(storage.payload),
        axis_classes=Tuple(storage.axis_classes),
    )
end

"""
    structured_payload(t)

Return a copy of the compact storage payload for `t`. Dense tensors return
their column-major dense payload.
"""
function structured_payload(t::Tensor)
    storage = t.structured_storage
    storage === nothing && return vec(copy(t.data))
    return copy(storage.payload)
end

commoninds(a::Tensor, b::Tensor) = commoninds(inds(a), inds(b))
uniqueinds(a::Tensor, b::Tensor) = uniqueinds(inds(a), inds(b))

"""
    hasinds(t, inds...)

Return `true` when all queried indices are attached to `t`.
"""
hasinds(t::Tensor, query::Index...) = all(index -> index in t.inds, query)

"""
    scalar(t)

Return the scalar value stored in a rank-0 tensor.
"""
function scalar(t::Tensor)
    rank(t) == 0 || throw(ArgumentError("scalar requires a rank-0 Tensor, got rank $(rank(t))"))
    return only(t.data)
end

struct OneHotTensor{T}
    index::Index
    value::Int
end

"""
    onehot(index => value; T=Float64)

Create a lazy one-hot tensor over `index` at the 1-based position `value`.
Contractions with tensors containing `index` slice the tensor instead of
materializing a dense basis vector.
"""
function onehot(index_value::Pair{Index,<:Integer}; T::Type=Float64)
    index = first(index_value)
    value = Int(last(index_value))
    1 <= value <= dim(index) || throw(ArgumentError(
        "onehot index value must be in 1:$(dim(index)), got $value",
    ))
    return OneHotTensor{T}(index, value)
end

inds(oh::OneHotTensor) = [oh.index]
rank(::OneHotTensor) = 1
dims(oh::OneHotTensor) = (dim(oh.index),)
Base.eltype(::OneHotTensor{T}) where {T} = T

function Tensor(oh::OneHotTensor{T}) where {T}
    data = zeros(T, dim(oh.index))
    data[oh.value] = one(T)
    return Tensor(data, [oh.index])
end

"""
    selectinds(t, selections...)
    selectinds(t, indices, values)

Fix one or more indices of `t` at 1-based coordinates and return the remaining
tensor. The selection is performed by the backend and materializes dense Julia
data only when the result is inspected as an array.
"""
function selectinds(
    t::Tensor,
    selected_indices::AbstractVector{<:Index},
    values::AbstractVector{<:Integer},
)
    length(selected_indices) == length(values) || throw(DimensionMismatch(
        "selectinds expected the same number of indices and values, got $(length(selected_indices)) and $(length(values))",
    ))
    isempty(selected_indices) && return t
    length(unique(selected_indices)) == length(selected_indices) || throw(
        ArgumentError("selectinds selected indices must not contain duplicates, got $selected_indices"),
    )

    tensor_indices = inds(t)
    for (index, value) in zip(selected_indices, values)
        axis = findfirst(==(index), tensor_indices)
        axis === nothing && throw(ArgumentError("selectinds index $index is not attached to tensor indices $tensor_indices"))
        1 <= value <= dim(index) || throw(ArgumentError(
            "selectinds value for index $index must be in 1:$(dim(index)), got $value",
        ))
    end

    scalar_kind = _tensor_scalar_kind(t)
    tn = _tensor_networks_module()
    t_handle = C_NULL
    selected_handles = Ptr{Cvoid}[]
    result_handle = C_NULL
    try
        t_handle = tn._new_tensor_handle(t, scalar_kind)
        for index in selected_indices
            push!(selected_handles, tn._new_index_handle(index))
        end
        positions = Csize_t.(values .- 1)

        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            tn._t4a(:t4a_tensor_select_indices),
            Cint,
            (Ptr{Cvoid}, Csize_t, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ref{Ptr{Cvoid}}),
            t_handle,
            Csize_t(length(selected_handles)),
            selected_handles,
            positions,
            out,
        )
        tn._check_backend_status(status, "selecting tensor indices")
        result_handle = out[]
        result = tn._lazy_tensor_from_owned_handle(result_handle)
        result_handle = C_NULL
        return result
    finally
        tn._release_tensor_handle(result_handle)
        for handle in reverse(selected_handles)
            tn._release_index_handle(handle)
        end
        tn._release_tensor_handle(t_handle)
    end
end

function selectinds(t::Tensor, selections::Pair{Index,<:Integer}...)
    return selectinds(t, Index[first(selection) for selection in selections], Int[last(selection) for selection in selections])
end

function _contract_tensor_onehot(t::Tensor, oh::OneHotTensor)
    axis = findfirst(==(oh.index), t.inds)
    axis === nothing && return contract(t, Tensor(oh))
    return selectinds(t, oh.index => oh.value)
end

contract(t::Tensor, oh::OneHotTensor) = _contract_tensor_onehot(t, oh)
contract(oh::OneHotTensor, t::Tensor) = _contract_tensor_onehot(t, oh)
Base.:*(t::Tensor, oh::OneHotTensor) = contract(t, oh)
Base.:*(oh::OneHotTensor, t::Tensor) = contract(oh, t)

function _tensor_scalar_kind(tensors::Tensor...)
    any_complex = false
    for tensor in tensors
        T = eltype(tensor)
        if T <: Real
            continue
        elseif T <: Complex
            any_complex = true
        else
            throw(ArgumentError("backend tensor operations support only real or complex tensors, got eltype $T"))
        end
    end
    return any_complex ? :c64 : :f64
end

_tensor_networks_module() = getfield(@__MODULE__, :TensorNetworks)

function _validate_tensor_left_inds(t::Tensor, left_inds::Vector{Index})
    isempty(left_inds) && throw(ArgumentError("left_inds must not be empty"))
    length(unique(left_inds)) == length(left_inds) || throw(
        ArgumentError("left_inds must not contain duplicates, got $left_inds"),
    )

    tensor_inds = inds(t)
    for idx in left_inds
        idx in tensor_inds || throw(
            ArgumentError("Index $idx not found in tensor indices $tensor_inds"),
        )
    end
    length(left_inds) == rank(t) && throw(ArgumentError("left_inds must not contain all indices"))
    return tensor_inds
end

function _validate_truncation_controls(; threshold::Real=0.0, maxdim::Integer=0)
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative, got $threshold"))
    maxdim >= 0 || throw(ArgumentError("maxdim must be nonnegative, got $maxdim"))
    return nothing
end

"""
    contract(a, b)

Contract two tensors over shared indices using the Rust backend.

Shared indices (matching by identity) are summed over. The result tensor
has the remaining (uncontracted) indices.
"""
function contract(a::Tensor, b::Tensor)
    scalar_kind = _tensor_scalar_kind(a, b)
    tn = _tensor_networks_module()
    a_handle = C_NULL
    b_handle = C_NULL
    result_handle = C_NULL

    try
        a_handle = tn._new_tensor_handle(a, scalar_kind)
        b_handle = tn._new_tensor_handle(b, scalar_kind)

        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            tn._t4a(:t4a_tensor_contract),
            Cint,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
            a_handle,
            b_handle,
            out,
        )
        tn._check_backend_status(status, "contracting tensors")
        result_handle = out[]
        result = tn._lazy_tensor_from_owned_handle(result_handle)
        result_handle = C_NULL
        return result
    finally
        tn._release_tensor_handle(result_handle)
        tn._release_tensor_handle(b_handle)
        tn._release_tensor_handle(a_handle)
    end
end

"""
    svd(t, left_inds; threshold=0.0, maxdim=0, svd_policy=nothing)

Compute a backend SVD of `t`, grouping `left_inds` as the left partition.

Truncation: `threshold` is the numeric amount; `svd_policy` chooses the
strategy (falls back to `TensorNetworks.default_svd_policy()` when `nothing`).
`threshold == 0` disables SVD-based truncation; `maxdim` caps the retained
rank independently.
"""
function svd(
    t::Tensor,
    left_inds::Vector{Index};
    threshold::Real=0.0,
    maxdim::Integer=0,
    svd_policy=nothing,
)
    _validate_tensor_left_inds(t, left_inds)
    _validate_truncation_controls(; threshold, maxdim)

    scalar_kind = _tensor_scalar_kind(t)
    tn = _tensor_networks_module()
    t_handle = C_NULL
    left_handles = Ptr{Cvoid}[]
    u_handle = C_NULL
    s_handle = C_NULL
    v_handle = C_NULL

    ffi_policy = tn._resolve_svd_policy(; threshold, svd_policy)

    try
        t_handle = tn._new_tensor_handle(t, scalar_kind)
        for idx in left_inds
            push!(left_handles, tn._new_index_handle(idx))
        end

        out_u = Ref{Ptr{Cvoid}}(C_NULL)
        out_s = Ref{Ptr{Cvoid}}(C_NULL)
        out_v = Ref{Ptr{Cvoid}}(C_NULL)
        status = tn._with_svd_policy_ptr(ffi_policy) do policy_ptr
            ccall(
                tn._t4a(:t4a_tensor_svd),
                Cint,
                (
                    Ptr{Cvoid},
                    Ptr{Ptr{Cvoid}},
                    Csize_t,
                    Ptr{Cvoid},
                    Csize_t,
                    Ref{Ptr{Cvoid}},
                    Ref{Ptr{Cvoid}},
                    Ref{Ptr{Cvoid}},
                ),
                t_handle,
                left_handles,
                Csize_t(length(left_handles)),
                policy_ptr,
                Csize_t(maxdim),
                out_u,
                out_s,
                out_v,
            )
        end
        tn._check_backend_status(status, "computing tensor SVD")
        u_handle = out_u[]
        s_handle = out_s[]
        v_handle = out_v[]
        u = tn._tensor_from_handle(u_handle)
        s = tn._tensor_from_handle(s_handle)
        v = tn._tensor_from_handle(v_handle)

        # Align V's surviving bond index with S so U * S * dag(V) reconstructs.
        v_inds = inds(v)
        s_inds = inds(s)
        v = Tensor(
            copy(v.data),
            replaceind(v_inds, last(v_inds), last(s_inds));
            backend_handle=v.backend_handle,
        )
        return (u, s, v)
    finally
        tn._release_tensor_handle(v_handle)
        tn._release_tensor_handle(s_handle)
        tn._release_tensor_handle(u_handle)
        for handle in reverse(left_handles)
            tn._release_index_handle(handle)
        end
        tn._release_tensor_handle(t_handle)
    end
end

svd(t::Tensor, left_inds::Index...; kwargs...) = svd(t, collect(left_inds); kwargs...)

"""
    qr(t, left_inds)

Compute a backend QR decomposition of `t`, grouping `left_inds` as the left
partition.
"""
function qr(t::Tensor, left_inds::Vector{Index})
    _validate_tensor_left_inds(t, left_inds)

    scalar_kind = _tensor_scalar_kind(t)
    tn = _tensor_networks_module()
    t_handle = C_NULL
    left_handles = Ptr{Cvoid}[]
    q_handle = C_NULL
    r_handle = C_NULL

    try
        t_handle = tn._new_tensor_handle(t, scalar_kind)
        for idx in left_inds
            push!(left_handles, tn._new_index_handle(idx))
        end

        out_q = Ref{Ptr{Cvoid}}(C_NULL)
        out_r = Ref{Ptr{Cvoid}}(C_NULL)
        status = ccall(
            tn._t4a(:t4a_tensor_qr),
            Cint,
            (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t, Ref{Ptr{Cvoid}}, Ref{Ptr{Cvoid}}),
            t_handle,
            left_handles,
            Csize_t(length(left_handles)),
            out_q,
            out_r,
        )
        tn._check_backend_status(status, "computing tensor QR")
        q_handle = out_q[]
        r_handle = out_r[]
        return (tn._tensor_from_handle(q_handle), tn._tensor_from_handle(r_handle))
    finally
        tn._release_tensor_handle(r_handle)
        tn._release_tensor_handle(q_handle)
        for handle in reverse(left_handles)
            tn._release_index_handle(handle)
        end
        tn._release_tensor_handle(t_handle)
    end
end

qr(t::Tensor, left_inds::Index...) = qr(t, collect(left_inds))
