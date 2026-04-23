import Base: +, -, *, /
import LinearAlgebra: norm, qr, svd

mutable struct BackendTensorHandle
    ptr::Ptr{Cvoid}
    owned::Bool
end

function _release_owned_backend_tensor_handle(handle::BackendTensorHandle)
    handle.owned || return nothing
    ptr = handle.ptr
    ptr == C_NULL && return nothing
    handle.ptr = C_NULL
    handle.owned = false
    _tensor_networks_module()._release_tensor_handle(ptr)
    return nothing
end

function BackendTensorHandle(ptr::Ptr{Cvoid}; owned::Bool=false)
    handle = BackendTensorHandle(ptr, owned)
    owned && finalizer(_release_owned_backend_tensor_handle, handle)
    return handle
end

const BackendTensorHandleLike = Union{Nothing,Ptr{Cvoid},BackendTensorHandle}

backend_handle_ptr(::Nothing) = C_NULL
backend_handle_ptr(ptr::Ptr{Cvoid}) = ptr
backend_handle_ptr(handle::BackendTensorHandle) = handle.ptr

function _owned_backend_tensor_handle(ptr::Ptr{Cvoid})
    ptr == C_NULL && throw(ArgumentError("Cannot wrap a null backend tensor handle"))
    return BackendTensorHandle(ptr; owned=true)
end

"""
    Tensor(data, inds; backend_handle=nothing)

Create a tensor skeleton from Julia-owned dense array data and index metadata.

This constructor validates metadata and shape consistency during the skeleton
phase. Backend-backed contraction and factorization behavior remains deferred.

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
struct Tensor{T,N}
    data::Array{T,N}
    inds::Vector{Index}
    backend_handle::BackendTensorHandleLike
end

function Tensor(
    data::Array{T,N},
    inds::AbstractVector{Index};
    backend_handle::BackendTensorHandleLike=nothing,
) where {T,N}
    length(inds) == N || throw(DimensionMismatch(
        "Tensor rank $N requires $N indices, got $(length(inds))",
    ))
    expected_dims = Tuple(dim.(inds))
    expected_dims == size(data) || throw(DimensionMismatch(
        "Tensor dimensions $expected_dims do not match data size $(size(data))",
    ))
    return Tensor{T,N}(copy(data), collect(inds), backend_handle)
end

backend_handle_ptr(t::Tensor) = backend_handle_ptr(t.backend_handle)

function Tensor(data::Array, inds::Index...; backend_handle::BackendTensorHandleLike=nothing)
    return Tensor(data, collect(inds); backend_handle)
end

function Tensor(data::AbstractArray, inds::Index...; backend_handle=nothing)
    return Tensor(data, collect(inds); backend_handle)
end

function Tensor(value::Number)
    data = Array{typeof(value),0}(undef)
    data[] = value
    return Tensor(data, Index[])
end

const ITensor = Tensor

Base.eltype(t::Tensor) = eltype(t.data)
Base.eltype(::Type{<:Tensor{T}}) where {T} = T

function Tensor(data::AbstractArray, inds::AbstractVector{Index}; backend_handle=nothing)
    throw(ArgumentError(
        "Array must be contiguous in memory for C API. Got $(typeof(data)). Use collect(data) to make a contiguous copy.",
    ))
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

Return the dense array dimensions of `t`.
"""
dims(t::Tensor) = size(t.data)

"""
    scalar(t)

Return the scalar value stored in a rank-0 tensor.
"""
function scalar(t::Tensor)
    rank(t) == 0 || throw(ArgumentError("scalar requires a rank-0 Tensor, got rank $(rank(t))"))
    return t.data[]
end

"""
    prime(t, n=1)

Return `t` with all attached indices primed by `n`.
"""
function prime(t::Tensor, n::Integer=1)
    return Tensor(copy(t.data), prime.(inds(t), Ref(n)); backend_handle=t.backend_handle)
end

"""
    dag(t)

Return the elementwise complex-conjugated tensor with the same index metadata.
"""
dag(t::Tensor) = Tensor(conj(t.data), inds(t); backend_handle=nothing)

"""
    swapinds(t, a, b)

Swap index metadata `a` and `b` on `t`.
"""
function swapinds(t::Tensor, a::Index, b::Index)
    swapped = map(inds(t)) do idx
        idx == a ? b : idx == b ? a : idx
    end
    return Tensor(copy(t.data), swapped; backend_handle=t.backend_handle)
end

"""
    replaceind(t, old, new)

Return a copy of `t` with index metadata `old` replaced by `new`.
"""
function replaceind(t::Tensor, old::Index, new::Index)
    return Tensor(t.data, replaceind(inds(t), old, new); backend_handle=t.backend_handle)
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
    return Tensor(t.data, replaceinds(inds(t), replacements...); backend_handle=t.backend_handle)
end

function replaceinds(
    t::Tensor,
    oldinds::AbstractVector{Index},
    newinds::AbstractVector{Index},
)
    return Tensor(t.data, replaceinds(inds(t), oldinds, newinds); backend_handle=t.backend_handle)
end

function replaceinds(
    t::Tensor,
    oldinds::Tuple{Vararg{Index}},
    newinds::Tuple{Vararg{Index}},
)
    return Tensor(t.data, replaceinds(inds(t), oldinds, newinds); backend_handle=t.backend_handle)
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

Base.:-(t::Tensor) = Tensor(-t.data, inds(t); backend_handle=nothing)

Base.:*(α::Number, t::Tensor) = Tensor(α .* t.data, inds(t); backend_handle=nothing)
Base.:*(t::Tensor, α::Number) = α * t
Base.:/(t::Tensor, α::Number) = Tensor(t.data ./ α, inds(t); backend_handle=nothing)

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
    perm = _match_index_permutation(inds(t), collect(requested_inds))
    return perm == Tuple(1:rank(t)) ? copy(t.data) : permutedims(t.data, perm)
end

function _tensor_scalar_kind(tensors::Tensor...)
    any_complex = false
    for tensor in tensors
        T = eltype(tensor.data)
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
        return tn._tensor_from_handle(result_handle)
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
