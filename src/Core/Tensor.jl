import Base: +, -, *, /
import LinearAlgebra: norm

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
    backend_handle::Union{Nothing,Ptr{Cvoid}}
end

function Tensor(
    data::Array{T,N},
    inds::AbstractVector{Index};
    backend_handle::Union{Nothing,Ptr{Cvoid}}=nothing,
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
    prime(t, n=1)

Return `t` with all attached indices primed by `n`.
"""
function prime(t::Tensor, n::Integer=1)
    return Tensor(copy(t.data), prime.(inds(t), Ref(n)); backend_handle=t.backend_handle)
end

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
