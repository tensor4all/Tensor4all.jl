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
    contract(a, b)

Placeholder for tensor contraction.

# Examples
```julia
julia> contract(a, b)
ERROR: SkeletonNotImplemented(...)
```
"""
contract(::Tensor, ::Tensor) = throw(SkeletonNotImplemented(:contract, :core))
