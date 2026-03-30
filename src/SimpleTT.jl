"""
    SimpleTT

High-level Julia wrappers for SimpleTT tensor trains from tensor4all-simplett.

SimpleTT is a simple tensor train (TT/MPS) library with statically determined shapes
(site dimensions are fixed at construction time).
"""
module SimpleTT

using LinearAlgebra
using ..C_API

export SimpleTensorTrain

const _SimpleTTScalar = Union{Float64, ComplexF64}

_suffix(::Type{Float64}) = "f64"
_suffix(::Type{ComplexF64}) = "c64"
_api(::Type{T}, name::Symbol) where {T<:_SimpleTTScalar} =
    getfield(C_API, Symbol("t4a_simplett_", _suffix(T), "_", name))

"""
    SimpleTensorTrain{T<:Union{Float64, ComplexF64}}

A simple tensor train (TT/MPS) with statically determined shapes.

SimpleTensorTrain is a simple tensor train library where site dimensions are fixed
at construction time.

Supports `Float64` and `ComplexF64` values.
"""
mutable struct SimpleTensorTrain{T<:_SimpleTTScalar}
    ptr::Ptr{Cvoid}

    function SimpleTensorTrain{T}(ptr::Ptr{Cvoid}) where {T<:_SimpleTTScalar}
        ptr == C_NULL && error("Failed to create SimpleTensorTrain: null pointer")
        tt = new{T}(ptr)
        finalizer(tt) do obj
            _api(T, :release)(obj.ptr)
        end
        return tt
    end
end

# Convenience constructor
SimpleTensorTrain(ptr::Ptr{Cvoid}) = SimpleTensorTrain{Float64}(ptr)

"""
    SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Real)

Create a constant tensor train with the given site dimensions and value.

# Example
```julia
tt = SimpleTensorTrain([2, 3, 4], 1.0)  # All elements equal to 1.0
```
"""
function SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Real)
    dims = Csize_t.(site_dims)
    ptr = _api(Float64, :constant)(dims, Float64(value))
    return SimpleTensorTrain{Float64}(ptr)
end

function SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Complex)
    dims = Csize_t.(site_dims)
    value_c64 = ComplexF64(value)
    ptr = _api(ComplexF64, :constant)(dims, real(value_c64), imag(value_c64))
    return SimpleTensorTrain{ComplexF64}(ptr)
end

"""
    zeros(::Type{SimpleTensorTrain}, site_dims::Vector{<:Integer})

Create a zero tensor train with the given site dimensions.

# Example
```julia
tt = zeros(SimpleTensorTrain, [2, 3, 4])
```
"""
function Base.zeros(::Type{SimpleTensorTrain}, site_dims::Vector{<:Integer})
    return zeros(SimpleTensorTrain{Float64}, site_dims)
end

function Base.zeros(::Type{SimpleTensorTrain{T}}, site_dims::Vector{<:Integer}) where {T<:_SimpleTTScalar}
    dims = Csize_t.(site_dims)
    ptr = _api(T, :zeros)(dims)
    return SimpleTensorTrain{T}(ptr)
end

"""
    copy(tt::SimpleTensorTrain)

Create a deep copy of the tensor train.
"""
function Base.copy(tt::SimpleTensorTrain{T}) where T
    new_ptr = _api(T, :clone)(tt.ptr)
    return SimpleTensorTrain{T}(new_ptr)
end

"""
    length(tt::SimpleTensorTrain) -> Int

Get the number of sites in the tensor train.
"""
function Base.length(tt::SimpleTensorTrain{T}) where T
    out_len = Ref{Csize_t}(0)
    status = _api(T, :len)(tt.ptr, out_len)
    C_API.check_status(status)
    return Int(out_len[])
end

"""
    site_dims(tt::SimpleTensorTrain) -> Vector{Int}

Get the site (physical) dimensions.
"""
function site_dims(tt::SimpleTensorTrain{T}) where T
    n = length(tt)
    dims = Vector{Csize_t}(undef, n)
    status = _api(T, :site_dims)(tt.ptr, dims)
    C_API.check_status(status)
    return Int.(dims)
end

"""
    link_dims(tt::SimpleTensorTrain) -> Vector{Int}

Get the link (bond) dimensions. Returns n-1 values for n sites.
"""
function link_dims(tt::SimpleTensorTrain{T}) where T
    n = length(tt)
    n <= 1 && return Int[]
    dims = Vector{Csize_t}(undef, n - 1)
    status = _api(T, :link_dims)(tt.ptr, dims)
    C_API.check_status(status)
    return Int.(dims)
end

"""
    rank(tt::SimpleTensorTrain) -> Int

Get the maximum bond dimension (rank).
"""
function rank(tt::SimpleTensorTrain{T}) where T
    out_rank = Ref{Csize_t}(0)
    status = _api(T, :rank)(tt.ptr, out_rank)
    C_API.check_status(status)
    return Int(out_rank[])
end

"""
    evaluate(tt::SimpleTensorTrain, indices::Vector{<:Integer}) -> T

Evaluate the tensor train at a given multi-index (0-based indexing).

# Example
```julia
tt = SimpleTensorTrain([2, 3, 4], 2.0)
val = evaluate(tt, [0, 1, 2])  # Returns 2.0
```
"""
function evaluate(tt::SimpleTensorTrain{T}, indices::Vector{<:Integer}) where T
    idx = Csize_t.(indices)
    if T === Float64
        out_value = Ref{Cdouble}(0.0)
        status = _api(T, :evaluate)(tt.ptr, idx, out_value)
        C_API.check_status(status)
        return out_value[]
    else
        out_value_re = Ref{Cdouble}(0.0)
        out_value_im = Ref{Cdouble}(0.0)
        status = _api(T, :evaluate)(tt.ptr, idx, out_value_re, out_value_im)
        C_API.check_status(status)
        return ComplexF64(out_value_re[], out_value_im[])
    end
end

# Callable interface
(tt::SimpleTensorTrain{T})(indices::Vector{<:Integer}) where T = evaluate(tt, indices)
(tt::SimpleTensorTrain{T})(indices::Integer...) where T = evaluate(tt, collect(indices))

"""
    sum(tt::SimpleTensorTrain) -> T

Compute the sum over all tensor train elements.
"""
function Base.sum(tt::SimpleTensorTrain{T}) where T
    if T === Float64
        out_value = Ref{Cdouble}(0.0)
        status = _api(T, :sum)(tt.ptr, out_value)
        C_API.check_status(status)
        return out_value[]
    else
        out_value_re = Ref{Cdouble}(0.0)
        out_value_im = Ref{Cdouble}(0.0)
        status = _api(T, :sum)(tt.ptr, out_value_re, out_value_im)
        C_API.check_status(status)
        return ComplexF64(out_value_re[], out_value_im[])
    end
end

"""
    norm(tt::SimpleTensorTrain) -> Float64

Compute the Frobenius norm of the tensor train.
"""
function LinearAlgebra.norm(tt::SimpleTensorTrain{T}) where T
    out_value = Ref{Cdouble}(0.0)
    status = _api(T, :norm)(tt.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    site_tensor(tt::SimpleTensorTrain, site::Integer) -> Array{T, 3}

Get the site tensor at a specific site (0-based indexing).
Returns array with shape (left_dim, site_dim, right_dim).
"""
function site_tensor(tt::SimpleTensorTrain{T}, site::Integer) where T
    # First get dimensions to allocate buffer
    n = length(tt)
    0 <= site < n || error("Site index out of bounds: $site (n=$n)")

    sdims = site_dims(tt)
    ldims = link_dims(tt)

    left_dim = site == 0 ? 1 : ldims[site]
    site_dim = sdims[site + 1]  # Julia 1-based
    right_dim = site == n - 1 ? 1 : ldims[site + 1]

    out_left = Ref{Csize_t}(0)
    out_site = Ref{Csize_t}(0)
    out_right = Ref{Csize_t}(0)

    total_size = left_dim * site_dim * right_dim

    if T === Float64
        data = Vector{Cdouble}(undef, total_size)
        status = _api(T, :site_tensor)(tt.ptr, site, data, out_left, out_site, out_right)
        C_API.check_status(status)
        return reshape(data, (Int(out_left[]), Int(out_site[]), Int(out_right[])))
    else
        data = Vector{Cdouble}(undef, 2 * total_size)
        status = _api(T, :site_tensor)(tt.ptr, site, data, out_left, out_site, out_right)
        C_API.check_status(status)
        result = reinterpret(ComplexF64, data)
        return reshape(result, (Int(out_left[]), Int(out_site[]), Int(out_right[])))
    end
end

"""
    show(io::IO, tt::SimpleTensorTrain)

Display tensor train information.
"""
function Base.show(io::IO, tt::SimpleTensorTrain{T}) where T
    n = length(tt)
    r = rank(tt)
    print(io, "SimpleTensorTrain{$T}(sites=$n, rank=$r)")
end

function Base.show(io::IO, ::MIME"text/plain", tt::SimpleTensorTrain{T}) where T
    n = length(tt)
    println(io, "SimpleTensorTrain{$T}")
    println(io, "  Sites: $n")
    println(io, "  Site dims: $(site_dims(tt))")
    println(io, "  Link dims: $(link_dims(tt))")
    println(io, "  Max rank: $(rank(tt))")
end

# ============================================================================
# Compression
# ============================================================================

"""
    compress!(tt::SimpleTensorTrain; method=:LU, tolerance=1e-12, max_bonddim=0)

Compress the tensor train in-place.

Methods: `:LU` (default), `:CI`, `:SVD`.
"""
function compress!(
    tt::SimpleTensorTrain{T};
    method::Symbol = :LU,
    tolerance::Float64 = 1e-12,
    max_bonddim::Int = 0,
) where T
    method_int = if method == :LU
        Cint(0)
    elseif method == :CI
        Cint(1)
    elseif method == :SVD
        Cint(2)
    else
        error("Unknown compression method: $method. Use :LU, :CI, or :SVD")
    end

    status = _api(T, :compress)(tt.ptr, method_int, tolerance, Csize_t(max_bonddim))
    C_API.check_status(status)
end

# ============================================================================
# Partial sum
# ============================================================================

"""
    partial_sum(tt::SimpleTensorTrain; dims) -> SimpleTensorTrain or scalar

Sum over selected dimensions (0-indexed). Returns a new SimpleTensorTrain
with the remaining dimensions. If all dimensions are summed, returns the
scalar value.
"""
function partial_sum(tt::SimpleTensorTrain{T}; dims) where T
    dim_vec = Csize_t.(collect(dims))
    out = Ref{Ptr{Cvoid}}(C_NULL)
    status = _api(T, :partial_sum)(tt.ptr, dim_vec, out)
    C_API.check_status(status)
    result = SimpleTensorTrain{T}(out[])

    # If all dims summed, return scalar
    if length(result) == 1 && length(dims) == length(tt)
        return evaluate(result, [0])
    end
    return result
end

end # module SimpleTT
