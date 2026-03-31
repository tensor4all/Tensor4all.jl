"""
    SimpleTT

High-level Julia wrappers for SimpleTT tensor trains from tensor4all-simplett.

SimpleTT is a simple tensor train (TT/MPS) library with statically determined shapes
(site dimensions are fixed at construction time).

Supports both `Float64` and `ComplexF64` element types.
"""
module SimpleTT

using LinearAlgebra
using ..C_API
import ..rank, ..linkdims, ..compress!, ..evaluate

export SimpleTensorTrain, sitedims, linkdims, rank, compress!, evaluate, sitetensor, fulltensor, scale!

# ============================================================================
# Type dispatch helpers
# ============================================================================

"""Scalar types supported by SimpleTensorTrain."""
const _SimpleTTScalar = Union{Float64, ComplexF64}

_suffix(::Type{Float64}) = "f64"
_suffix(::Type{ComplexF64}) = "c64"

"""
    _api(T, name) -> Function

Get the C API function for the given scalar type and operation name.
E.g., `_api(Float64, :clone)` returns `C_API.t4a_simplett_f64_clone`.
"""
_api(::Type{T}, name::Symbol) where {T<:_SimpleTTScalar} =
    getfield(C_API, Symbol("t4a_simplett_", _suffix(T), "_", name))

# ============================================================================
# Type definition
# ============================================================================

"""
    SimpleTensorTrain{T}

A simple tensor train (TT/MPS) with statically determined shapes.

`T` can be `Float64` or `ComplexF64`.
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

# ============================================================================
# Constructors
# ============================================================================

"""
    SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Float64)

Create a constant Float64 tensor train with the given site dimensions and value.

# Example
```julia
tt = SimpleTensorTrain([2, 3, 4], 1.0)  # All elements equal to 1.0
```
"""
function SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Float64)
    dims = Csize_t.(site_dims)
    ptr = C_API.t4a_simplett_f64_constant(dims, value)
    return SimpleTensorTrain{Float64}(ptr)
end

"""
    SimpleTensorTrain(site_dims::Vector{<:Integer}, value::ComplexF64)

Create a constant ComplexF64 tensor train with the given site dimensions and value.

# Example
```julia
tt = SimpleTensorTrain([2, 3], 1.0 + 2.0im)
```
"""
function SimpleTensorTrain(site_dims::Vector{<:Integer}, value::ComplexF64)
    dims = Csize_t.(site_dims)
    ptr = C_API.t4a_simplett_c64_constant(dims, real(value), imag(value))
    return SimpleTensorTrain{ComplexF64}(ptr)
end

# Allow implicit conversion from Real to Float64
function SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Real)
    return SimpleTensorTrain(site_dims, Float64(value))
end

# Allow implicit conversion from Complex to ComplexF64 (non-Float64 real part)
function SimpleTensorTrain(site_dims::Vector{<:Integer}, value::Complex)
    return SimpleTensorTrain(site_dims, ComplexF64(value))
end

"""
    SimpleTensorTrain(site_tensors::Vector{<:AbstractArray{Float64,3}})

Create a Float64 tensor train from a vector of 3D site tensors.
Each tensor has shape `(left_dim, site_dim, right_dim)`.
The first tensor must have `left_dim == 1` and the last must have `right_dim == 1`.
"""
function SimpleTensorTrain(site_tensors::Vector{<:AbstractArray{Float64,3}})
    n_sites = length(site_tensors)
    n_sites > 0 || error("site_tensors must be non-empty")

    left_dims_vec = Csize_t[]
    site_dims_vec = Csize_t[]
    right_dims_vec = Csize_t[]
    all_data = Cdouble[]

    for tensor in site_tensors
        l, s, r = size(tensor)
        push!(left_dims_vec, Csize_t(l))
        push!(site_dims_vec, Csize_t(s))
        push!(right_dims_vec, Csize_t(r))
        # Data is already column-major in Julia, matching C API expectation
        append!(all_data, vec(tensor))
    end

    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_simplett_f64_from_site_tensors(
        n_sites, left_dims_vec, site_dims_vec, right_dims_vec,
        all_data, length(all_data), out_ptr
    )
    C_API.check_status(status)
    return SimpleTensorTrain{Float64}(out_ptr[])
end

"""
    SimpleTensorTrain(site_tensors::Vector{<:AbstractArray{ComplexF64,3}})

Create a ComplexF64 tensor train from a vector of 3D site tensors.
Each tensor has shape `(left_dim, site_dim, right_dim)`.
Complex data is passed as interleaved (re, im) doubles.
"""
function SimpleTensorTrain(site_tensors::Vector{<:AbstractArray{ComplexF64,3}})
    n_sites = length(site_tensors)
    n_sites > 0 || error("site_tensors must be non-empty")

    left_dims_vec = Csize_t[]
    site_dims_vec = Csize_t[]
    right_dims_vec = Csize_t[]
    all_data = Cdouble[]

    for tensor in site_tensors
        l, s, r = size(tensor)
        push!(left_dims_vec, Csize_t(l))
        push!(site_dims_vec, Csize_t(s))
        push!(right_dims_vec, Csize_t(r))
        # Reinterpret ComplexF64 to interleaved doubles
        flat = vec(tensor)
        interleaved = reinterpret(Float64, flat)
        append!(all_data, interleaved)
    end

    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = C_API.t4a_simplett_c64_from_site_tensors(
        n_sites, left_dims_vec, site_dims_vec, right_dims_vec,
        all_data, length(all_data), out_ptr
    )
    C_API.check_status(status)
    return SimpleTensorTrain{ComplexF64}(out_ptr[])
end

"""
    zeros(::Type{SimpleTensorTrain}, site_dims::Vector{<:Integer})
    zeros(::Type{SimpleTensorTrain{T}}, site_dims::Vector{<:Integer})

Create a zero tensor train with the given site dimensions.

# Example
```julia
tt = zeros(SimpleTensorTrain, [2, 3, 4])          # Float64
tt = zeros(SimpleTensorTrain{ComplexF64}, [2, 3])  # ComplexF64
```
"""
function Base.zeros(::Type{SimpleTensorTrain{T}}, site_dims::Vector{<:Integer}) where {T<:_SimpleTTScalar}
    dims = Csize_t.(site_dims)
    ptr = _api(T, :zeros)(dims)
    return SimpleTensorTrain{T}(ptr)
end

Base.zeros(::Type{SimpleTensorTrain}, site_dims::Vector{<:Integer}) =
    zeros(SimpleTensorTrain{Float64}, site_dims)

# ============================================================================
# Copy
# ============================================================================

"""
    copy(tt::SimpleTensorTrain)

Create a deep copy of the tensor train.
"""
function Base.copy(tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    new_ptr = _api(T, :clone)(tt.ptr)
    return SimpleTensorTrain{T}(new_ptr)
end

# ============================================================================
# Basic queries
# ============================================================================

"""
    length(tt::SimpleTensorTrain) -> Int

Get the number of sites in the tensor train.
"""
function Base.length(tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    out_len = Ref{Csize_t}(0)
    status = _api(T, :len)(tt.ptr, out_len)
    C_API.check_status(status)
    return Int(out_len[])
end

"""
    sitedims(tt::SimpleTensorTrain) -> Vector{Int}

Get the site (physical) dimensions.
"""
function sitedims(tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    n = length(tt)
    dims = Vector{Csize_t}(undef, n)
    status = _api(T, :site_dims)(tt.ptr, dims)
    C_API.check_status(status)
    return Int.(dims)
end

"""
    linkdims(tt::SimpleTensorTrain) -> Vector{Int}

Get the link (bond) dimensions. Returns n-1 values for n sites.
"""
function linkdims(tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
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
function rank(tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    out_rank = Ref{Csize_t}(0)
    status = _api(T, :rank)(tt.ptr, out_rank)
    C_API.check_status(status)
    return Int(out_rank[])
end

# ============================================================================
# Evaluate (1-indexed)
# ============================================================================

"""
    evaluate(tt::SimpleTensorTrain{Float64}, indices::Vector{<:Integer}) -> Float64

Evaluate the tensor train at a given multi-index (1-based indexing).

# Example
```julia
tt = SimpleTensorTrain([2, 3, 4], 2.0)
val = evaluate(tt, [1, 2, 3])  # Returns 2.0
```
"""
function evaluate(tt::SimpleTensorTrain{Float64}, indices::Vector{<:Integer})
    idx = Csize_t.(indices .- 1)  # Convert to 0-based
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_f64_evaluate(tt.ptr, idx, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    evaluate(tt::SimpleTensorTrain{ComplexF64}, indices::Vector{<:Integer}) -> ComplexF64

Evaluate the complex tensor train at a given multi-index (1-based indexing).
"""
function evaluate(tt::SimpleTensorTrain{ComplexF64}, indices::Vector{<:Integer})
    idx = Csize_t.(indices .- 1)  # Convert to 0-based
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_c64_evaluate(tt.ptr, idx, out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

# Callable interface (1-indexed)
(tt::SimpleTensorTrain{T})(indices::Vector{<:Integer}) where {T<:_SimpleTTScalar} = evaluate(tt, indices)
(tt::SimpleTensorTrain{T})(indices::Integer...) where {T<:_SimpleTTScalar} = evaluate(tt, collect(indices))

# ============================================================================
# Sum, norm
# ============================================================================

"""
    sum(tt::SimpleTensorTrain{Float64}) -> Float64

Compute the sum over all tensor train elements.
"""
function Base.sum(tt::SimpleTensorTrain{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_f64_sum(tt.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    sum(tt::SimpleTensorTrain{ComplexF64}) -> ComplexF64

Compute the sum over all complex tensor train elements.
"""
function Base.sum(tt::SimpleTensorTrain{ComplexF64})
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_c64_sum(tt.ptr, out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

"""
    norm(tt::SimpleTensorTrain) -> Float64

Compute the Frobenius norm of the tensor train.
"""
function LinearAlgebra.norm(tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    out_value = Ref{Cdouble}(0.0)
    status = _api(T, :norm)(tt.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

# ============================================================================
# Site tensor (1-indexed)
# ============================================================================

"""
    sitetensor(tt::SimpleTensorTrain{Float64}, site::Integer) -> Array{Float64, 3}

Get the site tensor at a specific site (1-based indexing).
Returns array with shape `(left_dim, site_dim, right_dim)`.
"""
function sitetensor(tt::SimpleTensorTrain{Float64}, site::Integer)
    n = length(tt)
    1 <= site <= n || error("Site index out of bounds: $site (n=$n)")
    c_site = site - 1  # Convert to 0-based

    sdims = sitedims(tt)
    ldims = linkdims(tt)

    left_dim = site == 1 ? 1 : ldims[site - 1]
    site_dim = sdims[site]
    right_dim = site == n ? 1 : ldims[site]

    total_size = left_dim * site_dim * right_dim
    data = Vector{Cdouble}(undef, total_size)

    out_left = Ref{Csize_t}(0)
    out_site = Ref{Csize_t}(0)
    out_right = Ref{Csize_t}(0)

    status = C_API.t4a_simplett_f64_site_tensor(tt.ptr, c_site, data, out_left, out_site, out_right)
    C_API.check_status(status)

    return reshape(data, (Int(out_left[]), Int(out_site[]), Int(out_right[])))
end

"""
    sitetensor(tt::SimpleTensorTrain{ComplexF64}, site::Integer) -> Array{ComplexF64, 3}

Get the site tensor at a specific site (1-based indexing).
Returns array with shape `(left_dim, site_dim, right_dim)`.
Complex data is stored as interleaved (re, im) doubles in the C API.
"""
function sitetensor(tt::SimpleTensorTrain{ComplexF64}, site::Integer)
    n = length(tt)
    1 <= site <= n || error("Site index out of bounds: $site (n=$n)")
    c_site = site - 1  # Convert to 0-based

    sdims = sitedims(tt)
    ldims = linkdims(tt)

    left_dim = site == 1 ? 1 : ldims[site - 1]
    site_dim = sdims[site]
    right_dim = site == n ? 1 : ldims[site]

    total_size = left_dim * site_dim * right_dim
    # Buffer needs 2x doubles for interleaved (re, im) pairs
    data = Vector{Cdouble}(undef, 2 * total_size)

    out_left = Ref{Csize_t}(0)
    out_site = Ref{Csize_t}(0)
    out_right = Ref{Csize_t}(0)

    status = C_API.t4a_simplett_c64_site_tensor(tt.ptr, c_site, data, out_left, out_site, out_right)
    C_API.check_status(status)

    # Reinterpret interleaved doubles as ComplexF64
    complex_data = reinterpret(ComplexF64, data)
    return reshape(complex_data, (Int(out_left[]), Int(out_site[]), Int(out_right[])))
end

# ============================================================================
# Compress
# ============================================================================

"""
    compress!(tt::SimpleTensorTrain; method::Symbol=:SVD, tolerance::Float64=1e-12, max_bonddim::Int=typemax(Int))

Compress the tensor train in-place.
`method` can be `:SVD`, `:LU`, or `:CI`.
"""
function compress!(tt::SimpleTensorTrain{T};
                   method::Symbol=:SVD,
                   tolerance::Float64=1e-12,
                   max_bonddim::Int=typemax(Int)) where {T<:_SimpleTTScalar}
    method_int = if method == :SVD
        0
    elseif method == :LU
        1
    elseif method == :CI
        2
    else
        error("Unknown compression method: $method. Use :SVD, :LU, or :CI.")
    end
    status = _api(T, :compress)(tt.ptr, method_int, tolerance, max_bonddim)
    C_API.check_status(status)
    return tt
end

# ============================================================================
# Partial sum
# ============================================================================

"""
    partial_sum(tt::SimpleTensorTrain, dims::Vector{<:Integer})

Compute a partial sum over specified dimensions (1-based indexing).
Returns a new `SimpleTensorTrain` with the summed-over dimensions removed.
"""
function partial_sum(tt::SimpleTensorTrain{T}, dims::Vector{<:Integer}) where {T<:_SimpleTTScalar}
    c_dims = Csize_t.(dims .- 1)  # Convert to 0-based
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = _api(T, :partial_sum)(tt.ptr, c_dims, length(c_dims), out_ptr)
    C_API.check_status(status)
    return SimpleTensorTrain{T}(out_ptr[])
end

# ============================================================================
# Arithmetic: add, subtract, scale
# ============================================================================

"""
    +(a::SimpleTensorTrain{T}, b::SimpleTensorTrain{T}) -> SimpleTensorTrain{T}

Add two tensor trains. Returns a new tensor train.
"""
function Base.:+(a::SimpleTensorTrain{T}, b::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = _api(T, :add)(a.ptr, b.ptr, out_ptr)
    C_API.check_status(status)
    return SimpleTensorTrain{T}(out_ptr[])
end

"""
    -(a::SimpleTensorTrain{T}, b::SimpleTensorTrain{T}) -> SimpleTensorTrain{T}

Subtract two tensor trains. Returns a new tensor train.
Implemented as `a + (-one(T) * b)`.
"""
function Base.:-(a::SimpleTensorTrain{T}, b::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    return a + (-one(T) * b)
end

"""
    scale!(tt::SimpleTensorTrain{Float64}, alpha::Real)

Scale a Float64 tensor train in-place by a real factor.
"""
function scale!(tt::SimpleTensorTrain{Float64}, alpha::Real)
    status = C_API.t4a_simplett_f64_scale(tt.ptr, Float64(alpha))
    C_API.check_status(status)
    return tt
end

"""
    scale!(tt::SimpleTensorTrain{ComplexF64}, alpha::Number)

Scale a ComplexF64 tensor train in-place by a complex factor.
"""
function scale!(tt::SimpleTensorTrain{ComplexF64}, alpha::Number)
    c = ComplexF64(alpha)
    status = C_API.t4a_simplett_c64_scale(tt.ptr, real(c), imag(c))
    C_API.check_status(status)
    return tt
end

"""
    *(alpha::Number, tt::SimpleTensorTrain{T}) -> SimpleTensorTrain{T}

Scalar multiplication (creates a copy, then scales).
"""
function Base.:*(alpha::Number, tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    result = copy(tt)
    scale!(result, alpha)
    return result
end

Base.:*(tt::SimpleTensorTrain{T}, alpha::Number) where {T<:_SimpleTTScalar} = alpha * tt

# ============================================================================
# Conjugation
# ============================================================================

"""
    conj(tt::SimpleTensorTrain{ComplexF64}) -> SimpleTensorTrain{ComplexF64}

Return a new tensor train with all site tensors element-wise conjugated.
"""
function Base.conj(tt::SimpleTensorTrain{ComplexF64})
    n = length(tt)
    site_tensors = [conj(sitetensor(tt, i)) for i in 1:n]
    return SimpleTensorTrain(site_tensors)
end

# Float64 conjugation is a no-op (returns a copy for consistency).
Base.conj(tt::SimpleTensorTrain{Float64}) = copy(tt)

# ============================================================================
# Dot product
# ============================================================================

"""
    dot(a::SimpleTensorTrain{Float64}, b::SimpleTensorTrain{Float64}) -> Float64

Compute the dot product (inner product) of two Float64 tensor trains.
"""
function LinearAlgebra.dot(a::SimpleTensorTrain{Float64}, b::SimpleTensorTrain{Float64})
    out_value = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_f64_dot(a.ptr, b.ptr, out_value)
    C_API.check_status(status)
    return out_value[]
end

"""
    dot(a::SimpleTensorTrain{ComplexF64}, b::SimpleTensorTrain{ComplexF64}) -> ComplexF64

Compute the dot product (inner product) of two ComplexF64 tensor trains.
Follows Julia convention: `dot(a, b) = sum(conj(a_i) * b_i)`.

The Rust C API computes the bilinear form `sum(a_i * b_i)`, so we conjugate
the first argument before calling it.
"""
function LinearAlgebra.dot(a::SimpleTensorTrain{ComplexF64}, b::SimpleTensorTrain{ComplexF64})
    a_conj = conj(a)
    out_re = Ref{Cdouble}(0.0)
    out_im = Ref{Cdouble}(0.0)
    status = C_API.t4a_simplett_c64_dot(a_conj.ptr, b.ptr, out_re, out_im)
    C_API.check_status(status)
    return ComplexF64(out_re[], out_im[])
end

# ============================================================================
# Reverse
# ============================================================================

"""
    reverse(tt::SimpleTensorTrain) -> SimpleTensorTrain

Reverse the site ordering of the tensor train. Returns a new tensor train.
"""
function Base.reverse(tt::SimpleTensorTrain{T}) where {T<:_SimpleTTScalar}
    out_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    status = _api(T, :reverse)(tt.ptr, out_ptr)
    C_API.check_status(status)
    return SimpleTensorTrain{T}(out_ptr[])
end

# ============================================================================
# Full tensor
# ============================================================================

"""
    fulltensor(tt::SimpleTensorTrain{Float64}) -> Array{Float64}

Convert the tensor train to a full (dense) tensor.
Returns an Array with dimensions equal to `sitedims(tt)`.
"""
function fulltensor(tt::SimpleTensorTrain{Float64})
    # Query the required buffer size
    out_len = Ref{Csize_t}(0)
    status = C_API.t4a_simplett_f64_fulltensor(tt.ptr, C_NULL, 0, out_len)
    C_API.check_status(status)

    total_size = Int(out_len[])
    data = Vector{Cdouble}(undef, total_size)
    out_len2 = Ref{Csize_t}(0)
    status = C_API.t4a_simplett_f64_fulltensor(tt.ptr, data, total_size, out_len2)
    C_API.check_status(status)

    dims = sitedims(tt)
    return reshape(data, Tuple(dims))
end

"""
    fulltensor(tt::SimpleTensorTrain{ComplexF64}) -> Array{ComplexF64}

Convert the complex tensor train to a full (dense) tensor.
Returns an Array with dimensions equal to `sitedims(tt)`.
"""
function fulltensor(tt::SimpleTensorTrain{ComplexF64})
    # Query the required buffer size (returns number of doubles = 2 * n_elements)
    out_len = Ref{Csize_t}(0)
    status = C_API.t4a_simplett_c64_fulltensor(tt.ptr, C_NULL, 0, out_len)
    C_API.check_status(status)

    total_doubles = Int(out_len[])
    data = Vector{Cdouble}(undef, total_doubles)
    out_len2 = Ref{Csize_t}(0)
    status = C_API.t4a_simplett_c64_fulltensor(tt.ptr, data, total_doubles, out_len2)
    C_API.check_status(status)

    # Reinterpret interleaved doubles as ComplexF64
    complex_data = reinterpret(ComplexF64, data)
    dims = sitedims(tt)
    return reshape(complex_data, Tuple(dims))
end

# ============================================================================
# Display
# ============================================================================

"""
    show(io::IO, tt::SimpleTensorTrain)

Display tensor train information.
"""
function Base.show(io::IO, tt::SimpleTensorTrain{T}) where {T}
    n = length(tt)
    r = rank(tt)
    print(io, "SimpleTensorTrain{$T}(sites=$n, rank=$r)")
end

function Base.show(io::IO, ::MIME"text/plain", tt::SimpleTensorTrain{T}) where {T}
    n = length(tt)
    println(io, "SimpleTensorTrain{$T}")
    println(io, "  Sites: $n")
    println(io, "  Site dims: $(sitedims(tt))")
    println(io, "  Link dims: $(linkdims(tt))")
    println(io, "  Max rank: $(rank(tt))")
end

end # module SimpleTT
