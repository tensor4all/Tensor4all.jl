"""
    random_tt(sites; linkdims=1)
    random_tt(eltype, sites; linkdims=1)
    random_tt(rng, eltype, sites; linkdims=1)

Construct a normalized random `TensorTrain` over `sites` with bond dimensions
controlled by `linkdims`.

# Arguments

- `sites::Vector{Index}`: site indices, one per chain position.
- `eltype::Type{<:Number}`: element type (default `Float64`). `ComplexF64`
  is also supported.
- `linkdims`: target bond dimension. Either a positive integer (uniform) or
  a `Vector{<:Integer}` of length `length(sites) - 1` for non-uniform bonds.
  The actual bond dimension may be capped by the cumulative site dimension
  so that each tensor remains a partial isometry.
- `rng::AbstractRNG`: random number generator (default
  `Random.default_rng()`).

The construction follows the random-unitary-circuit recipe from
`ITensorMPS.random_mps` / `randomCircuitMPS`: each site tensor is built
from a Haar-random partial isometry obtained from the QR decomposition of
a Gaussian matrix. The result is automatically normalized.
"""
function random_tt(sites::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_tt(Random.default_rng(), Float64, sites; linkdims)
end

function random_tt(eltype::Type{<:Number}, sites::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_tt(Random.default_rng(), eltype, sites; linkdims)
end

function random_tt(rng::AbstractRNG, sites::Vector{Index}; linkdims::Union{Integer,Vector{<:Integer}}=1)
    return random_tt(rng, Float64, sites; linkdims)
end

function random_tt(
    rng::AbstractRNG,
    eltype::Type{<:Number},
    sites::Vector{Index};
    linkdims::Union{Integer,Vector{<:Integer}}=1,
)
    n = length(sites)
    n >= 1 || throw(ArgumentError("sites must not be empty"))

    link_dims_full = _resolve_linkdims(linkdims, sites)

    if n == 1
        data = randn(rng, eltype, dim(sites[1]))
        data ./= LinearAlgebra.norm(data)
        return TensorTrain([Tensor(data, [sites[1]])], 0, 2)
    end

    # Build right-to-left so each tensor is a partial isometry on the
    # right-most legs. The right-most tensor has shape (chi_{n-1}, d_n).
    link_indices = Vector{Index}(undef, n - 1)

    chi = min(link_dims_full[n - 1], dim(sites[n]))
    link_indices[n - 1] = Index(chi; tags=[_LINK_TAG, "l=$(n - 1)"])
    Q = _random_isometry(rng, eltype, chi, dim(sites[n]))
    site_tensors = Vector{Tensor}(undef, n)
    site_tensors[n] = Tensor(reshape(Q, chi, dim(sites[n])), [link_indices[n - 1], sites[n]])

    for j in (n - 1):-1:2
        prev_chi = chi
        chi *= dim(sites[j])
        chi = min(link_dims_full[j - 1], chi)
        link_indices[j - 1] = Index(chi; tags=[_LINK_TAG, "l=$(j - 1)"])
        Q = _random_isometry(rng, eltype, chi, dim(sites[j]) * prev_chi)
        site_tensors[j] = Tensor(
            reshape(Q, chi, dim(sites[j]), prev_chi),
            [link_indices[j - 1], sites[j], link_indices[j]],
        )
    end

    # Left boundary site: collapse the dim-1 leftmost link.
    Q = _random_isometry(rng, eltype, 1, dim(sites[1]) * dim(link_indices[1]))
    boundary = reshape(Q, dim(sites[1]), dim(link_indices[1]))
    site_tensors[1] = Tensor(boundary, [sites[1], link_indices[1]])

    return TensorTrain(site_tensors, 0, 2)
end

function _resolve_linkdims(linkdims::Integer, sites::Vector{Index})
    linkdims > 0 || throw(ArgumentError("linkdims must be positive, got $linkdims"))
    n = length(sites)
    return n <= 1 ? Int[] : fill(Int(linkdims), n - 1)
end

function _resolve_linkdims(linkdims::AbstractVector{<:Integer}, sites::Vector{Index})
    n = length(sites)
    expected = max(n - 1, 0)
    length(linkdims) == expected || throw(DimensionMismatch(
        "linkdims must have length $expected for $n sites, got $(length(linkdims))",
    ))
    all(d -> d > 0, linkdims) || throw(ArgumentError("all linkdims must be positive"))
    return collect(Int, linkdims)
end

# Haar-distributed partial isometry of shape (rows, cols).
# Returns a `rows × cols` matrix `Q` with `Q * Q' == I` when `rows <= cols`.
function _random_isometry(rng::AbstractRNG, eltype::Type{<:Number}, rows::Integer, cols::Integer)
    m, n = Int(rows), Int(cols)
    a = m <= n ? randn(rng, eltype, n, m) : randn(rng, eltype, m, n)
    F = LinearAlgebra.qr(a)
    Q = Matrix(F.Q)
    # Fix the sign convention so Q is Haar-distributed (multiply by sign of R diag).
    R = F.R
    d = LinearAlgebra.diag(R)
    phases = [iszero(z) ? one(z) : z / abs(z) for z in d]
    for j in 1:length(phases)
        for i in 1:size(Q, 1)
            Q[i, j] *= phases[j]
        end
    end
    return m <= n ? Matrix(transpose(Q)) : Q
end
