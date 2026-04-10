"""
    TreeTensorNetwork(tensors; adjacency, siteinds, linkinds, backend_handle=nothing)

Create a TreeTN-general review skeleton from local tensor metadata and explicit
topology maps.

The topology helpers are implemented during the skeleton phase, while numerical
operations remain stub-only until backend integration.

# Examples
```julia
julia> ttn = TreeTensorNetwork(...; adjacency=..., siteinds=..., linkinds=...)
```
"""
struct TreeTensorNetwork{V}
    tensors::Dict{V,Tensor}
    adjacency::Dict{V,Vector{V}}
    site_index_map::Dict{V,Vector{Index}}
    link_index_map::Dict{Tuple{V,V},Index}
    backend_handle::Union{Nothing,Ptr{Cvoid}}
end

function TreeTensorNetwork(
    tensors::AbstractDict{V,<:Tensor};
    adjacency::AbstractDict{V,<:AbstractVector{V}},
    siteinds::AbstractDict{V,<:AbstractVector{Index}},
    linkinds::AbstractDict{Tuple{V,V},Index},
    backend_handle::Union{Nothing,Ptr{Cvoid}}=nothing,
) where {V}
    normalized_tensors = Dict(v => tensor for (v, tensor) in tensors)
    normalized_adjacency = Dict(v => collect(neighs) for (v, neighs) in adjacency)
    normalized_siteinds = Dict(v => collect(sitevec) for (v, sitevec) in siteinds)
    normalized_linkinds = Dict(link => ind for (link, ind) in linkinds)
    return TreeTensorNetwork{V}(normalized_tensors, normalized_adjacency, normalized_siteinds, normalized_linkinds, backend_handle)
end

const TensorTrain = TreeTensorNetwork{Int}
const MPS = TensorTrain
const MPO = TensorTrain

@doc """
    TensorTrain

Alias for `TreeTensorNetwork{Int}`, the primary chain-shaped network type in
the Julia API.
""" TensorTrain

@doc """
    MPS

Alias for `TensorTrain`. MPS-ness is a runtime structural property rather than a
separate type.
""" MPS

@doc """
    MPO

Alias for `TensorTrain`. MPO-ness is a runtime structural property rather than a
separate type.
""" MPO

"""
    vertices(ttn)

Return the sorted vertex labels of `ttn`.
"""
vertices(ttn::TreeTensorNetwork) = sort!(collect(keys(ttn.tensors)))

"""
    neighbors(ttn, v)

Return the neighboring vertices of `v`.
"""
function neighbors(ttn::TreeTensorNetwork{V}, v::V) where {V}
    return copy(get(ttn.adjacency, v, V[]))
end

"""
    siteinds(ttn, v)

Return the site indices attached to vertex `v`.
"""
siteinds(ttn::TreeTensorNetwork, v) = copy(ttn.site_index_map[v])

"""
    linkind(ttn, a, b)

Return the link index connecting vertices `a` and `b`.
"""
function linkind(ttn::TreeTensorNetwork, a, b)
    if haskey(ttn.link_index_map, (a, b))
        return ttn.link_index_map[(a, b)]
    elseif haskey(ttn.link_index_map, (b, a))
        return ttn.link_index_map[(b, a)]
    end
    throw(KeyError((a, b)))
end

"""
    is_chain(ttn)

Return `true` when `ttn` has integer vertices `1:n` and chain connectivity.
"""
function is_chain(ttn::TreeTensorNetwork{Int})
    verts = vertices(ttn)
    verts == collect(1:length(verts)) || return false
    isempty(verts) && return true
    degrees = [length(neighbors(ttn, v)) for v in verts]
    length(verts) == 1 && return degrees == [0]
    count(==(1), degrees) == 2 || return false
    count(==(2), degrees) == length(verts) - 2 || return false
    return true
end

is_chain(::TreeTensorNetwork) = false

"""
    is_mps_like(ttn)

Return `true` when each vertex carries exactly one site index.
"""
is_mps_like(ttn::TreeTensorNetwork) = all(v -> length(siteinds(ttn, v)) == 1, vertices(ttn))

"""
    is_mpo_like(ttn)

Return `true` when each vertex carries exactly two site indices.
"""
is_mpo_like(ttn::TreeTensorNetwork) = all(v -> length(siteinds(ttn, v)) == 2, vertices(ttn))

function _require_chain(ttn::TreeTensorNetwork, opname::Symbol)
    is_chain(ttn) || throw(ArgumentError("`$opname` requires a chain topology with vertices 1:n"))
    return ttn
end

"""
    orthogonalize!(ttn, args...)

Placeholder for backend-backed orthogonalization.
"""
orthogonalize!(ttn::TreeTensorNetwork, args...) = throw(SkeletonNotImplemented(:orthogonalize!, :tt))

"""
    truncate!(ttn, args...)

Placeholder for backend-backed truncation.
"""
truncate!(ttn::TreeTensorNetwork, args...) = throw(SkeletonNotImplemented(:truncate!, :tt))

"""
    inner(a, b)

Placeholder for backend-backed TreeTN inner products.
"""
inner(::TreeTensorNetwork, ::TreeTensorNetwork) = throw(SkeletonNotImplemented(:inner, :tt))

"""
    norm(ttn)

Placeholder for backend-backed TreeTN norms.
"""
norm(::TreeTensorNetwork) = throw(SkeletonNotImplemented(:norm, :tt))

"""
    to_dense(ttn)

Placeholder for dense materialization of a TreeTN.
"""
to_dense(::TreeTensorNetwork) = throw(SkeletonNotImplemented(:to_dense, :tt))

"""
    evaluate(ttn, args...)

Placeholder for backend-backed TreeTN evaluation.
"""
evaluate(::TreeTensorNetwork, args...) = throw(SkeletonNotImplemented(:evaluate, :tt))

"""
    contract(a, b)

Placeholder for backend-backed TreeTN contraction.
"""
contract(::TreeTensorNetwork, ::TreeTensorNetwork) = throw(SkeletonNotImplemented(:contract, :tt))
