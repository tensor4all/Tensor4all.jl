const _next_index_id = Ref{UInt64}(0)

next_index_id() = (_next_index_id[] += UInt64(1))

"""
    Index(dim; tags=String[], plev=0, id=next_index_id(), backend_handle=nothing)

Create a Julia-side review skeleton for an indexed tensor leg.

The metadata behavior is implemented during the skeleton phase, while the
optional backend handle keeps the public shape aligned with the eventual
backend-facing design.

# Examples
```jldoctest
julia> using Tensor4all

julia> i = Index(4; tags=["x"], plev=1);

julia> (dim(i), tags(i), plev(i))
(4, ["x"], 1)
```
"""
struct Index
    dim::Int
    id::UInt64
    tags::Vector{String}
    plev::Int
    backend_handle::Union{Nothing,Ptr{Cvoid}}
end

function Index(
    dim::Integer;
    tags::AbstractVector{<:AbstractString}=String[],
    plev::Integer=0,
    id::Integer=next_index_id(),
    backend_handle::Union{Nothing,Ptr{Cvoid}}=nothing,
)
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    plev >= 0 || throw(ArgumentError("Index prime level must be nonnegative, got $plev"))
    id >= 0 || throw(ArgumentError("Index id must be nonnegative, got $id"))
    return Index(
        Int(dim),
        UInt64(id),
        collect(String.(tags)),
        Int(plev),
        backend_handle,
    )
end

"""
    dim(i)

Return the dimension of `i`.
"""
dim(i::Index) = i.dim

"""
    id(i)

Return the stable identifier of `i`.
"""
id(i::Index) = i.id

"""
    tags(i)

Return a copy of the string tags attached to `i`.
"""
tags(i::Index) = copy(i.tags)

"""
    plev(i)

Return the prime level of `i`.
"""
plev(i::Index) = i.plev

"""
    hastag(i, tag)

Return `true` when `tag` is attached to `i`.
"""
hastag(i::Index, tag::AbstractString) = String(tag) in i.tags

"""
    sim(i)

Return a similar index with matching metadata and a fresh identifier.
"""
sim(i::Index) = Index(dim(i); tags=tags(i), plev=plev(i))

"""
    prime(i, n=1)

Return `i` with its prime level increased by `n`.
"""
function prime(i::Index, n::Integer=1)
    new_plev = plev(i) + Int(n)
    new_plev >= 0 || throw(ArgumentError("Index prime level must stay nonnegative, got $new_plev"))
    return Index(dim(i); tags=tags(i), plev=new_plev, id=id(i), backend_handle=i.backend_handle)
end

"""
    noprime(i)

Return `i` with prime level reset to zero.
"""
noprime(i::Index) = Index(dim(i); tags=tags(i), plev=0, id=id(i), backend_handle=i.backend_handle)

"""
    setprime(i, n)

Return `i` with prime level set to `n`.
"""
function setprime(i::Index, n::Integer)
    n >= 0 || throw(ArgumentError("Index prime level must be nonnegative, got $n"))
    return Index(dim(i); tags=tags(i), plev=Int(n), id=id(i), backend_handle=i.backend_handle)
end

Base.:(==)(a::Index, b::Index) = (
    a.dim == b.dim &&
    a.id == b.id &&
    a.tags == b.tags &&
    a.plev == b.plev
)

Base.hash(i::Index, h::UInt) = hash((i.dim, i.id, i.tags, i.plev), h)

function Base.show(io::IO, i::Index)
    tagstring = isempty(i.tags) ? "-" : join(i.tags, ",")
    print(io, "Index(", dim(i), "|", tagstring, "; plev=", plev(i), ")")
end

"""
    replaceind(xs, old, new)

Replace `old` by `new` in the index collection `xs`.
"""
replaceind(xs::AbstractVector{Index}, old::Index, new::Index) = [x == old ? new : x for x in xs]

"""
    replaceinds(xs, replacements...)

Apply multiple index replacements in sequence to `xs`.
"""
function replaceinds(xs::AbstractVector{Index}, replacements::Pair{Index,Index}...)
    ys = collect(xs)
    for (old, new) in replacements
        ys = replaceind(ys, old, new)
    end
    return ys
end

"""
    commoninds(xs, ys)

Return the indices in `xs` that also occur in `ys`, preserving the order from
`xs`.
"""
commoninds(xs::AbstractVector{Index}, ys::AbstractVector{Index}) = [x for x in xs if x in ys]

"""
    uniqueinds(xs, ys)

Return the indices in `xs` that do not occur in `ys`, preserving the order from
`xs`.
"""
uniqueinds(xs::AbstractVector{Index}, ys::AbstractVector{Index}) = [x for x in xs if x ∉ ys]
