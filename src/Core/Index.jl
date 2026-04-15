"""
    Index(dim; tags=String[], plev=0, id=nothing)

Create a backend-backed indexed tensor leg.

The public Julia API stays metadata-first, but values are stored in the
`tensor4all-capi` backend once the backend library is available.

# Examples
```jldoctest
julia> using Tensor4all

julia> i = Index(4; tags=["x"], plev=1);

julia> (dim(i), tags(i), plev(i))
(4, ["x"], 1)
```
"""
mutable struct Index
    ptr::Ptr{Cvoid}
end

function _normalized_tags(tags::AbstractVector{<:AbstractString})
    normalized = String.(tags)
    sort!(unique!(normalized))
    return normalized
end

_tags_csv(tags::AbstractVector{String}) = join(tags, ",")

function _adopt_index(ptr::Ptr{Cvoid}, context::AbstractString)
    ptr = _check_ptr(ptr, context)
    index = Index(ptr)
    finalizer(i -> _release_handle!(:t4a_index_release, i, :ptr), index)
    return index
end

function _clone_index(i::Index)
    ptr = ccall(_capi_symbol(:t4a_index_clone), Ptr{Cvoid}, (Ptr{Cvoid},), i.ptr)
    return _adopt_index(ptr, "t4a_index_clone")
end

function Index(dim::Integer; tags::AbstractVector{<:AbstractString}=String[], plev::Integer=0, id=nothing)
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    plev >= 0 || throw(ArgumentError("Index prime level must be nonnegative, got $plev"))
    normalized_tags = _normalized_tags(tags)
    tags_csv = _tags_csv(normalized_tags)
    ptr = if id === nothing
        isempty(normalized_tags) ?
            ccall(_capi_symbol(:t4a_index_new), Ptr{Cvoid}, (Csize_t,), dim) :
            ccall(_capi_symbol(:t4a_index_new_with_tags), Ptr{Cvoid}, (Csize_t, Cstring), dim, tags_csv)
    else
        id >= 0 || throw(ArgumentError("Index id must be nonnegative, got $id"))
        isempty(normalized_tags) ?
            ccall(_capi_symbol(:t4a_index_new_with_id), Ptr{Cvoid}, (Csize_t, UInt64, Ptr{Cchar}), dim, UInt64(id), C_NULL) :
            ccall(_capi_symbol(:t4a_index_new_with_id), Ptr{Cvoid}, (Csize_t, UInt64, Cstring), dim, UInt64(id), tags_csv)
    end
    index = _adopt_index(ptr, "t4a_index_new")
    plev == 0 || _check_status(
        ccall(_capi_symbol(:t4a_index_set_plev), _StatusCode, (Ptr{Cvoid}, Int64), index.ptr, plev),
        "t4a_index_set_plev",
    )
    return index
end

"""
    dim(i)

    Return the dimension of `i`.
"""
function dim(i::Index)
    out = Ref{Csize_t}(0)
    _check_status(
        ccall(_capi_symbol(:t4a_index_dim), _StatusCode, (Ptr{Cvoid}, Ref{Csize_t}), i.ptr, out),
        "t4a_index_dim",
    )
    return Int(out[])
end

"""
    id(i)

    Return the stable identifier of `i`.
"""
function id(i::Index)
    out = Ref{UInt64}(0)
    _check_status(
        ccall(_capi_symbol(:t4a_index_id), _StatusCode, (Ptr{Cvoid}, Ref{UInt64}), i.ptr, out),
        "t4a_index_id",
    )
    return out[]
end

"""
    tags(i)

    Return a copy of the string tags attached to `i`.
"""
function tags(i::Index)
    raw = _query_cstring(:t4a_index_get_tags, i.ptr, "t4a_index_get_tags")
    return isempty(raw) ? String[] : String.(split(raw, ","))
end

"""
    plev(i)

    Return the prime level of `i`.
"""
function plev(i::Index)
    out = Ref{Int64}(0)
    _check_status(
        ccall(_capi_symbol(:t4a_index_get_plev), _StatusCode, (Ptr{Cvoid}, Ref{Int64}), i.ptr, out),
        "t4a_index_get_plev",
    )
    return Int(out[])
end

"""
    hastag(i, tag)

    Return `true` when `tag` is attached to `i`.
"""
function hastag(i::Index, tag::AbstractString)
    rc = ccall(_capi_symbol(:t4a_index_has_tag), Cint, (Ptr{Cvoid}, Cstring), i.ptr, String(tag))
    rc >= 0 || _throw_last_error("t4a_index_has_tag")
    return rc == 1
end

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
    j = _clone_index(i)
    _check_status(
        ccall(_capi_symbol(:t4a_index_set_plev), _StatusCode, (Ptr{Cvoid}, Int64), j.ptr, new_plev),
        "t4a_index_set_plev",
    )
    return j
end

"""
    noprime(i)

    Return `i` with prime level reset to zero.
"""
noprime(i::Index) = setprime(i, 0)

"""
    setprime(i, n)

    Return `i` with prime level set to `n`.
"""
function setprime(i::Index, n::Integer)
    n >= 0 || throw(ArgumentError("Index prime level must be nonnegative, got $n"))
    j = _clone_index(i)
    _check_status(
        ccall(_capi_symbol(:t4a_index_set_plev), _StatusCode, (Ptr{Cvoid}, Int64), j.ptr, n),
        "t4a_index_set_plev",
    )
    return j
end

Base.:(==)(a::Index, b::Index) = (
    dim(a) == dim(b) &&
    id(a) == id(b) &&
    tags(a) == tags(b) &&
    plev(a) == plev(b)
)

Base.hash(i::Index, h::UInt) = hash((dim(i), id(i), tags(i), plev(i)), h)

function Base.show(io::IO, i::Index)
    index_tags = tags(i)
    tagstring = isempty(index_tags) ? "-" : join(index_tags, ",")
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
