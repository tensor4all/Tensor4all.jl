module ITensorCompat

import Random
import ..Tensor4all: Index, Tensor, commoninds, dim, rank, inds, scalar, prime, setprime, sim, plev
import ..SimpleTT
import ..TensorNetworks
import ..TensorNetworks: TensorTrain
import ..TensorNetworks: add, apply, dag, dot, evaluate, inner, linkdims, linkind, linkinds, norm, siteinds, to_dense
import ..TensorNetworks: orthogonalize, truncate
import ..TensorNetworks: replace_siteinds, replace_siteinds!, replace_siteinds_shared
import ..TensorNetworks: SvdTruncationPolicy

export MPS, MPO
export siteind, siteinds, linkind, linkinds, linkdims, rank
export add, apply, dag, dot, evaluate, inner, norm, replace_siteinds, replace_siteinds!, to_dense
export fixinds, suminds, projectinds, scalar
export maxlinkdim, data
export prime, prime!, replaceprime
export orthogonalize!, truncate!
export random_itensor, random_mps

const ITENSORS_CUTOFF_POLICY = SvdTruncationPolicy(
    measure=:squared_value,
    rule=:discarded_tail_sum,
)

function _validate_chain_topology(tt::TensorTrain, label::AbstractString)
    length(tt) <= 1 && return nothing
    for position in eachindex(tt.data)
        length(Set(inds(tt[position]))) == length(inds(tt[position])) || throw(ArgumentError(
            "$label tensor $position contains duplicate indices",
        ))
    end
    for i in 1:(length(tt) - 1)
        shared = commoninds(inds(tt[i]), inds(tt[i + 1]))
        length(shared) == 1 || throw(ArgumentError(
            "$label expects exactly 1 shared link index between sites $i and $(i + 1), got $(length(shared))",
        ))
    end
    for i in 1:length(tt), j in (i + 2):length(tt)
        shared = commoninds(inds(tt[i]), inds(tt[j]))
        isempty(shared) || throw(ArgumentError(
            "$label expects a chain topology, but sites $i and $j share non-adjacent index/indices $shared",
        ))
    end
    return nothing
end

"""
    MPS(tt)

ITensorMPS-style wrapper over a `TensorNetworks.TensorTrain` whose tensors each
have exactly one site index. The wrapped `TensorTrain` remains the storage and
execution boundary.
"""
mutable struct MPS
    tt::TensorTrain
    function MPS(tt::TensorTrain)
        _is_scalar_mps_train(tt) && return new(tt)
        _validate_chain_topology(tt, "MPS")
        groups = TensorNetworks.siteinds(tt)
        for (position, group) in pairs(groups)
            length(group) == 1 || throw(ArgumentError(
                "MPS expects exactly one site index at tensor $position, got $(length(group))",
            ))
        end
        return new(tt)
    end
end

_is_scalar_mps_train(tt::TensorTrain) =
    length(tt.data) == 1 && rank(only(tt.data)) == 0

Base.length(m::MPS) = _is_scalar_mps_train(m.tt) ? 0 : length(m.tt)
Base.keys(m::MPS) = Base.OneTo(length(m))
Base.firstindex(m::MPS) = firstindex(keys(m))
Base.lastindex(m::MPS) = lastindex(keys(m))
Base.axes(m::MPS) = (keys(m),)
Base.axes(m::MPS, d::Integer) = d == 1 ? keys(m) : Base.OneTo(1)
Base.eachindex(m::MPS) = keys(m)
function Base.iterate(m::MPS, state::Int=firstindex(m))
    state > lastindex(m) && return nothing
    return (m[state], state + 1)
end
Base.pairs(m::MPS) = (i => m[i] for i in eachindex(m))
Base.getindex(m::MPS, i::Int) = m.tt[i]
Base.getindex(m::MPS, indices::AbstractVector{<:Integer}) = [m[Int(i)] for i in indices]
Base.getindex(m::MPS, ::Colon) = [m[i] for i in eachindex(m)]
Base.setindex!(m::MPS, tensor::Tensor, i::Int) = setindex!(m.tt, tensor, i)
Base.eltype(m::MPS) = eltype(first(m.tt))
Base.IteratorEltype(::Type{MPS}) = Base.EltypeUnknown()

siteinds(m::MPS) = _is_scalar_mps_train(m.tt) ? Index[] : [only(group) for group in TensorNetworks.siteinds(m.tt)]

"""
    siteind(m::MPS, i) -> Index
    siteind(W::MPO, i; plev=nothing) -> Index

Return a single site index at position `i`. For an `MPS`, this is
`siteinds(m)[i]`. For an `MPO`, pass `plev=` when the tensor has both input
and output site indices; omitting `plev` is only accepted when there is exactly
one site index.
"""
siteind(m::MPS, i::Integer) = siteinds(m)[i]
linkinds(m::MPS) = TensorNetworks.linkinds(m.tt)
linkind(m::MPS, i::Integer) = TensorNetworks.linkind(m.tt, i)
linkdims(m::MPS) = TensorNetworks.linkdims(m.tt)
rank(m::MPS) = maximum(linkdims(m); init=0)

"""
    siteinds(dim, n; tags="Site") -> Vector{Index}

Construct `n` site indices with dimension `dim`. The generated tags are
`tags` and `"n=i"`, matching Tensor4all's MPS constructor convention while
covering the common ITensor-style `siteinds(dim, n)` test/helper pattern.
"""
function siteinds(dim::Integer, n::Integer; tags::Union{AbstractString,AbstractVector{<:AbstractString}}="Site")
    n >= 0 || throw(ArgumentError("number of site indices must be nonnegative, got $n"))
    base_tags = tags isa AbstractString ? [String(tags)] : String.(tags)
    return [Index(dim; tags=[base_tags..., "n=$i"]) for i in 1:Int(n)]
end

function _mps_links(blocks::AbstractVector{<:Array{T,3}}) where {T}
    return [Index(size(blocks[i], 3); tags=["Link", "l=$i"]) for i in 1:(length(blocks) - 1)]
end

function _validate_mps_blocks(blocks::AbstractVector{<:Array{T,3}}, sites::AbstractVector{<:Index}) where {T}
    length(blocks) == length(sites) || throw(DimensionMismatch(
        "Need one MPS block per site, got $(length(blocks)) blocks and $(length(sites)) sites",
    ))
    isempty(blocks) && throw(ArgumentError("MPS blocks must not be empty"))

    size(first(blocks), 1) == 1 || throw(DimensionMismatch(
        "First MPS block must have left boundary dimension 1, got $(size(first(blocks), 1))",
    ))
    size(last(blocks), 3) == 1 || throw(DimensionMismatch(
        "Last MPS block must have right boundary dimension 1, got $(size(last(blocks), 3))",
    ))

    for i in eachindex(blocks)
        size(blocks[i], 2) == dim(sites[i]) || throw(DimensionMismatch(
            "Block $i physical dimension $(size(blocks[i], 2)) does not match site dimension $(dim(sites[i]))",
        ))
    end
    for i in 1:(length(blocks) - 1)
        size(blocks[i], 3) == size(blocks[i + 1], 1) || throw(DimensionMismatch(
            "MPS bond $i has mismatched dimensions $(size(blocks[i], 3)) and $(size(blocks[i + 1], 1))",
        ))
    end
    return nothing
end

function MPS(blocks::AbstractVector{<:Array{T,3}}, sites::AbstractVector{<:Index}) where {T}
    _validate_mps_blocks(blocks, sites)
    links = _mps_links(blocks)
    tensors = Tensor[]
    for i in eachindex(blocks)
        if length(blocks) == 1
            push!(tensors, Tensor(collect(dropdims(blocks[i]; dims=(1, 3))), [sites[i]]))
        elseif i == 1
            push!(tensors, Tensor(collect(dropdims(blocks[i]; dims=1)), [sites[i], links[i]]))
        elseif i == length(blocks)
            push!(tensors, Tensor(collect(dropdims(blocks[i]; dims=3)), [links[i - 1], sites[i]]))
        else
            push!(tensors, Tensor(blocks[i], [links[i - 1], sites[i], links[i]]))
        end
    end
    return MPS(TensorTrain(tensors))
end

function MPS(blocks::AbstractVector{<:Array{T,3}}) where {T}
    sites = [Index(size(blocks[i], 2); tags=["Site", "n=$i"]) for i in eachindex(blocks)]
    return MPS(blocks, sites)
end

dot(a::MPS, b::MPS) = TensorNetworks.dot(a.tt, b.tt)
inner(a::MPS, b::MPS) = TensorNetworks.inner(a.tt, b.tt)
norm(m::MPS) = TensorNetworks.norm(m.tt)
to_dense(m::MPS) = _is_scalar_mps_train(m.tt) ? only(m.tt.data) : TensorNetworks.to_dense(m.tt)

function evaluate(m::MPS, args...)
    if _is_scalar_mps_train(m.tt)
        isempty(args) || throw(ArgumentError("scalar MPS evaluate does not accept indices or values"))
        return scalar(m)
    end
    return TensorNetworks.evaluate(m.tt, args...)
end

function scalar(m::MPS)
    _is_scalar_mps_train(m.tt) && return scalar(only(m.tt.data))
    return scalar(to_dense(m))
end

function _maybe_wrap_mps(tt::TensorTrain)
    try
        return MPS(tt)
    catch err
        err isa ArgumentError || rethrow()
        return tt
    end
end

function _compat_truncation_kwargs(; cutoff::Real=0.0, maxdim::Integer=0, require_control::Bool=false, kwargs...)
    native_keys = (:threshold, :svd_policy)
    used_native = [key for key in keys(kwargs) if key in native_keys]
    isempty(used_native) || throw(ArgumentError(
        "ITensorCompat truncation is cutoff-only; got native Tensor4all keyword(s) $(Tuple(used_native)). Use TensorNetworks.truncate for threshold or svd_policy.",
    ))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative, got $cutoff"))
    maxdim >= 0 || throw(ArgumentError("maxdim must be nonnegative, got $maxdim"))
    isempty(kwargs) || throw(ArgumentError(
        "Unknown ITensorCompat truncation keyword(s) $(Tuple(keys(kwargs)))",
    ))
    (!require_control || cutoff > 0 || maxdim > 0) || throw(ArgumentError(
        "At least one of cutoff or maxdim must be specified",
    ))
    return (; threshold=cutoff, maxdim=maxdim, svd_policy=ITENSORS_CUTOFF_POLICY)
end

function add(a::MPS, b::MPS; cutoff::Real=0.0, maxdim::Integer=0, kwargs...)
    resolved = _compat_truncation_kwargs(; cutoff, maxdim, kwargs...)
    return MPS(TensorNetworks.add(a.tt, b.tt; resolved...))
end

Base.:+(a::MPS, b::MPS) = add(a, b)
Base.:-(m::MPS) = (-1) * m
Base.:-(a::MPS, b::MPS) = add(a, -b)
Base.:*(alpha::Number, m::MPS) = MPS(alpha * m.tt)
Base.:*(m::MPS, alpha::Number) = alpha * m
Base.:/(m::MPS, alpha::Number) = MPS(m.tt / alpha)
dag(m::MPS) = MPS(TensorNetworks.dag(m.tt))

function replace_siteinds(m::MPS, oldsites, newsites)
    return MPS(TensorNetworks.replace_siteinds(m.tt, collect(oldsites), collect(newsites)))
end

function replace_siteinds!(m::MPS, oldsites, newsites)
    TensorNetworks.replace_siteinds!(m.tt, collect(oldsites), collect(newsites))
    return m
end

fixinds(m::MPS, replacements::Pair{Index,<:Integer}...) =
    _maybe_wrap_mps(TensorNetworks.fixinds(m.tt, replacements...))

suminds(m::MPS, indices::Index...) =
    _maybe_wrap_mps(TensorNetworks.suminds(m.tt, indices...))

projectinds(m::MPS, replacements::Pair{Index,<:AbstractVector{<:Integer}}...) =
    _maybe_wrap_mps(TensorNetworks.projectinds(m.tt, replacements...))

"""
    orthogonalize!(m, site; kwargs...)

Canonicalize `m` at `site`, replacing the wrapped `TensorTrain` and returning
`m`.
"""
function orthogonalize!(m::MPS, site::Integer; kwargs...)
    m.tt = TensorNetworks.orthogonalize(m.tt, site; kwargs...)
    return m
end

"""
    truncate!(m; cutoff=0.0, maxdim=0, kwargs...)

Truncate `m` with ITensors-style `cutoff` semantics. Tensor4all-native
`threshold` and `svd_policy` keywords are intentionally rejected in this
compatibility facade.
"""
function truncate!(m::MPS; cutoff::Real=0.0, maxdim::Integer=0, kwargs...)
    resolved = _compat_truncation_kwargs(; cutoff, maxdim, require_control=true, kwargs...)
    _is_scalar_mps_train(m.tt) && return m
    m.tt = TensorNetworks.truncate(m.tt; resolved...)
    return m
end

"""
    MPO(tt)

ITensorMPS-style wrapper over a `TensorNetworks.TensorTrain` whose tensors each
have exactly two site indices.
"""
mutable struct MPO
    tt::TensorTrain
    function MPO(tt::TensorTrain)
        _validate_chain_topology(tt, "MPO")
        groups = TensorNetworks.siteinds(tt)
        for (position, group) in pairs(groups)
            length(group) == 2 || throw(ArgumentError(
                "MPO expects exactly two site indices at tensor $position, got $(length(group))",
            ))
        end
        return new(tt)
    end
    """
        MPO(tensors::Vector{Tensor})

    Construct an MPO from an existing vector of `Tensor4all.Tensor` objects.
    Each tensor must have at least three indices (two site indices and at least
    one link index). Compatible with the ITensorMPS `MPO(::Vector{ITensor})`
    constructor.
    """
    function MPO(tensors::Vector{<:Tensor})
        isempty(tensors) && throw(ArgumentError("MPO tensor vector must not be empty"))
        return MPO(TensorTrain(tensors))
    end
end

Base.length(W::MPO) = length(W.tt)
Base.keys(W::MPO) = Base.OneTo(length(W))
Base.firstindex(W::MPO) = firstindex(keys(W))
Base.lastindex(W::MPO) = lastindex(keys(W))
Base.axes(W::MPO) = (keys(W),)
Base.axes(W::MPO, d::Integer) = d == 1 ? keys(W) : Base.OneTo(1)
Base.eachindex(W::MPO) = keys(W)
function Base.iterate(W::MPO, state::Int=firstindex(W))
    state > lastindex(W) && return nothing
    return (W[state], state + 1)
end
Base.pairs(W::MPO) = (i => W[i] for i in eachindex(W))
Base.getindex(W::MPO, i::Int) = W.tt[i]
Base.getindex(W::MPO, indices::AbstractVector{<:Integer}) = [W[Int(i)] for i in indices]
Base.getindex(W::MPO, ::Colon) = [W[i] for i in eachindex(W)]
Base.setindex!(W::MPO, tensor::Tensor, i::Int) = setindex!(W.tt, tensor, i)
Base.eltype(W::MPO) = eltype(first(W.tt))
Base.IteratorEltype(::Type{MPO}) = Base.EltypeUnknown()

siteinds(W::MPO) = TensorNetworks.siteinds(W.tt)
function siteind(W::MPO, i::Integer; plev::Union{Nothing,Integer}=nothing)
    group = siteinds(W)[i]
    if plev === nothing
        length(group) == 1 && return only(group)
        throw(ArgumentError(
            "MPO site $i has $(length(group)) site indices; pass plev=... or use siteinds(W)[$i].",
        ))
    end
    matches = filter(index -> index.plev == Int(plev), group)
    length(matches) == 1 || throw(ArgumentError(
        "MPO site $i has $(length(matches)) site indices with plev=$plev; expected exactly one.",
    ))
    return only(matches)
end
siteind(W::MPO, i::Integer, plev::Integer) = siteind(W, i; plev)
linkinds(W::MPO) = TensorNetworks.linkinds(W.tt)
linkind(W::MPO, i::Integer) = TensorNetworks.linkind(W.tt, i)
linkdims(W::MPO) = TensorNetworks.linkdims(W.tt)
rank(W::MPO) = maximum(linkdims(W); init=0)

"""
    to_dense(W::MPO) -> Tensor

Materialize `W` to a dense `Tensor` using the Tensor4all TreeTN backend.
Non-chain MPO topologies are rejected by the `MPO` constructor.
"""
to_dense(W::MPO) = TensorNetworks.to_dense(W.tt)
norm(W::MPO) = TensorNetworks.norm(W.tt)
dag(W::MPO) = MPO(TensorNetworks.dag(W.tt))

"""
    add(A::MPO, B::MPO; cutoff=0.0, maxdim=0) -> MPO
    A + B
    A - B

Add or subtract two MPOs with ITensor-style `cutoff`/`maxdim` keywords. When
backend addition produces a TensorTrain that cannot be represented by the
strict chain `MPO` wrapper, this throws.
"""
function add(a::MPO, b::MPO; cutoff::Real=0.0, maxdim::Integer=0, kwargs...)
    resolved = _compat_truncation_kwargs(; cutoff, maxdim, kwargs...)
    return MPO(TensorNetworks.add(a.tt, b.tt; resolved...))
end

Base.:+(a::MPO, b::MPO) = add(a, b)
Base.:-(w::MPO) = (-1) * w
Base.:-(a::MPO, b::MPO) = add(a, -b)
Base.:*(alpha::Number, w::MPO) = MPO(alpha * w.tt)
Base.:*(w::MPO, alpha::Number) = alpha * w
Base.:/(w::MPO, alpha::Number) = MPO(w.tt / alpha)

function _raw_mpo_blocks(blocks::AbstractVector{<:Array{T,4}}) where {T}
    isempty(blocks) && throw(ArgumentError("MPO raw block constructor requires at least one block"))
    return [copy(block) for block in blocks]
end

"""
    MPO(blocks, input_sites, output_sites)

Construct an MPO wrapper from raw blocks in `(left_link, input_site,
output_site, right_link)` order.
"""
function MPO(
    blocks::AbstractVector{<:Array{T,4}},
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index},
) where {T}
    stt = SimpleTT.TensorTrain{T,4}(_raw_mpo_blocks(blocks))
    return MPO(TensorNetworks.TensorTrain(stt, input_sites, output_sites))
end

"""
    maxlinkdim(m::MPS) -> Int
    maxlinkdim(w::MPO) -> Int

Return the maximum bond dimension (link index dimension) of `m` or `w`.
Equivalent to `rank(m)` / `rank(w)`. Compatible with `ITensorMPS.maxlinkdim`.
"""
maxlinkdim(m::MPS) = rank(m)
maxlinkdim(w::MPO) = rank(w)

"""
    data(m::MPS) -> Vector{Tensor}
    data(w::MPO) -> Vector{Tensor}
    data(tt::TensorTrain) -> Vector{Tensor}

Return the underlying tensor storage vector. Compatible with `ITensors.data`.
"""
data(m::MPS) = m.tt.data
data(w::MPO) = w.tt.data
data(tt::TensorTrain) = tt.data

Base.getproperty(m::MPS, name::Symbol) =
    name === :data ? data(m) : getfield(m, name)
Base.getproperty(w::MPO, name::Symbol) =
    name === :data ? data(w) : getfield(w, name)

Base.propertynames(::MPS, private::Bool=false) =
    private ? (:tt, :data) : (:data,)
Base.propertynames(::MPO, private::Bool=false) =
    private ? (:tt, :data) : (:data,)

function _tensor_vector(value)
    return Tensor[tensor for tensor in value]
end

function Base.setproperty!(m::MPS, name::Symbol, value)
    if name === :data
        tt = TensorTrain(_tensor_vector(value))
        MPS(tt) # validate MPS site structure before mutating `m`
        setfield!(m, :tt, tt)
        return value
    end
    return setfield!(m, name, value)
end

function Base.setproperty!(w::MPO, name::Symbol, value)
    if name === :data
        tt = TensorTrain(_tensor_vector(value))
        MPO(tt) # validate MPO site structure before mutating `w`
        setfield!(w, :tt, tt)
        return value
    end
    return setfield!(w, name, value)
end

"""
    random_itensor([rng], [T=Float64], inds...) -> Tensor

Create a random dense `Tensor` over `inds`, filled with Gaussian entries.
This mirrors the common `ITensors.random_itensor` helper used in tests and
examples. Pass an `AbstractRNG` and/or element type explicitly for
deterministic or complex-valued tensors.
"""
random_itensor(indices::Index...) =
    random_itensor(Random.default_rng(), Float64, indices...)
random_itensor(indices::AbstractVector{<:Index}) =
    random_itensor(Random.default_rng(), Float64, indices...)
random_itensor(T::Type{<:Number}, indices::Index...) =
    random_itensor(Random.default_rng(), T, indices...)
random_itensor(T::Type{<:Number}, indices::AbstractVector{<:Index}) =
    random_itensor(Random.default_rng(), T, indices...)
random_itensor(rng::Random.AbstractRNG, indices::Index...) =
    random_itensor(rng, Float64, indices...)
random_itensor(rng::Random.AbstractRNG, indices::AbstractVector{<:Index}) =
    random_itensor(rng, Float64, indices...)

function random_itensor(rng::Random.AbstractRNG, T::Type{<:Number}, indices::Index...)
    return Tensor(randn(rng, T, dim.(indices)...), indices...)
end

function random_itensor(rng::Random.AbstractRNG, T::Type{<:Number}, indices::AbstractVector{<:Index})
    return random_itensor(rng, T, indices...)
end

"""
    random_mps(sites; linkdims=1)
    random_mps(T, sites; linkdims=1)
    random_mps(rng, T, sites; linkdims=1)

Construct a normalized random `MPS` over `sites` using Tensor4all's
`TensorNetworks.random_tt` implementation. The call patterns mirror
`ITensorMPS.random_mps` for migration-oriented tests and examples.
"""
random_mps(sites::AbstractVector{<:Index}; linkdims=1) =
    MPS(TensorNetworks.random_tt(collect(sites); linkdims))
random_mps(T::Type{<:Number}, sites::AbstractVector{<:Index}; linkdims=1) =
    MPS(TensorNetworks.random_tt(T, collect(sites); linkdims))
random_mps(rng::Random.AbstractRNG, sites::AbstractVector{<:Index}; linkdims=1) =
    MPS(TensorNetworks.random_tt(rng, collect(sites); linkdims))
random_mps(rng::Random.AbstractRNG, T::Type{<:Number}, sites::AbstractVector{<:Index}; linkdims=1) =
    MPS(TensorNetworks.random_tt(rng, T, collect(sites); linkdims))

function _compat_method(method, alg)
    alg === nothing || return Symbol(alg)
    method === nothing && return :zipup
    return Symbol(method)
end

function _compat_backend_kwargs(;
    cutoff::Real=0.0,
    threshold::Real=cutoff,
    alg=nothing,
    method=nothing,
    kwargs...,
)
    return (; method=_compat_method(method, alg), threshold=threshold, kwargs...)
end

function _mpo_input_output_sites(W::MPO)
    groups = siteinds(W)
    all(group -> length(group) == 2, groups) || throw(ArgumentError(
        "ITensorCompat.apply expects each MPO tensor to have exactly two site indices.",
    ))
    input = [group[1] for group in groups]
    output = [group[2] for group in groups]
    return input, output
end

"""
    apply(W::MPO, m::MPS; cutoff=0.0, alg=nothing, method=nothing, kwargs...) -> MPS

Apply MPO `W` to MPS `m` and return an `MPS`. The fast path uses
`TensorNetworks.apply` with `W` represented as a chain-compatible
`LinearOperator`. `cutoff` is translated to Tensor4all's `threshold`; `alg`
accepts ITensor-style strings/symbols such as `"naive"` and is translated to
Tensor4all's `method` keyword.
"""
function apply(W::MPO, m::MPS; cutoff::Real=0.0, alg=nothing, method=nothing, kwargs...)
    input, output = _mpo_input_output_sites(W)
    length(input) == length(m) || throw(DimensionMismatch(
        "MPO has $(length(input)) sites but MPS has $(length(m)) sites",
    ))
    resolved = _compat_backend_kwargs(; cutoff, alg, method, kwargs...)
    op = TensorNetworks.LinearOperator(; mpo=W.tt, input_indices=input, output_indices=output)
    TensorNetworks.set_iospaces!(op, siteinds(m), output)
    return MPS(TensorNetworks.apply(op, m.tt; resolved...))
end

"""
    apply(A::MPO, B::MPO; cutoff=0.0, alg=nothing, method=nothing, kwargs...) -> MPO

Compose two MPOs by contracting their shared site indices. This is a thin
ITensorCompat wrapper over `TensorNetworks.contract`; `cutoff` and `alg` are
mapped as in `apply(MPO, MPS)`.
"""
function apply(A::MPO, B::MPO; cutoff::Real=0.0, alg=nothing, method=nothing, kwargs...)
    resolved = _compat_backend_kwargs(; cutoff, alg, method, kwargs...)
    return MPO(TensorNetworks.contract(A.tt, B.tt; resolved...))
end

"""
    prime(m::MPS, n::Integer=1) -> MPS
    prime(w::MPO, n::Integer=1) -> MPO

Return a copy of `m`/`w` with site index prime levels increased by `n`.
Tensor data is shared (not copied). Compatible with `ITensorMPS.prime`.
"""
function prime(m::MPS, n::Integer=1)
    sites = siteinds(m)
    primed = prime.(sites, Ref(n))
    tt = replace_siteinds_shared(m.tt, sites, primed)
    return MPS(tt)
end

function prime(w::MPO, n::Integer=1)
    groups = TensorNetworks.siteinds(w.tt)
    flat_old = reduce(vcat, groups)
    flat_new = [prime(idx, n) for idx in flat_old]
    tt = replace_siteinds_shared(w.tt, flat_old, flat_new)
    return MPO(tt)
end

"""
    prime!(m::MPS, n::Integer=1) -> MPS
    prime!(w::MPO, n::Integer=1) -> MPO

In-place version of [`prime`](@ref). Modifies site index prime levels in place
and returns the mutated MPS/MPO. Compatible with `ITensorMPS.prime!`.
"""
function prime!(m::MPS, n::Integer=1)
    sites = siteinds(m)
    primed = prime.(sites, Ref(n))
    TensorNetworks.replace_siteinds!(m.tt, sites, primed)
    return m
end

function prime!(w::MPO, n::Integer=1)
    groups = TensorNetworks.siteinds(w.tt)
    flat_old = reduce(vcat, groups)
    flat_new = [prime(idx, n) for idx in flat_old]
    TensorNetworks.replace_siteinds!(w.tt, flat_old, flat_new)
    return w
end

"""
    replaceprime(m::MPS, pairs::Pair{Int,Int}...) -> MPS
    replaceprime(w::MPO, pairs::Pair{Int,Int}...) -> MPO

Replace prime levels in site indices of `m`/`w` according to `pairs`.
Each pair `old => new` replaces indices with `plev == old` to `plev == new`.
Compatible with `ITensorMPS.replaceprime`.
"""
function replaceprime(m::MPS, pairs::Pair{Int,Int}...)
    sites = siteinds(m)
    mapped = map(sites) do idx
        for (old, new) in pairs
            plev(idx) == old && return setprime(idx, new)
        end
        return idx
    end
    tt = replace_siteinds_shared(m.tt, sites, mapped)
    return MPS(tt)
end

function replaceprime(w::MPO, pairs::Pair{Int,Int}...)
    groups = TensorNetworks.siteinds(w.tt)
    mapped = map(groups) do group
        map(group) do idx
            for (old, new) in pairs
                plev(idx) == old && return setprime(idx, new)
            end
            return idx
        end
    end
    flat_old = reduce(vcat, groups)
    flat_new = reduce(vcat, mapped)
    tt = replace_siteinds_shared(w.tt, flat_old, flat_new)
    return MPO(tt)
end

"""
    sim(::typeof(siteinds), m::MPS) -> Vector{Index}
    sim(::typeof(siteinds), w::MPO) -> Vector{Vector{Index}}

Return cloned site indices with fresh IDs but matching dimensions, tags, and
prime levels. Compatible with ITensorMPS's `sim(siteinds, mps)` pattern.
"""
sim(::typeof(siteinds), m::MPS) = [sim(idx) for idx in siteinds(m)]

sim(::typeof(siteinds), w::MPO) = [[sim(idx) for idx in group] for group in TensorNetworks.siteinds(w.tt)]

end
