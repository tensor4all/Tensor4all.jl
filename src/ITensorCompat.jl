module ITensorCompat

import ..Tensor4all: rank, scalar
using ..Tensor4all: Index, Tensor, dim, inds
using ..Tensor4all.SimpleTT
using ..Tensor4all.TensorNetworks

export MPS, MPO
export siteinds, linkinds, linkdims
export inner, dot, norm, add, dag
export replace_siteinds!, replace_siteinds, fixinds, suminds, projectinds
export to_dense, evaluate, scalar, orthogonalize!, truncate!

mutable struct MPS
    tt::TensorNetworks.TensorTrain

    function MPS(tt::TensorNetworks.TensorTrain)
        _validate_mps(tt)
        return new(tt)
    end
end

mutable struct MPO
    tt::TensorNetworks.TensorTrain

    function MPO(tt::TensorNetworks.TensorTrain)
        _validate_mpo(tt)
        return new(tt)
    end
end

_is_scalar_mps_train(tt::TensorNetworks.TensorTrain) =
    length(tt.data) == 1 && rank(only(tt.data)) == 0

function _validate_site_arity(tt::TensorNetworks.TensorTrain, expected::Integer, label::AbstractString)
    isempty(tt.data) && return nothing
    site_groups = TensorNetworks.siteinds(tt)
    for (position, group) in pairs(site_groups)
        length(group) == expected || throw(ArgumentError(
            "$label tensor $position expected $expected site-like index/indices, got $(length(group)): $group",
        ))
    end
    return nothing
end

function _validate_mps(tt::TensorNetworks.TensorTrain)
    _is_scalar_mps_train(tt) && return nothing
    return _validate_site_arity(tt, 1, "MPS")
end

function _validate_mpo(tt::TensorNetworks.TensorTrain)
    return _validate_site_arity(tt, 2, "MPO")
end

Base.length(m::MPS) = _is_scalar_mps_train(m.tt) ? 0 : length(m.tt)
Base.length(m::MPO) = length(m.tt)
Base.iterate(m::Union{MPS,MPO}, state...) = iterate(m.tt.data, state...)
Base.getindex(m::Union{MPS,MPO}, i::Int) = m.tt[i]

function Base.setindex!(m::Union{MPS,MPO}, value::Tensor, i::Int)
    m.tt[i] = value
    return value
end

function siteinds(m::MPS)
    _is_scalar_mps_train(m.tt) && return Index[]
    return [only(group) for group in TensorNetworks.siteinds(m.tt)]
end

siteinds(m::MPO) = TensorNetworks.siteinds(m.tt)

linkinds(m::Union{MPS,MPO}) = TensorNetworks.linkinds(m.tt)
linkdims(m::Union{MPS,MPO}) = TensorNetworks.linkdims(m.tt)
rank(m::Union{MPS,MPO}) = length(m)

function Base.eltype(m::Union{MPS,MPO})
    isempty(m.tt.data) && return Float64
    return promote_type(map(eltype, m.tt.data)...)
end

function _raw_blocks(blocks::AbstractVector{<:Array{T,N}}, label::AbstractString) where {T,N}
    isempty(blocks) && throw(ArgumentError("$label raw block constructor requires at least one block"))
    return [copy(block) for block in blocks]
end

function _generated_mps_sites(blocks::AbstractVector{<:Array{T,3}}) where {T}
    return [Index(size(blocks[i], 2); tags=["Site", "n=$i"]) for i in eachindex(blocks)]
end

"""
    MPS(blocks, sites)
    MPS(blocks)

Construct an MPS wrapper from raw blocks in `(left_link, site, right_link)`
order. When `sites` are omitted, stable `["Site", "n=i"]` site tags are
generated.
"""
function MPS(blocks::AbstractVector{<:Array{T,3}}, sites::AbstractVector{<:Index}) where {T}
    stt = SimpleTT.TensorTrain{T,3}(_raw_blocks(blocks, "MPS"))
    return MPS(TensorNetworks.TensorTrain(stt, sites))
end

MPS(blocks::AbstractVector{<:Array{T,3}}) where {T} =
    MPS(blocks, _generated_mps_sites(blocks))

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
    stt = SimpleTT.TensorTrain{T,4}(_raw_blocks(blocks, "MPO"))
    return MPO(TensorNetworks.TensorTrain(stt, input_sites, output_sites))
end

const ITENSORS_CUTOFF_POLICY = TensorNetworks.SvdTruncationPolicy(;
    measure=:squared_value,
    rule=:discarded_tail_sum,
)

function _reject_native_truncation_kwargs(kwargs)
    bad = intersect(collect(keys(kwargs)), [:threshold, :svd_policy])
    isempty(bad) || throw(ArgumentError(
        "ITensorCompat truncation is cutoff-only; got native Tensor4all keyword(s) $(Tuple(bad)). Use TensorNetworks for threshold or svd_policy.",
    ))
    isempty(kwargs) || throw(ArgumentError(
        "Unknown ITensorCompat truncation keyword(s) $(Tuple(keys(kwargs)))",
    ))
    return nothing
end

_wrap_mps(tt::TensorNetworks.TensorTrain) = MPS(tt)

function _maybe_wrap_mps(tt::TensorNetworks.TensorTrain)
    try
        return MPS(tt)
    catch err
        err isa ArgumentError || rethrow()
        return tt
    end
end

dot(a::MPS, b::MPS) = TensorNetworks.dot(a.tt, b.tt)
inner(a::MPS, b::MPS) = TensorNetworks.inner(a.tt, b.tt)
norm(m::MPS) = TensorNetworks.norm(m.tt)

function add(a::MPS, b::MPS; cutoff::Real=0.0, maxdim::Integer=0, kwargs...)
    _reject_native_truncation_kwargs(kwargs)
    tt = TensorNetworks.add(
        a.tt,
        b.tt;
        threshold=cutoff,
        maxdim=maxdim,
        svd_policy=ITENSORS_CUTOFF_POLICY,
    )
    return MPS(tt)
end

Base.:+(a::MPS, b::MPS) = add(a, b)
Base.:*(α::Number, m::MPS) = MPS(α * m.tt)
Base.:*(m::MPS, α::Number) = α * m
Base.:/(m::MPS, α::Number) = MPS(m.tt / α)

dag(m::MPS) = MPS(TensorNetworks.dag(m.tt))

function replace_siteinds!(m::MPS, oldsites::AbstractVector{<:Index}, newsites::AbstractVector{<:Index})
    TensorNetworks.replace_siteinds!(m.tt, oldsites, newsites)
    return m
end

replace_siteinds(m::MPS, oldsites::AbstractVector{<:Index}, newsites::AbstractVector{<:Index}) =
    MPS(TensorNetworks.replace_siteinds(m.tt, oldsites, newsites))

fixinds(m::MPS, replacements::Pair{Index,<:Integer}...) =
    _maybe_wrap_mps(TensorNetworks.fixinds(m.tt, replacements...))

suminds(m::MPS, indices::Index...) =
    _maybe_wrap_mps(TensorNetworks.suminds(m.tt, indices...))

projectinds(m::MPS, replacements::Pair{Index,<:AbstractVector{<:Integer}}...) =
    _maybe_wrap_mps(TensorNetworks.projectinds(m.tt, replacements...))

function to_dense(m::MPS)
    _is_scalar_mps_train(m.tt) && return only(m.tt.data)
    return TensorNetworks.to_dense(m.tt)
end

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

function orthogonalize!(m::MPS, site::Integer; kwargs...)
    m.tt = TensorNetworks.orthogonalize(m.tt, site; kwargs...)
    return m
end

function truncate!(m::MPS; cutoff::Real=0.0, maxdim::Integer=0, kwargs...)
    _reject_native_truncation_kwargs(kwargs)
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative, got $cutoff"))
    maxdim >= 0 || throw(ArgumentError("maxdim must be nonnegative, got $maxdim"))
    (cutoff > 0 || maxdim > 0) || throw(ArgumentError(
        "At least one of cutoff or maxdim must be specified",
    ))
    _is_scalar_mps_train(m.tt) && return m

    m.tt = TensorNetworks.truncate(
        m.tt;
        threshold=cutoff,
        maxdim=maxdim,
        svd_policy=ITENSORS_CUTOFF_POLICY,
    )
    return m
end

end
