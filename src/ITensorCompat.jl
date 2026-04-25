module ITensorCompat

import ..Tensor4all: Index, Tensor, dim, inds, rank, scalar
import ..SimpleTT
import ..TensorNetworks
import ..TensorNetworks: TensorTrain
import ..TensorNetworks: add, dag, dot, evaluate, inner, linkdims, linkinds, norm, siteinds, to_dense
import ..TensorNetworks: orthogonalize, truncate
import ..TensorNetworks: replace_siteinds, replace_siteinds!
import ..TensorNetworks: SvdTruncationPolicy

export MPS, MPO
export siteinds, linkinds, linkdims, rank
export add, dag, dot, evaluate, inner, norm, replace_siteinds, replace_siteinds!, to_dense
export fixinds, suminds, projectinds, scalar
export maxlinkdim, data
export orthogonalize!, truncate!

const ITENSORS_CUTOFF_POLICY = SvdTruncationPolicy(
    measure=:squared_value,
    rule=:discarded_tail_sum,
)

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
Base.iterate(m::MPS, state...) = iterate(m.tt, state...)
Base.getindex(m::MPS, i::Int) = m.tt[i]
Base.setindex!(m::MPS, tensor::Tensor, i::Int) = setindex!(m.tt, tensor, i)
Base.eltype(m::MPS) = eltype(first(m.tt).data)

siteinds(m::MPS) = _is_scalar_mps_train(m.tt) ? Index[] : [only(group) for group in TensorNetworks.siteinds(m.tt)]
linkinds(m::MPS) = TensorNetworks.linkinds(m.tt)
linkdims(m::MPS) = TensorNetworks.linkdims(m.tt)
rank(m::MPS) = maximum(linkdims(m); init=0)

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
        groups = TensorNetworks.siteinds(tt)
        for (position, group) in pairs(groups)
            length(group) == 2 || throw(ArgumentError(
                "MPO expects exactly two site indices at tensor $position, got $(length(group))",
            ))
        end
        return new(tt)
    end
end

Base.length(W::MPO) = length(W.tt)
Base.getindex(W::MPO, i::Int) = W.tt[i]
Base.setindex!(W::MPO, tensor::Tensor, i::Int) = setindex!(W.tt, tensor, i)

siteinds(W::MPO) = TensorNetworks.siteinds(W.tt)
linkinds(W::MPO) = TensorNetworks.linkinds(W.tt)
linkdims(W::MPO) = TensorNetworks.linkdims(W.tt)
rank(W::MPO) = maximum(linkdims(W); init=0)
dag(W::MPO) = MPO(TensorNetworks.dag(W.tt))

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

end
