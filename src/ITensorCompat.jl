module ITensorCompat

import ..Tensor4all: Index, Tensor, inds
import ..TensorNetworks
import ..TensorNetworks: TensorTrain
import ..TensorNetworks: add, dag, dot, evaluate, inner, linkdims, linkinds, norm, siteinds, to_dense
import ..TensorNetworks: orthogonalize, truncate
import ..TensorNetworks: replace_siteinds, replace_siteinds!
import ..TensorNetworks: SvdTruncationPolicy

export MPS, MPO
export siteinds, linkinds, linkdims, rank
export add, dag, dot, evaluate, inner, norm, replace_siteinds, replace_siteinds!, to_dense
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
        groups = TensorNetworks.siteinds(tt)
        for (position, group) in pairs(groups)
            length(group) == 1 || throw(ArgumentError(
                "MPS expects exactly one site index at tensor $position, got $(length(group))",
            ))
        end
        return new(tt)
    end
end

Base.length(m::MPS) = length(m.tt)
Base.iterate(m::MPS, state...) = iterate(m.tt, state...)
Base.getindex(m::MPS, i::Int) = m.tt[i]
Base.setindex!(m::MPS, tensor::Tensor, i::Int) = setindex!(m.tt, tensor, i)
Base.eltype(m::MPS) = eltype(first(m.tt).data)

siteinds(m::MPS) = [only(group) for group in TensorNetworks.siteinds(m.tt)]
linkinds(m::MPS) = TensorNetworks.linkinds(m.tt)
linkdims(m::MPS) = TensorNetworks.linkdims(m.tt)
rank(m::MPS) = maximum(linkdims(m); init=0)

dot(a::MPS, b::MPS) = TensorNetworks.dot(a.tt, b.tt)
inner(a::MPS, b::MPS) = TensorNetworks.inner(a.tt, b.tt)
norm(m::MPS) = TensorNetworks.norm(m.tt)
to_dense(m::MPS) = TensorNetworks.to_dense(m.tt)
evaluate(m::MPS, indices, values) = TensorNetworks.evaluate(m.tt, collect(indices), values)

function add(a::MPS, b::MPS; kwargs...)
    return MPS(TensorNetworks.add(a.tt, b.tt; kwargs...))
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

function _compat_truncation_kwargs(; cutoff::Real=0.0, kwargs...)
    native_keys = (:threshold, :svd_policy)
    used_native = [key for key in keys(kwargs) if key in native_keys]
    isempty(used_native) || throw(ArgumentError(
        "ITensorCompat truncation is cutoff-only; got native Tensor4all keyword(s) $(Tuple(used_native)). Use TensorNetworks.truncate for threshold or svd_policy.",
    ))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative, got $cutoff"))
    cutoff == 0.0 && return kwargs
    return (; threshold=cutoff, svd_policy=ITENSORS_CUTOFF_POLICY, kwargs...)
end

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
function truncate!(m::MPS; cutoff::Real=0.0, kwargs...)
    resolved = _compat_truncation_kwargs(; cutoff, kwargs...)
    m.tt = TensorNetworks.truncate(m.tt; resolved...)
    return m
end

"""
    MPO(tt)

Placeholder wrapper for an ITensorMPS-style matrix-product operator.
"""
mutable struct MPO
    tt::TensorTrain
end

end
