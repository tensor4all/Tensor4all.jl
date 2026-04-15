module TensorCI

using TensorCrossInterpolation

"""
    TensorCI2

Public alias for the multi-site tensor-cross-interpolation result type from
`TensorCrossInterpolation.jl`.
"""
const TensorCI2 = TensorCrossInterpolation.TensorCI2

function _reexportable_symbols()
    return filter(names(TensorCrossInterpolation)) do sym
        sym !== :crossinterpolate2 &&
            sym !== nameof(TensorCrossInterpolation) &&
            Base.isidentifier(sym) &&
            sym ∉ (:eval, :include)
    end
end

for sym in _reexportable_symbols()
    @eval const $(sym) = getfield(TensorCrossInterpolation, $(QuoteNode(sym)))
    @eval export $(sym)
end

export TensorCI2, crossinterpolate2

"""
    crossinterpolate2(T, f, localdims; kwargs...)

Run tensor cross interpolation and return the public `TensorCI2` result.
"""
function crossinterpolate2(::Type{T}, f, localdims; kwargs...) where {T}
    length(localdims) >= 2 || throw(
        ArgumentError(
            "TensorCI.crossinterpolate2 currently requires at least two local dimensions because the public return type is TensorCI2.",
        ),
    )
    tci, _, _ = TensorCrossInterpolation.crossinterpolate2(T, f, localdims; kwargs...)
    return tci
end

end
