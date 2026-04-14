module TensorCI

using TensorCrossInterpolation

const TensorCI2 = TensorCrossInterpolation.TensorCI2

for sym in names(TensorCrossInterpolation)
    if sym !== :crossinterpolate2 && Base.isidentifier(sym) && sym ∉ (:eval, :include)
        @eval const $(sym) = getfield(TensorCrossInterpolation, $(QuoteNode(sym)))
        @eval export $(sym)
    end
end

export TensorCI2, crossinterpolate2

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
