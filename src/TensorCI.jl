module TensorCI

using TensorCrossInterpolation
using ..SimpleTT

export crossinterpolate2

function crossinterpolate2(::Type{T}, f, localdims; kwargs...) where {T}
    if length(localdims) == 1
        site = Array(reshape([convert(T, f([i])) for i in 1:localdims[1]], 1, localdims[1], 1))
        return SimpleTT.TensorTrain{T,3}([site])
    end

    tci, _, _ = TensorCrossInterpolation.crossinterpolate2(T, f, localdims; kwargs...)
    tt = TensorCrossInterpolation.tensortrain(tci)
    return SimpleTT.TensorTrain{T,3}(TensorCrossInterpolation.sitetensors(tt))
end

end
