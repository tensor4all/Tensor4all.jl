module InterpolativeQTT

import ..UpstreamInterpolativeQTT

function _reexportable_symbols()
    return filter(names(UpstreamInterpolativeQTT)) do sym
        sym !== nameof(UpstreamInterpolativeQTT) &&
            Base.isidentifier(sym) &&
            sym ∉ (:eval, :include)
    end
end

for sym in _reexportable_symbols()
    @eval const $(sym) = getfield(UpstreamInterpolativeQTT, $(QuoteNode(sym)))
    @eval export $(sym)
end

end
