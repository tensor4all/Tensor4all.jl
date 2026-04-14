module QuanticsTCI

import ..UpstreamQuanticsTCI

function _reexportable_symbols()
    return filter(names(UpstreamQuanticsTCI)) do sym
        sym !== nameof(UpstreamQuanticsTCI) &&
            Base.isidentifier(sym) &&
            sym ∉ (:eval, :include)
    end
end

for sym in _reexportable_symbols()
    @eval const $(sym) = getfield(UpstreamQuanticsTCI, $(QuoteNode(sym)))
    @eval export $(sym)
end

end
