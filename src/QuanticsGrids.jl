module QuanticsGrids

import ..UpstreamQuanticsGrids

function _reexportable_symbols()
    return filter(names(UpstreamQuanticsGrids)) do sym
        sym !== nameof(UpstreamQuanticsGrids) &&
            Base.isidentifier(sym) &&
            sym ∉ (:eval, :include)
    end
end

for sym in _reexportable_symbols()
    @eval const $(sym) = getfield(UpstreamQuanticsGrids, $(QuoteNode(sym)))
    @eval export $(sym)
end

end
