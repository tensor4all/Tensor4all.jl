module QuanticsGrids

import ..UpstreamQuanticsGrids

for sym in names(UpstreamQuanticsGrids)
    if Base.isidentifier(sym) && sym ∉ (:eval, :include)
        @eval const $(sym) = getfield(UpstreamQuanticsGrids, $(QuoteNode(sym)))
        @eval export $(sym)
    end
end

end
