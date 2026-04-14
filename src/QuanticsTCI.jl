module QuanticsTCI

import ..UpstreamQuanticsTCI

for sym in names(UpstreamQuanticsTCI)
    if Base.isidentifier(sym) && sym ∉ (:eval, :include)
        @eval const $(sym) = getfield(UpstreamQuanticsTCI, $(QuoteNode(sym)))
        @eval export $(sym)
    end
end

end
