"""
    Tensor4allTCIExt

Extension module providing bidirectional conversion between
Tensor4all.SimpleTT.SimpleTensorTrain and TensorCrossInterpolation.TensorTrain.
"""
module Tensor4allTCIExt

using Tensor4all
using TensorCrossInterpolation

import Tensor4all.SimpleTT: SimpleTensorTrain, sitetensor

"""
    SimpleTensorTrain(tt::TensorCrossInterpolation.TensorTrain{V,3}) where V

Convert a TensorCrossInterpolation.TensorTrain to a SimpleTensorTrain.

Extracts the site tensors (each of shape (left, site, right)) and
constructs a SimpleTensorTrain from them.
"""
function Tensor4all.SimpleTT.SimpleTensorTrain(tt::TensorCrossInterpolation.TensorTrain{V,3}) where V
    return SimpleTensorTrain(tt.sitetensors)
end

"""
    TensorCrossInterpolation.TensorTrain(stt::SimpleTensorTrain{T}) where T

Convert a SimpleTensorTrain to a TensorCrossInterpolation.TensorTrain.

Extracts site tensors via sitetensor(stt, i) for each site and
constructs a TensorCrossInterpolation.TensorTrain from them.
"""
function TensorCrossInterpolation.TensorTrain(stt::SimpleTensorTrain{T}) where T
    n = length(stt)
    site_tensors = [sitetensor(stt, i) for i in 1:n]
    return TensorCrossInterpolation.TensorTrain(site_tensors)
end

end # module Tensor4allTCIExt
