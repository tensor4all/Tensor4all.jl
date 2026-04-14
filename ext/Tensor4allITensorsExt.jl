"""
    Tensor4allITensorsExt

Placeholder extension module for the skeleton-review reset.

The real ITensors compatibility layer is intentionally deferred until the new
Julia frontend API has been reviewed and restubbed.
"""
module Tensor4allITensorsExt

using Tensor4all
using ITensors

to_itensor(::Tensor4all.Index) =
    throw(Tensor4all.SkeletonNotImplemented(:to_itensor, :extensions))

from_itensor(::ITensors.Index) =
    throw(Tensor4all.SkeletonNotImplemented(:from_itensor, :extensions))

end # module
