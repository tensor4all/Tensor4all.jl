"""
    Tensor4all

`Tensor4all.jl` is in the implementation phase of the restored Julia frontend.

The public object model is Julia-owned and centered on the restored module
split documented under `docs/design/`. Some backend-backed execution paths are
still deferred, but the main chain-facing surface is implemented and testable.
"""
module Tensor4all

using Libdl
import QuanticsGrids as UpstreamQuanticsGrids
import QuanticsTCI as UpstreamQuanticsTCI

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/Index.jl")
include("Core/Tensor.jl")

include("SimpleTT.jl")
include("TensorNetworks.jl")
include("TensorCI.jl")
include("QuanticsGrids.jl")
include("QuanticsTCI.jl")
include("QuanticsTransform.jl")

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend
export Index, dim, id, tags, plev, hastag
export sim, prime, noprime, setprime
export replaceind, replaceinds, commoninds, uniqueinds
export Tensor, inds, rank, dims, swapinds, contract
export TensorNetworks, SimpleTT, TensorCI, QuanticsGrids, QuanticsTCI, QuanticsTransform

end
