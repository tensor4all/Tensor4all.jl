"""
    Tensor4all

`Tensor4all.jl` is an implementation-phase Julia frontend over `tensor4all-rs`.

Backend-backed core types are available today, while some higher-level tensor
network operations are still being filled in.
"""
module Tensor4all

using Libdl
import QuanticsGrids as UpstreamQuanticsGrids
import QuanticsTCI as UpstreamQuanticsTCI

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/CAPI.jl")
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
