"""
    Tensor4all

`Tensor4all.jl` is in the implementation phase of the restored Julia frontend.

The public object model is Julia-owned and centered on the restored module
split documented under `docs/design/`. Some backend-backed execution paths are
still deferred, but the main chain-facing surface is implemented and testable.
"""
module Tensor4all

using Libdl
using LinearAlgebra: norm
import QuanticsGrids as UpstreamQuanticsGrids
import QuanticsTCI as UpstreamQuanticsTCI

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/Index.jl")
include("Core/Tensor.jl")
include("Core/IndexOps.jl")

include("SimpleTT.jl")
include("TensorNetworks.jl")
include("ITensorCompat.jl")
include("TensorCI.jl")
include("QuanticsGrids.jl")
include("QuanticsTCI.jl")
include("QuanticsTransform/QuanticsTransform.jl")

using .TensorNetworks: add, apply, dot, inner, dist, linkdims, linkind, linkinds, siteinds, orthogonalize, truncate
using .ITensorCompat: random_itensor, random_mps, siteind

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend
export Index, dim, id, tags, plev, hasplev, hastag
export sim, prime, noprime, setprime
export replaceind, replaceinds, replaceind!, replaceinds!, commoninds, uniqueinds
export hasinds, scalar
export add, apply
export dag
export dot, inner, dist
export linkind, linkinds, linkdims, siteind, siteinds
export norm
export orthogonalize, truncate
export Tensor, ITensor, inds, rank, dims, swapinds, contract
export delta, isdiag, structured_storage_info, structured_payload
export onehot, fixinds, suminds, projectinds
export svd, qr
export random_itensor, random_mps
export TensorNetworks, ITensorCompat, SimpleTT, TensorCI, QuanticsGrids, QuanticsTCI, QuanticsTransform

end
