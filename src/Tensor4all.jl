"""
    Tensor4all

`Tensor4all.jl` is in an API-skeleton review phase.

The previous implementation has been intentionally removed so the package can be
rebuilt against the design documents in `docs/design/` without stale public
surface area leaking into the rework. Importing the package is expected to work,
but the real backend-facing types and operations are deferred to a later phase.
"""
module Tensor4all

using Libdl

const SKELETON_PHASE = true

include("Core/Errors.jl")
include("Core/Backend.jl")
include("Core/Index.jl")
include("Core/Tensor.jl")
include("TreeTN/TreeTensorNetwork.jl")

export SKELETON_PHASE
export SkeletonPhaseError, SkeletonNotImplemented, BackendUnavailableError
export backend_library_path, require_backend
export Index, dim, id, tags, plev, hastag
export sim, prime, noprime, setprime
export replaceind, replaceinds, commoninds, uniqueinds
export Tensor, inds, rank, dims, swapinds, contract
export TreeTensorNetwork, TensorTrain, MPS, MPO
export vertices, neighbors, siteinds, linkind
export is_chain, is_mps_like, is_mpo_like
export orthogonalize!, truncate!, inner, norm, to_dense, evaluate

end
