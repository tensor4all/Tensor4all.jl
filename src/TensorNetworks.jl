module TensorNetworks

using Libdl
import LinearAlgebra
import LinearAlgebra: norm
import Random
import Random: AbstractRNG
import ..Tensor4all: dag, contract
using ..Tensor4all: BackendUnavailableError, Index, Tensor, SkeletonNotImplemented, commoninds, dim, hastag, id, inds, plev, prime, rank, require_backend, tags
import ..SimpleTT

const _LINK_TAG = "Link"

export TensorTrain, LinearOperator
export add, dag, dot, inner, dist
export linkinds, linkdims, siteinds
export norm
export orthogonalize, truncate
export to_dense, evaluate, random_tt
export set_input_space!, set_output_space!, set_iospaces!, apply, linsolve
export findsite, findsites, findallsiteinds_by_tag, findallsites_by_tag
export replace_siteinds!, replace_siteinds, replace_siteinds_part!
export rearrange_siteinds, makesitediagonal, extractdiagonal, matchsiteinds
export fuse_to, split_to, swap_site_indices, restructure_to
export save_as_mps, load_tt

include("TensorNetworks/types.jl")
include("TensorNetworks/operator_spaces.jl")
include("TensorNetworks/site_helpers.jl")
include("TensorNetworks/matchsiteinds.jl")
include("TensorNetworks/transforms.jl")
include("TensorNetworks/backend/capi.jl")
include("TensorNetworks/backend/tensors.jl")
include("TensorNetworks/backend/treetn.jl")
include("TensorNetworks/backend/treetn_queries.jl")
include("TensorNetworks/backend/treetn_dense.jl")
include("TensorNetworks/backend/apply.jl")
include("TensorNetworks/backend/treetn_contract.jl")
include("TensorNetworks/backend/treetn_evaluate.jl")
include("TensorNetworks/backend/restructure/helpers.jl")
include("TensorNetworks/backend/restructure/fuse_to.jl")
include("TensorNetworks/backend/restructure/split_to.jl")
include("TensorNetworks/backend/restructure/swap_site_indices.jl")
include("TensorNetworks/backend/restructure/restructure_to.jl")
include("TensorNetworks/backend/linsolve.jl")
include("TensorNetworks/random.jl")
include("TensorNetworks/bridge.jl")
include("TensorNetworks/deferred.jl")

end
