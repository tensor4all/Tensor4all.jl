module TensorNetworks

using Libdl
import LinearAlgebra
import LinearAlgebra: norm
import ..Tensor4all: dag
using ..Tensor4all: BackendUnavailableError, Index, Tensor, SkeletonNotImplemented, commoninds, dim, hastag, id, inds, plev, prime, rank, require_backend, tags

const _LINK_TAG = "Link"

export TensorTrain, LinearOperator
export add, dag, dot, inner, dist
export linkinds, linkdims, siteinds
export norm
export set_input_space!, set_output_space!, set_iospaces!, apply
export findsite, findsites, findallsiteinds_by_tag, findallsites_by_tag
export replace_siteinds!, replace_siteinds, replace_siteinds_part!
export rearrange_siteinds, makesitediagonal, extractdiagonal, matchsiteinds
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
include("TensorNetworks/backend/apply.jl")
include("TensorNetworks/deferred.jl")

end
