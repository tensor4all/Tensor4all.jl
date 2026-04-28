module TensorNetworks

using Libdl
import LinearAlgebra
import LinearAlgebra: norm
import Random
import Random: AbstractRNG
import ScopedValues
import ..Tensor4all: dag, contract, fixinds, suminds, projectinds
using ..Tensor4all: BackendUnavailableError, Index, Tensor, TensorHandle, StructuredTensorStorage, SkeletonNotImplemented, _backend_handle_ptr, _copy_structured_storage, _normalize_tags, _structured_storage_from_tensor, _tensor_from_backend_handle, commoninds, copy_data, delta, dim, hastag, id, inds, plev, prime, rank, replaceinds!, require_backend, tags
import ..SimpleTT

const _LINK_TAG = "Link"

export TensorTrain, LinearOperator, SvdTruncationPolicy
export invalidate_canonical!, replaceblock!, insert_site!, delete_site!
export default_svd_policy, set_default_svd_policy!, with_svd_policy
export add, dag, dot, inner, dist
export linkind, linkinds, linkdims, siteinds
export norm
export orthogonalize, truncate
export to_dense, evaluate, evaluate!, random_tt
export TensorTrainEvaluator, TensorTrainEvalWorkspace
export fixinds, suminds, projectinds
export identity_link_tensor, insert_identity!
export set_input_space!, set_output_space!, set_iospaces!, apply, linsolve
export findsite, findsites, findallsiteinds_by_tag, findallsites_by_tag
export replace_siteinds!, replace_siteinds, replace_siteinds_shared, replace_siteinds_part!
export insert_operator_identity!, delete_operator_site!, delete_operator_sites!
export permute_operator_sites!, replace_operator_input_indices!, replace_operator_output_indices!
export rearrange_siteinds, makesitediagonal, extractdiagonal, matchsiteinds
export fuse_to, split_to, swap_site_indices, restructure_to
export save_as_mps, load_tt

include("TensorNetworks/types.jl")
include("TensorNetworks/operator_spaces.jl")
include("TensorNetworks/site_helpers.jl")
include("TensorNetworks/matchsiteinds.jl")
include("TensorNetworks/operator_canonical.jl")
include("TensorNetworks/operator_mutations.jl")
include("TensorNetworks/transforms.jl")
include("TensorNetworks/index_ops.jl")
include("TensorNetworks/identity_helpers.jl")
include("TensorNetworks/backend/capi.jl")
include("TensorNetworks/truncation_policy.jl")
include("TensorNetworks/backend/tensors.jl")
include("TensorNetworks/backend/treetn.jl")
include("TensorNetworks/backend/treetn_queries.jl")
include("TensorNetworks/backend/treetn_dense.jl")
include("TensorNetworks/backend/apply.jl")
include("TensorNetworks/backend/treetn_contract.jl")
include("TensorNetworks/backend/treetn_evaluate.jl")
include("TensorNetworks/evaluator.jl")
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
