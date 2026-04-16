using Test
using Tensor4all

include("api/skeleton_alignment.jl")
include("core/tensor_arithmetic.jl")
include("core/tensor_contract.jl")
include("tensornetworks/tensortrain.jl")
include("tensornetworks/index_queries.jl")
include("tensornetworks/matchsiteinds.jl")
include("tensornetworks/apply.jl")
include("tensornetworks/transform_helpers.jl")
include("tensornetworks/skeleton_surface.jl")
include("simplett/surface.jl")
include("simplett/compress.jl")
include("simplett/contraction.jl")
include("tensorci/surface.jl")
include("tensorci/crossinterpolate2.jl")
include("quanticsgrids/surface.jl")
include("quanticstci/surface.jl")
if get(ENV, "T4A_SKIP_HDF5_TESTS", "0") != "1" && Base.find_package("HDF5") !== nothing
    include("extensions/hdf5_roundtrip.jl")
end
include("quanticstransform/surface.jl")
