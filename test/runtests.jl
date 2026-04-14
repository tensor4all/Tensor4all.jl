using Test
using Tensor4all

include("api/skeleton_alignment.jl")
include("tensornetworks/tensortrain.jl")
include("simplett/surface.jl")
include("simplett/compress.jl")
include("simplett/contraction.jl")
include("tensorci/surface.jl")
include("tensorci/crossinterpolate2.jl")
if get(ENV, "T4A_SKIP_HDF5_TESTS", "0") != "1" && Base.find_package("HDF5") !== nothing
    include("extensions/hdf5_roundtrip.jl")
end
include("quanticstransform/surface.jl")
