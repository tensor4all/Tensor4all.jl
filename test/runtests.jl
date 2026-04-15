using Test
using Tensor4all

include("core/bootstrap.jl")
include("core/index.jl")
include("core/tensor.jl")
include("api/skeleton_alignment.jl")
include("tensornetworks/tensortrain.jl")
include("tensornetworks/skeleton_surface.jl")
include("simplett/surface.jl")
include("simplett/compress.jl")
include("simplett/contraction.jl")
include("tensorci/surface.jl")
include("tensorci/crossinterpolate2.jl")
include("quanticsgrids/surface.jl")
include("quanticstci/surface.jl")
if get(ENV, "T4A_SKIP_HDF5_TESTS", "0") != "1"
    include("extensions/hdf5_roundtrip.jl")
    if Base.find_package("ITensors") !== nothing && Base.find_package("ITensorMPS") !== nothing
        include("extensions/hdf5_itensors_interop.jl")
    end
end
include("quanticstransform/surface.jl")
