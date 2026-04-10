using Test
using Tensor4all
using HDF5

@testset "HDF5 extension skeleton" begin
    ext = Base.get_extension(Tensor4all, :Tensor4allHDF5Ext)
    @test ext !== nothing
    @test_throws Tensor4all.SkeletonNotImplemented ext.save_hdf5("tmp.h5", nothing)
end
