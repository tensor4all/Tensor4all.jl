using Test
using Tensor4all
using ITensors

@testset "ITensors extension skeleton" begin
    ext = Base.get_extension(Tensor4all, :Tensor4allITensorsExt)
    @test ext !== nothing
    @test_throws Tensor4all.SkeletonNotImplemented ext.to_itensor(Tensor4all.Index(2))
end
