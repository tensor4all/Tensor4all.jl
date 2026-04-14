using Test
using Tensor4all

@testset "Legacy smoke" begin
    i = Tensor4all.Index(2; tags=["Site", "n=1"])
    t = Tensor4all.Tensor(ones(2), [i])
    @test Tensor4all.rank(t) == 1

    descriptor = Tensor4all.affine_transform(matrix=reshape([1.0], 1, 1), shift=[0.0])
    @test !isa(descriptor, Module)
    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.materialize_transform(descriptor)
end
