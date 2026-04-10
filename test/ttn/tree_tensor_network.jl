using Test
using Tensor4all

@testset "TreeTensorNetwork skeleton" begin
    s1 = Tensor4all.Index(2; tags=["x1"])
    s2 = Tensor4all.Index(2; tags=["x2"])
    s3 = Tensor4all.Index(2; tags=["x3"])
    l12 = Tensor4all.Index(3; tags=["l12"])
    l23 = Tensor4all.Index(3; tags=["l23"])

    t1 = Tensor4all.Tensor(rand(2, 3), [s1, l12])
    t2 = Tensor4all.Tensor(rand(3, 2, 3), [l12, s2, l23])
    t3 = Tensor4all.Tensor(rand(3, 2), [l23, s3])

    tt = Tensor4all.TreeTensorNetwork(
        Dict(1 => t1, 2 => t2, 3 => t3);
        adjacency=Dict(1 => [2], 2 => [1, 3], 3 => [2]),
        siteinds=Dict(1 => [s1], 2 => [s2], 3 => [s3]),
        linkinds=Dict((1, 2) => l12, (2, 3) => l23),
    )

    @test Tensor4all.TensorTrain === Tensor4all.TreeTensorNetwork{Int}
    @test Tensor4all.MPS === Tensor4all.TensorTrain
    @test Tensor4all.MPO === Tensor4all.TensorTrain
    @test Tensor4all.vertices(tt) == [1, 2, 3]
    @test Tensor4all.neighbors(tt, 2) == [1, 3]
    @test Tensor4all.siteinds(tt, 2) == [s2]
    @test Tensor4all.linkind(tt, 1, 2) == l12
    @test Tensor4all.is_chain(tt)
    @test Tensor4all.is_mps_like(tt)
    @test !Tensor4all.is_mpo_like(tt)

    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.norm(tt)
    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.to_dense(tt)
end
