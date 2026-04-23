using Test
using Tensor4all

const TN_INDEX_OPS = Tensor4all.TensorNetworks

@testset "TensorTrain index fixing summation and projection" begin
    s1 = Index(2; tags=["x", "x=1"])
    s2 = Index(3; tags=["x", "x=2"])
    link = Index(2; tags=["Link", "l=1"])
    tt = TN_INDEX_OPS.TensorTrain(
        Tensor[
            Tensor(reshape(collect(1.0:4.0), 2, 2), [s1, link]),
            Tensor(reshape(collect(1.0:6.0), 2, 3), [link, s2]),
        ],
        1,
        2,
    )
    dense = TN_INDEX_OPS.to_dense(tt)

    fixed = TN_INDEX_OPS.fixinds(tt, s1 => 2)
    expected_fixed = Tensor4all.fixinds(dense, s1 => 2)
    @test Array(TN_INDEX_OPS.to_dense(fixed), s2) ≈ Array(expected_fixed, s2)
    @test (fixed.llim, fixed.rlim) == (0, 2)

    summed = TN_INDEX_OPS.suminds(tt, s1)
    expected_summed = Tensor4all.suminds(dense, s1)
    @test Array(TN_INDEX_OPS.to_dense(summed), s2) ≈ Array(expected_summed, s2)

    projected = TN_INDEX_OPS.projectinds(tt, s1 => [2])
    projected_dense = TN_INDEX_OPS.to_dense(projected)
    expected_projected = Tensor4all.projectinds(dense, s1 => [2])
    @test dims(projected_dense) == dims(expected_projected)
    @test Array(projected_dense, inds(projected_dense)...) ≈ Array(expected_projected, inds(expected_projected)...)
end
