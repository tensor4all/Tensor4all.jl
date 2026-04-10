using Test
using Tensor4all

@testset "Quantics transform metadata" begin
    shift = Tensor4all.shift_transform(; offsets=(x=1,))
    affine = Tensor4all.affine_transform(; matrix=[1.0 0.0; 0.0 1.0], shift=[0.0, 0.0])
    options = Tensor4all.QTCIOptions()

    @test shift.kind == :shift
    @test affine.kind == :affine
    @test options.max_rank == 64
    @test_throws Tensor4all.SkeletonNotImplemented Tensor4all.materialize_transform(shift)
end
