using Test
using Tensor4all

@testset "Tensor4all skeleton phase" begin
    @test Tensor4all.SKELETON_PHASE === true

    err = Tensor4all.SkeletonPhaseError("skeleton placeholder")
    @test err isa Exception
    @test sprint(showerror, err) == "skeleton placeholder"
end
