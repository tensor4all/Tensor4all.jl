using Test
using Tensor4all

@testset "TensorCI surface" begin
    @test isdefined(Tensor4all, :TensorCI)
    @test isdefined(Tensor4all.TensorCI, :crossinterpolate2)
    @test_throws ErrorException Tensor4all.TensorCI.crossinterpolate2()
end
