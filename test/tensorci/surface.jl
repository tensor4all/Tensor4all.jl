using Test
using Tensor4all

@testset "TensorCI surface" begin
    @test isdefined(Tensor4all, :TensorCI)
    @test isdefined(Tensor4all.TensorCI, :crossinterpolate2)
    @test isdefined(Tensor4all.TensorCI, :TensorCI2)
    @test length(Base.methods(Tensor4all.TensorCI.crossinterpolate2)) >= 1
end
