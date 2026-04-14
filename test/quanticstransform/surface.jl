using Test
using Tensor4all

@testset "QuanticsTransform surface" begin
    @test isdefined(Tensor4all, :QuanticsTransform)
    @test isdefined(Tensor4all.QuanticsTransform, :LinearOperator)
    @test length(fieldnames(Tensor4all.QuanticsTransform.LinearOperator)) == 1
end
