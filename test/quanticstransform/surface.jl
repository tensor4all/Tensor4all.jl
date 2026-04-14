using Test
using Tensor4all

@testset "QuanticsTransform surface" begin
    @test isdefined(Tensor4all, :QuanticsTransform)
    @test isdefined(Tensor4all.QuanticsTransform, :LinearOperator)
    @test length(fieldnames(Tensor4all.QuanticsTransform.LinearOperator)) == 1

    op = Tensor4all.QuanticsTransform.LinearOperator(:payload)
    @test typeof(op) === Tensor4all.QuanticsTransform.LinearOperator
    @test op.payload === :payload
end
