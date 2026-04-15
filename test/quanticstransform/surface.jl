using Test
using Tensor4all

@testset "QuanticsTransform surface" begin
    @test isdefined(Tensor4all, :QuanticsTransform)
    @test isdefined(Tensor4all.QuanticsTransform, :shift_operator)
    @test isdefined(Tensor4all.QuanticsTransform, :flip_operator)
    @test isdefined(Tensor4all.QuanticsTransform, :fourier_operator)
    @test !isdefined(Tensor4all.QuanticsTransform, :LinearOperator)

    op = Tensor4all.QuanticsTransform.shift_operator(4, 1)
    @test op isa Tensor4all.TensorNetworks.LinearOperator
    @test op.metadata.kind == :shift
    @test op.metadata.r == 4
    @test op.metadata.offset == 1
end
