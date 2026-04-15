using Test
using Tensor4all
using QuanticsGrids

@testset "Skeleton API alignment" begin
    @test isdefined(Tensor4all, :TensorNetworks)
    @test isdefined(Tensor4all, :SimpleTT)
    @test isdefined(Tensor4all, :TensorCI)
    @test isdefined(Tensor4all, :QuanticsTransform)
    @test isdefined(Tensor4all, :QuanticsTCI)

    @test !isdefined(Tensor4all, :TreeTensorNetwork)
    @test !isdefined(Tensor4all, :MPS)
    @test !isdefined(Tensor4all, :MPO)
    @test !isdefined(Tensor4all, :affine_transform)
    @test !isdefined(Tensor4all, :shift_transform)

    @test isdefined(Tensor4all, :QuanticsGrids)
    @test Tensor4all.QuanticsGrids.DiscretizedGrid === QuanticsGrids.DiscretizedGrid

    @test isdefined(Tensor4all.TensorNetworks, :TensorTrain)
    @test isdefined(Tensor4all.TensorNetworks, :LinearOperator)
    @test isdefined(Tensor4all.TensorNetworks, :apply)
    @test !isdefined(Tensor4all.QuanticsTransform, :LinearOperator)
end
