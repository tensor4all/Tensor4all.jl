using Test
using Tensor4all

@testset "SimpleTT surface" begin
    @test isdefined(Tensor4all, :SimpleTT)
    @test isdefined(Tensor4all.SimpleTT, :TensorTrain)

    tt = Tensor4all.SimpleTT.TensorTrain([ones(1, 2, 1)])
    @test length(tt) == 1
    @test isa(tt, Tensor4all.SimpleTT.TensorTrain{Float64,3})
    @test tt.sitetensors == [ones(1, 2, 1)]
end
