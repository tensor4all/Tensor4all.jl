using Test
using Tensor4all

@testset "TensorNetworks surface" begin
    @test isdefined(Tensor4all, :TensorNetworks)
    @test isdefined(Tensor4all.TensorNetworks, :TensorTrain)
    @test :TensorTrain ∉ names(Tensor4all)

    tt = Tensor4all.TensorNetworks.TensorTrain([ones(1, 2, 1), ones(2, 2, 1)])
    @test length(tt) == 2
    @test isa(tt.sitetensors, Vector{Array{Float64,3}})
end
