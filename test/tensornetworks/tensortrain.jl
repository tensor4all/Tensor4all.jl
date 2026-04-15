using Test
using Tensor4all

@testset "TensorNetworks.TensorTrain" begin
    i = Index(2; tags=["i"])
    t = Tensor(ones(2), [i])
    tt = Tensor4all.TensorNetworks.TensorTrain([t], 0, 2)

    @test length(tt.data) == 1
    @test tt.llim == 0
    @test tt.rlim == 2
    @test length(tt) == 1
end
