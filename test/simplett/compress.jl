using Test
using Tensor4all

@testset "SimpleTT compress" begin
    tt = Tensor4all.SimpleTT.TensorTrain([
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([1.0, 0.0, 0.0, 1e-13], 2, 2, 1),
    ])

    Tensor4all.SimpleTT.compress!(tt, :SVD; tolerance=1e-12)

    @test size(tt.sitetensors[1]) == (1, 2, 1)
    @test size(tt.sitetensors[2]) == (1, 2, 1)
end
