@testset "TensorCI crossinterpolate2 boundary" begin
    f1(v) = Float64(v[1])
    f2(v) = Float64(v[1] == 1 && v[2] == 1)

    @test_throws ArgumentError Tensor4all.TensorCI.crossinterpolate2(
        Float64,
        f1,
        [2];
        tolerance=1e-12,
    )

    tci = Tensor4all.TensorCI.crossinterpolate2(
        Float64,
        f2,
        [2, 2];
        tolerance=1e-12,
        maxbonddim=2,
    )

    @test tci isa Tensor4all.TensorCI.TensorCI2{Float64}
    @test length(tci) == 2

    tt = Tensor4all.SimpleTT.TensorTrain(tci)
    @test tt isa Tensor4all.SimpleTT.TensorTrain{Float64,3}
    @test length(tt) == 2
    @test size(tt.sitetensors[1]) == (1, 2, 1)
    @test size(tt.sitetensors[2]) == (1, 2, 1)
    @test vec(tt.sitetensors[1]) == [1.0, 0.0]
    @test vec(tt.sitetensors[2]) == [1.0, 0.0]

    tci_with_pivots = Tensor4all.TensorCI.crossinterpolate2(
        Float64,
        f2,
        [2, 2],
        [[1, 1]];
        tolerance=1e-12,
        maxbonddim=2,
    )

    @test tci_with_pivots isa Tensor4all.TensorCI.TensorCI2{Float64}
    @test Tensor4all.SimpleTT.TensorTrain(tci_with_pivots) isa Tensor4all.SimpleTT.TensorTrain{Float64,3}
end
