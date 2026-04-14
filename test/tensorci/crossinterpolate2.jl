@testset "TensorCI crossinterpolate2 boundary" begin
    f1(v) = Float64(v[1])

    tt = Tensor4all.TensorCI.crossinterpolate2(Float64, f1, [2]; tolerance=1e-12)

    @test tt isa Tensor4all.SimpleTT.TensorTrain{Float64,3}
    @test length(tt) == 1
    @test size(tt.sitetensors[1]) == (1, 2, 1)

    f2(v) = Float64(v[1] == 1 && v[2] == 1)
    tt2 = Tensor4all.TensorCI.crossinterpolate2(Float64, f2, [2, 2]; tolerance=1e-12, maxbonddim=2)

    @test tt2 isa Tensor4all.SimpleTT.TensorTrain{Float64,3}
    @test length(tt2) == 2
    @test size(tt2.sitetensors[1]) == (1, 2, 1)
    @test size(tt2.sitetensors[2]) == (1, 2, 1)
end
