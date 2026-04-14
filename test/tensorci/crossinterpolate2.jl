@testset "TensorCI crossinterpolate2 boundary" begin
    f(v) = Float64(v[1])

    tt = Tensor4all.TensorCI.crossinterpolate2(Float64, f, [2]; tolerance=1e-12)

    @test tt isa Tensor4all.SimpleTT.TensorTrain{Float64,3}
end
