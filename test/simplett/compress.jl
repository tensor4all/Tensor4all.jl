using Test
using Tensor4all

function dense2(tt)
    left = reshape(tt.sitetensors[1], prod(size(tt.sitetensors[1])[1:end-1]), size(tt.sitetensors[1])[end])
    right = reshape(tt.sitetensors[2], size(tt.sitetensors[2], 1), prod(size(tt.sitetensors[2])[2:end]))
    return left * right
end

@testset "SimpleTT compress" begin
    tt = Tensor4all.SimpleTT.TensorTrain([
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([1.0, 0.0, 0.0, 1e-13], 2, 2, 1),
    ])
    dense_before = dense2(tt)

    Tensor4all.SimpleTT.compress!(tt, :SVD; tolerance=1e-12)

    @test size(tt.sitetensors[1]) == (1, 2, 1)
    @test size(tt.sitetensors[2]) == (1, 2, 1)
    @test isapprox(dense2(tt), dense_before; atol=1e-12, rtol=1e-12)
end

@testset "SimpleTT factorize smoke" begin
    A = [1.0 0.0; 0.0 1e-13]

    left_lu, right_lu, rank_lu = Tensor4all.SimpleTT._factorize(
        A,
        :LU;
        tolerance=1e-12,
        maxbonddim=1,
        leftorthogonal=false,
    )
    @test rank_lu == 1
    @test size(left_lu) == (2, 1)
    @test size(right_lu) == (1, 2)

    left_ci, right_ci, rank_ci = Tensor4all.SimpleTT._factorize(
        A,
        :CI;
        tolerance=1e-12,
        maxbonddim=1,
        leftorthogonal=false,
    )
    @test rank_ci == 1
    @test size(left_ci) == (2, 1)
    @test size(right_ci) == (1, 2)
end
