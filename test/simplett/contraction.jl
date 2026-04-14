using Test
using Tensor4all

function dense1(tt)
    return tt.sitetensors[1]
end

@testset "SimpleTT contraction" begin
    a = Tensor4all.SimpleTT.TensorTrain([
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2, 1),
    ])
    b = Tensor4all.SimpleTT.TensorTrain([
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2, 1),
    ])

    naive = Tensor4all.SimpleTT.contract(a, b; algorithm=:naive)
    @test size(naive.sitetensors[1]) == (1, 2, 2, 1)
    @test dense1(naive) == dense1(a)

    mpo = Tensor4all.SimpleTT.TensorTrain([
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2, 1),
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2, 1),
    ])
    for method in (:LU, :SVD)
        zipup = Tensor4all.SimpleTT.contract(mpo, mpo; algorithm=:zipup, method=method)
        @test length(zipup) == 2
        @test size(zipup.sitetensors[1]) == (1, 2, 2, 1)
        @test size(zipup.sitetensors[2]) == (1, 2, 2, 1)
        @test isapprox(zipup.sitetensors[1], mpo.sitetensors[1]; atol=1e-12, rtol=1e-12)
        @test isapprox(zipup.sitetensors[2], mpo.sitetensors[2]; atol=1e-12, rtol=1e-12)
    end

    @test_throws ArgumentError Tensor4all.SimpleTT.contract(
        mpo,
        mpo;
        algorithm=:zipup,
        unexpected=:kw,
    )
end
