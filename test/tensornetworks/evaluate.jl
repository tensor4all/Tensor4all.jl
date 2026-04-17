using Test
using Tensor4all

const TN_EVAL = Tensor4all.TensorNetworks

function _eval_two_site_mps()
    s1 = Index(2; tags=["s", "s=1"])
    s2 = Index(2; tags=["s", "s=2"])
    link = Index(2; tags=["L", "l=1"])
    tt = TN_EVAL.TensorTrain([
        Tensor([1.0 0.5; 0.0 1.0], [s1, link]),
        Tensor([2.0 0.0; 0.0 -1.0], [link, s2]),
    ])
    return tt, [s1, s2]
end

@testset "TensorTrain evaluate" begin
    @testset "single-point real" begin
        tt, sites = _eval_two_site_mps()
        dense = TN_EVAL.to_dense(tt)
        # tt has dense form; pick (s1=1, s2=1) → dense[1,1]
        v = TN_EVAL.evaluate(tt, sites, [1, 1])
        @test real(v) ≈ Array(dense, sites...)[1, 1]
        @test imag(v) ≈ 0.0 atol=1e-12
    end

    @testset "single-point matches dense at every grid point" begin
        tt, sites = _eval_two_site_mps()
        dense = Array(TN_EVAL.to_dense(tt), sites...)
        for i in 1:2, j in 1:2
            v = TN_EVAL.evaluate(tt, sites, [i, j])
            @test real(v) ≈ dense[i, j]
        end
    end

    @testset "multi-point matrix" begin
        tt, sites = _eval_two_site_mps()
        dense = Array(TN_EVAL.to_dense(tt), sites...)
        # 4 points covering the grid (col-major: each column is one point)
        values = [1 1 2 2;
                  1 2 1 2]
        result = TN_EVAL.evaluate(tt, sites, values)
        @test length(result) == 4
        @test real(result[1]) ≈ dense[1, 1]
        @test real(result[2]) ≈ dense[1, 2]
        @test real(result[3]) ≈ dense[2, 1]
        @test real(result[4]) ≈ dense[2, 2]
    end

    @testset "complex scalar value" begin
        s1 = Index(2; tags=["s", "s=1"])
        s2 = Index(2; tags=["s", "s=2"])
        link = Index(2; tags=["L", "l=1"])
        tt = TN_EVAL.TensorTrain([
            Tensor(ComplexF64[1.0+0im 0.5; 0.0 1.0im], [s1, link]),
            Tensor(ComplexF64[2.0 0.0; 0.0 1.0+1.0im], [link, s2]),
        ])
        dense = Array(TN_EVAL.to_dense(tt), [s1, s2]...)
        v = TN_EVAL.evaluate(tt, [s1, s2], [2, 2])
        @test v ≈ dense[2, 2]
    end

    @testset "argument validation" begin
        tt, sites = _eval_two_site_mps()
        empty_tt = TN_EVAL.TensorTrain(Tensor[])
        @test_throws ArgumentError TN_EVAL.evaluate(empty_tt, sites, [1, 1])
        @test_throws ArgumentError TN_EVAL.evaluate(tt, Index[], [1, 1])
        @test_throws DimensionMismatch TN_EVAL.evaluate(tt, sites, [1, 1, 1])     # too many values
        @test_throws DimensionMismatch TN_EVAL.evaluate(tt, sites, reshape([1, 1, 1, 1], 1, 4))  # wrong row count
        @test_throws ArgumentError TN_EVAL.evaluate(tt, sites, [0, 1])            # 0 is out of 1-based range
        @test_throws ArgumentError TN_EVAL.evaluate(tt, sites, [3, 1])            # exceeds dim
    end
end
