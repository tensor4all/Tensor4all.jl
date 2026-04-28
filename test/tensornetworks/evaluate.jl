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

@testset "TensorTrainEvaluator" begin
    @testset "scalar one-site train" begin
        s = Index(3; tags=["s"])
        tt = TN_EVAL.TensorTrain([Tensor([2.0, 3.0, 5.0], [s])])
        ev = TN_EVAL.TensorTrainEvaluator(tt)
        ws = TN_EVAL.TensorTrainEvalWorkspace(ev)

        @test TN_EVAL.evaluate!(ws, ev, [[2]]) == 3.0
        @test TN_EVAL.evaluate(ev, [[3]]) == 5.0
        @test TN_EVAL.evaluate!(ws, ev, [s], [1]) == TN_EVAL.evaluate(tt, [s], [1])
    end

    @testset "MPS-like chain matches backend evaluate" begin
        tt, sites = _eval_two_site_mps()
        ev = TN_EVAL.TensorTrainEvaluator(tt)
        ws = TN_EVAL.TensorTrainEvalWorkspace(ev)

        for i in 1:2, j in 1:2
            @test TN_EVAL.evaluate!(ws, ev, [[i], [j]]) ≈ TN_EVAL.evaluate(tt, sites, [i, j])
            @test TN_EVAL.evaluate(ev, sites, [i, j]) ≈ TN_EVAL.evaluate(tt, sites, [i, j])
        end
    end

    @testset "grouped site indices match dense materialization" begin
        x1 = Index(2; tags=["x1"])
        y1 = Index(2; tags=["y1"])
        x2 = Index(2; tags=["x2"])
        y2 = Index(2; tags=["y2"])
        link = Index(2; tags=["link"])
        a = reshape(collect(1.0:8.0), 2, 2, 2)
        b = reshape(collect(1.0:8.0) ./ 10, 2, 2, 2)
        tt = TN_EVAL.TensorTrain([
            Tensor(a, [link, x1, y1]),
            Tensor(b, [link, x2, y2]),
        ])
        sites = [x1, y1, x2, y2]
        dense = Array(TN_EVAL.to_dense(tt), sites...)
        ev = TN_EVAL.TensorTrainEvaluator(tt)
        ws = TN_EVAL.TensorTrainEvalWorkspace(ev)

        @test ev.site_groups == [[x1, y1], [x2, y2]]
        @test TN_EVAL.evaluate!(ws, ev, [[2, 1], [1, 2]]) ≈ dense[2, 1, 1, 2]
        @test TN_EVAL.evaluate!(ws, ev, sites, [2, 1, 1, 2]) ≈ dense[2, 1, 1, 2]
    end

    @testset "ComplexF64 evaluation does not conjugate blocks" begin
        s1 = Index(2; tags=["s1"])
        s2 = Index(2; tags=["s2"])
        link = Index(2; tags=["link"])
        tt = TN_EVAL.TensorTrain([
            Tensor(ComplexF64[1+im 2-im; 3+2im 4-3im], [link, s1]),
            Tensor(ComplexF64[2-im 3+im; 5+2im 7-4im], [link, s2]),
        ])
        ev = TN_EVAL.TensorTrainEvaluator(tt)
        ws = TN_EVAL.TensorTrainEvalWorkspace(ev)
        dense = Array(TN_EVAL.to_dense(tt), s1, s2)

        @test TN_EVAL.evaluate!(ws, ev, [[2], [1]]) ≈ dense[2, 1]
        @test TN_EVAL.evaluate!(ws, ev, [[2], [1]]) != conj(dense[2, 1])
    end

    @testset "validation" begin
        tt, sites = _eval_two_site_mps()
        ev = TN_EVAL.TensorTrainEvaluator(tt)
        ws = TN_EVAL.TensorTrainEvalWorkspace(ev)
        duplicate_sites = [sites[1], sites[1]]
        missing_site = Index(2; tags=["missing"])
        same_id_different_tags = Index(dim(sites[1]); id=id(sites[1]), tags=["other"])

        @test_throws ArgumentError TN_EVAL.TensorTrainEvaluator(TN_EVAL.TensorTrain(Tensor[]))
        @test_throws DimensionMismatch TN_EVAL.evaluate!(ws, ev, [[1]])
        @test_throws DimensionMismatch TN_EVAL.evaluate!(ws, ev, [[1], [1, 1]])
        @test_throws ArgumentError TN_EVAL.evaluate!(ws, ev, [[0], [1]])
        @test_throws ArgumentError TN_EVAL.evaluate!(ws, ev, [[3], [1]])
        @test_throws ArgumentError TN_EVAL.evaluate!(ws, ev, duplicate_sites, [1, 1])
        @test_throws ArgumentError TN_EVAL.evaluate!(ws, ev, [sites[1], missing_site], [1, 1])
        @test_throws ArgumentError TN_EVAL.evaluate!(ws, ev, [same_id_different_tags, sites[2]], [1, 1])
    end

    @testset "grouped hot path avoids per-call heap allocation" begin
        tt, = _eval_two_site_mps()
        ev = TN_EVAL.TensorTrainEvaluator(tt)
        ws = TN_EVAL.TensorTrainEvalWorkspace(ev)
        grouped_values = [[1], [2]]
        TN_EVAL.evaluate!(ws, ev, grouped_values)

        @test @allocated(TN_EVAL.evaluate!(ws, ev, grouped_values)) <= 64
    end
end
