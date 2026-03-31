using Test
using Tensor4all.TreeTCI
import Tensor4all: evaluate, maxbonderror, maxrank
import Tensor4all.TreeTCI: maxsamplevalue,
    bonddims, to_treetn, sweep!, add_global_pivots!

@testset "TreeTCI" begin
    @testset "1-indexed crossinterpolate2 - linear chain" begin
        # Linear chain: 0-1-2-3
        graph = TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])

        # Batch evaluation function: product of indices (1-based)
        # f(batch) where batch is (n_sites, n_points) with 1-based indices
        function f_product(batch)
            n_sites, n_pts = size(batch)
            [prod(Float64(batch[i, j]) for i in 1:n_sites) for j in 1:n_pts]
        end

        tci, ranks, errors = crossinterpolate2(Float64, f_product, [3, 3, 3, 3], graph;
            tolerance=1e-10, maxiter=20,
            initialpivots=[ones(Int, 4)])

        @test tci isa SimpleTreeTci{Float64}
        @test length(ranks) > 0
        @test length(errors) > 0
        @test ranks isa Vector{Int}
        @test errors isa Vector{Float64}

        @testset "state inspection" begin
            @test maxrank(tci) isa Int
            @test maxrank(tci) >= 1
            @test maxbonderror(tci) isa Float64
            @test maxsamplevalue(tci) isa Float64
            @test maxsamplevalue(tci) > 0.0

            bd = bonddims(tci)
            @test bd isa Vector{Int}
            @test length(bd) == 3  # n_edges = n_sites - 1 for linear chain
        end

        @testset "materialize to TreeTN" begin
            ttn = to_treetn(tci, f_product)
            @test ttn !== nothing

            @testset "evaluate with 1-based indices" begin
                # f(1,1,1,1) = 1*1*1*1 = 1.0
                val = evaluate(ttn, [1, 1, 1, 1])
                @test val ≈ 1.0 atol=1e-8

                # f(2,3,1,2) = 2*3*1*2 = 12.0
                val = evaluate(ttn, [2, 3, 1, 2])
                @test val ≈ 12.0 atol=1e-6

                # f(3,3,3,3) = 3*3*3*3 = 81.0
                val = evaluate(ttn, [3, 3, 3, 3])
                @test val ≈ 81.0 atol=1e-6
            end

            @testset "batch evaluate with 1-based indices" begin
                batch = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
                vals = evaluate(ttn, batch)
                @test vals[1] ≈ 1.0 atol=1e-8
                @test vals[2] ≈ 16.0 atol=1e-6
                @test vals[3] ≈ 81.0 atol=1e-6
            end

            @testset "matrix evaluate with 1-based indices" begin
                # 4 sites, 2 points
                batch_mat = [1 2; 1 2; 1 2; 1 2]
                vals = evaluate(ttn, batch_mat)
                @test vals[1] ≈ 1.0 atol=1e-8
                @test vals[2] ≈ 16.0 atol=1e-6
            end
        end
    end

    @testset "1-indexed crossinterpolate2 - sum function" begin
        graph = TreeTciGraph(3, [(0, 1), (1, 2)])

        # Sum function (1-based indices)
        function f_sum(batch)
            n_sites, n_pts = size(batch)
            [Base.sum(Float64(batch[i, j]) for i in 1:n_sites) for j in 1:n_pts]
        end

        tci, ranks, errors = crossinterpolate2(Float64, f_sum, [4, 4, 4], graph;
            tolerance=1e-10, maxiter=20)

        @test tci isa SimpleTreeTci{Float64}

        ttn = to_treetn(tci, f_sum)

        # f(1,1,1) = 1+1+1 = 3.0
        @test evaluate(ttn, [1, 1, 1]) ≈ 3.0 atol=1e-8
        # f(4,4,4) = 4+4+4 = 12.0
        @test evaluate(ttn, [4, 4, 4]) ≈ 12.0 atol=1e-6
        # f(1,2,3) = 1+2+3 = 6.0
        @test evaluate(ttn, [1, 2, 3]) ≈ 6.0 atol=1e-6
    end

    @testset "default initialpivots is ones" begin
        graph = TreeTciGraph(3, [(0, 1), (1, 2)])

        function f_const(batch)
            n_sites, n_pts = size(batch)
            fill(42.0, n_pts)
        end

        # Should work without specifying initialpivots (default is ones)
        tci, ranks, errors = crossinterpolate2(Float64, f_const, [2, 2, 2], graph;
            tolerance=1e-8, maxiter=5)

        @test tci isa SimpleTreeTci{Float64}
    end

    @testset "type inference" begin
        graph = TreeTciGraph(2, [(0, 1)])

        # Float64 inference
        f_real(batch) = fill(1.0, size(batch, 2))
        tci, _, _ = crossinterpolate2(f_real, [2, 2], graph; maxiter=3)
        @test tci isa SimpleTreeTci{Float64}
    end

    @testset "star graph topology" begin
        # Star graph: site 0 connected to sites 1,2,3
        graph = TreeTciGraph(4, [(0, 1), (0, 2), (0, 3)])

        function f_star(batch)
            n_sites, n_pts = size(batch)
            [Float64(batch[1, j]) * Float64(batch[2, j]) + Float64(batch[3, j]) + Float64(batch[4, j])
             for j in 1:n_pts]
        end

        tci, ranks, errors = crossinterpolate2(Float64, f_star, [3, 3, 3, 3], graph;
            tolerance=1e-10, maxiter=20, initialpivots=[ones(Int, 4)])

        ttn = to_treetn(tci, f_star)

        # f(1,1,1,1) = 1*1 + 1 + 1 = 3.0
        @test evaluate(ttn, [1, 1, 1, 1]) ≈ 3.0 atol=1e-6
        # f(2,3,1,1) = 2*3 + 1 + 1 = 8.0
        @test evaluate(ttn, [2, 3, 1, 1]) ≈ 8.0 atol=1e-6
    end
end
