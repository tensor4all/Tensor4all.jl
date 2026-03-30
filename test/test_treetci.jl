using Tensor4all.TreeTCI
using Tensor4all.TreeTN: TreeTensorNetwork
using Test

@testset "TreeTCI" begin
    @testset "TreeTciGraph" begin
        # Linear chain: 0-1-2-3
        graph = TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])
        @test graph.n_sites == 4
        @test graph.ptr != C_NULL

        # Star graph: 0 at center
        graph_star = TreeTciGraph(4, [(0, 1), (0, 2), (0, 3)])
        @test graph_star.n_sites == 4

        # Invalid: disconnected
        @test_throws ErrorException TreeTciGraph(4, [(0, 1), (2, 3)])
    end

    @testset "Stateful API" begin
        n_sites = 7
        local_dims = fill(2, n_sites)
        edges = [(0, i) for i in 1:6]  # star graph
        graph = TreeTciGraph(n_sites, edges)

        function f_batch(batch)
            n_pts = size(batch, 2)
            results = Vector{Float64}(undef, n_pts)
            for j in 1:n_pts
                val = 1.0
                for i in 1:size(batch, 1)
                    val *= (batch[i, j] + 1.0)
                end
                results[j] = val
            end
            return results
        end

        tci = SimpleTreeTci(local_dims, graph)
        add_global_pivots!(tci, [zeros(Int, n_sites)])

        for _ in 1:4
            sweep!(tci, f_batch; tolerance=1e-12)
        end

        @test max_bond_error(tci) < 1e-10
        @test max_rank(tci) >= 1
        @test max_sample_value(tci) > 0.0

        bd = bond_dims(tci)
        @test length(bd) == 6
        @test all(d -> d >= 1, bd)

        ttn = to_treetn(tci, f_batch)
        @test ttn isa TreeTensorNetwork
    end
end
