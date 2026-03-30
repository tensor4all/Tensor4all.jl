using Tensor4all: TreeTCI
using Tensor4all.TreeTN: TreeTensorNetwork
using Test

@testset "TreeTCI" begin
    @testset "TreeTciGraph" begin
        # Linear chain: 0-1-2-3
        graph = TreeTCI.TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])
        @test graph.n_sites == 4
        @test graph.ptr != C_NULL

        # Star graph: 0 at center
        graph_star = TreeTCI.TreeTciGraph(4, [(0, 1), (0, 2), (0, 3)])
        @test graph_star.n_sites == 4

        # Invalid: disconnected
        @test_throws ErrorException TreeTCI.TreeTciGraph(4, [(0, 1), (2, 3)])
    end

    @testset "Stateful API" begin
        n_sites = 7
        local_dims = fill(2, n_sites)
        edges = [(0, i) for i in 1:6]  # star graph
        graph = TreeTCI.TreeTciGraph(n_sites, edges)

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

        tci = TreeTCI.SimpleTreeTci(local_dims, graph)
        TreeTCI.add_global_pivots!(tci, [zeros(Int, n_sites)])

        for _ in 1:4
            TreeTCI.sweep!(tci, f_batch; tolerance=1e-12)
        end

        @test TreeTCI.max_bond_error(tci) < 1e-10
        @test TreeTCI.max_rank(tci) >= 1
        @test TreeTCI.max_sample_value(tci) > 0.0

        bd = TreeTCI.bond_dims(tci)
        @test length(bd) == 6
        @test all(d -> d >= 1, bd)

        ttn = TreeTCI.to_treetn(tci, f_batch)
        @test ttn isa TreeTensorNetwork
    end

    @testset "High-level API" begin
        n_sites = 4
        local_dims = fill(3, n_sites)
        graph = TreeTCI.TreeTciGraph(n_sites, [(0, 1), (1, 2), (2, 3)])

        f_batch(batch) = [sum(Float64, batch[:, j]) for j in 1:size(batch, 2)]

        ttn, ranks, errors = TreeTCI.crossinterpolate_tree(
            f_batch,
            local_dims,
            graph;
            initial_pivots=[zeros(Int, n_sites)],
            tolerance=1e-10,
            max_iter=20,
        )

        @test ttn isa TreeTensorNetwork
        @test length(ranks) > 0
        @test last(errors) < 1e-8
    end

    @testset "High-level API without initial pivots" begin
        n_sites = 3
        local_dims = fill(2, n_sites)
        graph = TreeTCI.TreeTciGraph(n_sites, [(0, 1), (1, 2)])

        f_batch(batch) = [prod(batch[i, j] + 1.0 for i in 1:size(batch, 1)) for j in 1:size(batch, 2)]

        ttn, ranks, errors = TreeTCI.crossinterpolate_tree(f_batch, local_dims, graph)

        @test ttn isa TreeTensorNetwork
        @test length(ranks) > 0
    end
end
