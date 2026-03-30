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
end
