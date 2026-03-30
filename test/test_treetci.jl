import Tensor4all
using Tensor4all: TreeTCI
using Tensor4all.TreeTN: TreeTensorNetwork
using Test

# ============================================================================
# Test helpers
# ============================================================================

"""7-site branching tree matching Rust parity tests."""
function sample_graph_7site()
    #     0
    #     |
    #     1---2
    #     |
    #     3
    #     |
    #     4
    #    / \
    #   5   6
    TreeTCI.TreeTciGraph(7, [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)])
end

"""Product function: f(idx) = prod(idx[s] + 1.0). Exact with low bond dim."""
function product_batch(batch)
    n_sites, n_pts = size(batch)
    results = Vector{Float64}(undef, n_pts)
    for j in 1:n_pts
        val = 1.0
        for i in 1:n_sites
            val *= (batch[i, j] + 1.0)
        end
        results[j] = val
    end
    results
end

"""Rational function: f(idx) = 1 / (1 + sum((idx[s]+1)^2)). Non-exact."""
function rational_batch(batch)
    n_sites, n_pts = size(batch)
    results = Vector{Float64}(undef, n_pts)
    for j in 1:n_pts
        norm_sq = 0.0
        for i in 1:n_sites
            x = Float64(batch[i, j]) + 1.0
            norm_sq += x * x
        end
        results[j] = 1.0 / (1.0 + norm_sq)
    end
    results
end

"""Complex product function matching Rust c64 parity tests."""
function complex_product_batch(batch)
    n_sites, n_pts = size(batch)
    results = Vector{ComplexF64}(undef, n_pts)
    for j in 1:n_pts
        val = ComplexF64(1.0, 0.0)
        for i in 1:n_sites
            idx = batch[i, j]
            val *= (idx + 1) + im * (2 * idx + 1)
        end
        results[j] = val
    end
    results
end

"""Evaluate TreeTN at a multi-index (0-based) by calling the batch function."""
function evaluate_product_at(idx::Vector{Int})
    val = 1.0
    for s in idx
        val *= (s + 1.0)
    end
    val
end

function evaluate_rational_at(idx::Vector{Int})
    norm_sq = sum((s + 1.0)^2 for s in idx)
    1.0 / (1.0 + norm_sq)
end

# ============================================================================
# Tests
# ============================================================================

@testset "TreeTCI" begin
    @testset "TreeTciGraph" begin
        # Linear chain
        graph = TreeTCI.TreeTciGraph(4, [(0, 1), (1, 2), (2, 3)])
        @test graph.n_sites == 4
        @test graph.ptr != C_NULL

        # Star graph
        graph_star = TreeTCI.TreeTciGraph(4, [(0, 1), (0, 2), (0, 3)])
        @test graph_star.n_sites == 4

        # Branching tree (Rust parity)
        graph7 = sample_graph_7site()
        @test graph7.n_sites == 7

        # 2-site tree
        graph2 = TreeTCI.TreeTciGraph(2, [(0, 1)])
        @test graph2.n_sites == 2

        # Invalid: disconnected
        @test_throws ErrorException TreeTCI.TreeTciGraph(4, [(0, 1), (2, 3)])
    end

    @testset "Stateful API - product function on branching tree" begin
        graph = sample_graph_7site()
        n_sites = 7
        local_dims = fill(2, n_sites)

        tci = TreeTCI.SimpleTreeTci(local_dims, graph)
        TreeTCI.add_global_pivots!(tci, [zeros(Int, n_sites)])

        for _ in 1:8
            TreeTCI.sweep!(tci, product_batch; tolerance=1e-12)
        end

        @test TreeTCI.max_bond_error(tci) < 1e-10
        @test TreeTCI.max_rank(tci) >= 1
        @test TreeTCI.max_sample_value(tci) > 0.0

        bd = TreeTCI.bond_dims(tci)
        @test length(bd) == 6  # n_edges for 7-site tree
        @test all(d -> d >= 1, bd)

        ttn = TreeTCI.to_treetn(tci, product_batch)
        @test ttn isa TreeTensorNetwork
    end

    @testset "Stateful API - complex product on branching tree" begin
        graph = sample_graph_7site()
        n_sites = 7
        local_dims = fill(2, n_sites)

        tci = TreeTCI.SimpleTreeTci{ComplexF64}(local_dims, graph)
        TreeTCI.add_global_pivots!(tci, [zeros(Int, n_sites)])

        for _ in 1:8
            TreeTCI.sweep!(tci, complex_product_batch; tolerance=1e-12, max_bond_dim=2)
        end

        @test TreeTCI.max_bond_error(tci) < 1e-12
        @test TreeTCI.max_rank(tci) == 1
        @test TreeTCI.max_sample_value(tci) > 0.0

        bd = TreeTCI.bond_dims(tci)
        @test bd == fill(1, 6)

        ttn = TreeTCI.to_treetn(tci, complex_product_batch)
        @test ttn isa TreeTensorNetwork
        @test Tensor4all.storage_kind(ttn[1]) == Tensor4all.DenseC64
        @test eltype(Tensor4all.data(ttn[1])) == ComplexF64
    end

    @testset "Stateful API - rational function convergence" begin
        graph = sample_graph_7site()
        n_sites = 7
        local_dims = fill(2, n_sites)

        tci = TreeTCI.SimpleTreeTci(local_dims, graph)
        TreeTCI.add_global_pivots!(tci, [zeros(Int, n_sites)])

        for _ in 1:10
            TreeTCI.sweep!(tci, rational_batch; tolerance=1e-8, max_bond_dim=5)
            TreeTCI.max_bond_error(tci) < 1e-8 && break
        end

        @test TreeTCI.max_bond_error(tci) < 1e-8

        # Verify bond dims respect max_bond_dim
        bd = TreeTCI.bond_dims(tci)
        @test all(d -> d <= 5, bd)
    end

    @testset "Batch callback receives n_points > 1" begin
        graph = sample_graph_7site()
        n_sites = 7
        local_dims = fill(2, n_sites)

        max_n_points_seen = Ref(0)

        function tracking_batch(batch)
            n_pts = size(batch, 2)
            max_n_points_seen[] = max(max_n_points_seen[], n_pts)
            return product_batch(batch)
        end

        tci = TreeTCI.SimpleTreeTci(local_dims, graph)
        TreeTCI.add_global_pivots!(tci, [zeros(Int, n_sites)])

        for _ in 1:4
            TreeTCI.sweep!(tci, tracking_batch; tolerance=1e-12)
        end

        # The batch callback should have been called with multiple points
        @test max_n_points_seen[] > 1
    end

    @testset "crossinterpolate2 - 2-site tree product exact" begin
        graph = TreeTCI.TreeTciGraph(2, [(0, 1)])
        local_dims = [2, 2]

        tci, ranks, errors = TreeTCI.crossinterpolate2(
            product_batch, local_dims, graph;
            initialpivots=[zeros(Int, 2)],
            tolerance=1e-12,
            maxiter=8,
            maxbonddim=2,
        )

        @test tci isa TreeTCI.SimpleTreeTci{Float64}
        @test length(errors) > 0
        @test last(errors) < 1e-10

        ttn = TreeTCI.to_treetn(tci, product_batch)
        @test ttn isa TreeTensorNetwork
    end

    @testset "crossinterpolate2 - product on branching tree" begin
        graph = sample_graph_7site()
        n_sites = 7
        local_dims = fill(2, n_sites)

        tci, ranks, errors = TreeTCI.crossinterpolate2(
            product_batch, local_dims, graph;
            initialpivots=[zeros(Int, n_sites), [1, 0, 1, 0, 1, 0, 1]],
            tolerance=1e-12,
            maxiter=8,
            maxbonddim=2,
        )

        @test tci isa TreeTCI.SimpleTreeTci{Float64}
        @test length(ranks) > 0
        @test last(errors) < 1e-10
        @test all(e -> e < 1e-10 || e == 0.0, errors)
    end

    @testset "crossinterpolate2 - complex product on branching tree" begin
        graph = sample_graph_7site()
        n_sites = 7
        local_dims = fill(2, n_sites)

        tci, ranks, errors = TreeTCI.crossinterpolate2(
            complex_product_batch, local_dims, graph;
            initialpivots=[zeros(Int, n_sites)],
            tolerance=1e-12,
            maxiter=8,
            maxbonddim=2,
        )

        @test tci isa TreeTCI.SimpleTreeTci{ComplexF64}
        @test length(ranks) > 0
        @test last(errors) < 1e-8

        ttn = TreeTCI.to_treetn(tci, complex_product_batch)
        @test ttn isa TreeTensorNetwork
        @test Tensor4all.storage_kind(ttn[1]) == Tensor4all.DenseC64
    end

    @testset "crossinterpolate2 - rational function convergence" begin
        graph = sample_graph_7site()
        n_sites = 7
        local_dims = fill(2, n_sites)

        tci, ranks, errors = TreeTCI.crossinterpolate2(
            rational_batch, local_dims, graph;
            initialpivots=[zeros(Int, n_sites)],
            tolerance=1e-8,
            maxiter=10,
            maxbonddim=5,
        )

        @test tci isa TreeTCI.SimpleTreeTci{Float64}
        @test length(ranks) > 0
        @test last(errors) < 1e-8
    end

    @testset "crossinterpolate2 - default zero pivot" begin
        n_sites = 3
        local_dims = fill(2, n_sites)
        graph = TreeTCI.TreeTciGraph(n_sites, [(0, 1), (1, 2)])

        tci, ranks, errors = TreeTCI.crossinterpolate2(
            product_batch, local_dims, graph;
            tolerance=1e-12,
        )

        @test tci isa TreeTCI.SimpleTreeTci{Float64}
        @test length(ranks) > 0
    end

    @testset "crossinterpolate2 - 4-site branching tree" begin
        graph = TreeTCI.TreeTciGraph(4, [(0, 1), (1, 2), (1, 3)])
        local_dims = fill(3, 4)

        f_batch(batch) = [sum(Float64, batch[:, j]) for j in 1:size(batch, 2)]

        tci, ranks, errors = TreeTCI.crossinterpolate2(
            f_batch, local_dims, graph;
            initialpivots=[zeros(Int, 4)],
            tolerance=1e-10,
            maxiter=20,
        )

        @test tci isa TreeTCI.SimpleTreeTci{Float64}
        @test length(ranks) > 0
        @test last(errors) < 1e-8
    end
end
