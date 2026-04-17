using Test
using Tensor4all
using LinearAlgebra: norm
using Random: MersenneTwister

const TN_RESTRUCT = Tensor4all.TensorNetworks

function _restruct_chain(sites_per_node::Vector{Vector{Index}}, link_dim::Integer=2; seed::Integer=1)
    rng = MersenneTwister(seed)
    n = length(sites_per_node)
    links = [Index(link_dim; tags=["Link", "l=$i"]) for i in 1:(n - 1)]

    tensors = Tensor[]
    for i in 1:n
        site_dims = Tuple(dim(s) for s in sites_per_node[i])
        if n == 1
            indices = sites_per_node[i]
            data = randn(rng, site_dims...)
        elseif i == 1
            indices = [sites_per_node[i]..., links[1]]
            data = randn(rng, site_dims..., dim(links[1]))
        elseif i == n
            indices = [links[i - 1], sites_per_node[i]...]
            data = randn(rng, dim(links[i - 1]), site_dims...)
        else
            indices = [links[i - 1], sites_per_node[i]..., links[i]]
            data = randn(rng, dim(links[i - 1]), site_dims..., dim(links[i]))
        end
        push!(tensors, Tensor(data, indices))
    end
    return TN_RESTRUCT.TensorTrain(tensors)
end

function _dense_compare(a::Tensor, b::Tensor; rtol::Real=1e-10)
    return isapprox(a, b; rtol=rtol)
end

@testset "TensorTrain restructure" begin
    @testset "fuse_to: pairs adjacent MPS sites" begin
        # Build a 4-site MPS-like chain and fuse it into 2 nodes of 2 sites each.
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:4]
        tt = _restruct_chain([[sites[1]], [sites[2]], [sites[3]], [sites[4]]]; seed=11)
        before = TN_RESTRUCT.to_dense(tt)

        target_groups = [[sites[1], sites[2]], [sites[3], sites[4]]]
        fused = TN_RESTRUCT.fuse_to(tt, target_groups)
        @test length(fused) == 2
        after = TN_RESTRUCT.to_dense(fused)
        @test _dense_compare(before, after)
    end

    @testset "fuse_to: collapses to one node" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        tt = _restruct_chain([[sites[1]], [sites[2]], [sites[3]]]; seed=12)
        before = TN_RESTRUCT.to_dense(tt)

        # Single-node target with no edges.
        fused = TN_RESTRUCT.fuse_to(tt, [sites]; edges=Tuple{Int,Int}[])
        @test length(fused) == 1
        after = TN_RESTRUCT.to_dense(fused)
        @test _dense_compare(before, after)
    end

    @testset "split_to: separates fused groups" begin
        # Start from a 2-node chain whose nodes each carry 2 site indices.
        sx = [Index(2; tags=["x", "x=$i"]) for i in 1:2]
        sy = [Index(2; tags=["y", "y=$i"]) for i in 1:2]
        tt = _restruct_chain([[sx[1], sy[1]], [sx[2], sy[2]]]; seed=13)
        before = TN_RESTRUCT.to_dense(tt)

        # Split each group into two single-site nodes (4-node chain).
        target_groups = [[sx[1]], [sy[1]], [sx[2]], [sy[2]]]
        split = TN_RESTRUCT.split_to(tt, target_groups)
        @test length(split) == 4
        after = TN_RESTRUCT.to_dense(split)
        @test _dense_compare(before, after)
    end

    @testset "split_to honours truncation knobs" begin
        sx = [Index(2; tags=["x", "x=$i"]) for i in 1:2]
        sy = [Index(2; tags=["y", "y=$i"]) for i in 1:2]
        tt = _restruct_chain([[sx[1], sy[1]], [sx[2], sy[2]]], 4; seed=14)

        target_groups = [[sx[1]], [sy[1]], [sx[2]], [sy[2]]]
        # Exact split should preserve the dense form.
        split_exact = TN_RESTRUCT.split_to(tt, target_groups)
        @test _dense_compare(TN_RESTRUCT.to_dense(tt), TN_RESTRUCT.to_dense(split_exact))

        # Heavy truncation with a final sweep reduces every shared bond.
        split_trunc = TN_RESTRUCT.split_to(tt, target_groups; maxdim=1, final_sweep=true)
        for position in 1:(length(split_trunc) - 1)
            shared = commoninds(inds(split_trunc[position]), inds(split_trunc[position + 1]))
            for index in shared
                @test dim(index) <= 1
            end
        end
    end

    @testset "swap_site_indices: relabel target node" begin
        # Three-node chain of single-site nodes; swap site index between
        # adjacent nodes by reassigning ownership.
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        tt = _restruct_chain([[sites[1]], [sites[2]], [sites[3]]]; seed=15)
        before = TN_RESTRUCT.to_dense(tt)

        # Move sites[1] from node 1 to node 2 (and vice versa).
        target_assignment = Dict{Index, Int}(sites[1] => 2, sites[2] => 1)
        swapped = TN_RESTRUCT.swap_site_indices(tt, target_assignment)
        @test length(swapped) == 3
        after = TN_RESTRUCT.to_dense(swapped)
        @test _dense_compare(before, after)
    end

    @testset "swap_site_indices: empty assignment is a no-op" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:2]
        tt = _restruct_chain([[sites[1]], [sites[2]]]; seed=16)
        before = TN_RESTRUCT.to_dense(tt)
        swapped = TN_RESTRUCT.swap_site_indices(tt, Dict{Index, Int}())
        after = TN_RESTRUCT.to_dense(swapped)
        @test _dense_compare(before, after)
    end

    @testset "restructure_to: fuse-only path" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:4]
        tt = _restruct_chain([[sites[1]], [sites[2]], [sites[3]], [sites[4]]]; seed=21)
        before = TN_RESTRUCT.to_dense(tt)

        target_groups = [[sites[1], sites[2]], [sites[3], sites[4]]]
        result = TN_RESTRUCT.restructure_to(tt, target_groups)
        @test length(result) == 2
        @test _dense_compare(before, TN_RESTRUCT.to_dense(result))
    end

    @testset "restructure_to: split-only path" begin
        sx = [Index(2; tags=["x", "x=$i"]) for i in 1:2]
        sy = [Index(2; tags=["y", "y=$i"]) for i in 1:2]
        tt = _restruct_chain([[sx[1], sy[1]], [sx[2], sy[2]]]; seed=22)
        before = TN_RESTRUCT.to_dense(tt)

        target_groups = [[sx[1]], [sy[1]], [sx[2]], [sy[2]]]
        result = TN_RESTRUCT.restructure_to(tt, target_groups)
        @test length(result) == 4
        @test _dense_compare(before, TN_RESTRUCT.to_dense(result))
    end

    @testset "restructure_to: quantics fused -> interleaved" begin
        # 2 variables x 2 scales = 4 site indices.
        # Source: fused per scale [[x1,y1], [x2,y2]] (two nodes).
        # Target: interleaved per index [[x1],[y1],[x2],[y2]] (four nodes).
        x1 = Index(2; tags=["x", "x=1"])
        y1 = Index(2; tags=["y", "y=1"])
        x2 = Index(2; tags=["x", "x=2"])
        y2 = Index(2; tags=["y", "y=2"])

        fused = _restruct_chain([[x1, y1], [x2, y2]]; seed=23)
        before = TN_RESTRUCT.to_dense(fused)

        interleaved_groups = [[x1], [y1], [x2], [y2]]
        result = TN_RESTRUCT.restructure_to(fused, interleaved_groups)
        @test length(result) == 4
        @test _dense_compare(before, TN_RESTRUCT.to_dense(result))

        # Round trip: interleaved back to fused must reproduce the original.
        fused_back = TN_RESTRUCT.restructure_to(result, [[x1, y1], [x2, y2]])
        @test length(fused_back) == 2
        @test _dense_compare(before, TN_RESTRUCT.to_dense(fused_back))
    end

    @testset "argument validation" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:2]
        tt = _restruct_chain([[sites[1]], [sites[2]]]; seed=31)
        empty_tt = TN_RESTRUCT.TensorTrain(Tensor[])

        @test_throws ArgumentError TN_RESTRUCT.fuse_to(empty_tt, [[sites[1]], [sites[2]]])
        @test_throws ArgumentError TN_RESTRUCT.split_to(empty_tt, [[sites[1]], [sites[2]]])
        @test_throws ArgumentError TN_RESTRUCT.swap_site_indices(empty_tt, Dict{Index, Int}(sites[1] => 2))
        @test_throws ArgumentError TN_RESTRUCT.restructure_to(empty_tt, [[sites[1]], [sites[2]]])

        # Empty target_groups
        @test_throws ArgumentError TN_RESTRUCT.fuse_to(tt, Vector{Vector{Index}}())

        # Missing site
        @test_throws ArgumentError TN_RESTRUCT.fuse_to(tt, [[sites[1]]])

        # Duplicated site across nodes
        @test_throws ArgumentError TN_RESTRUCT.fuse_to(
            tt, [[sites[1], sites[2]], [sites[1]]],
        )

        # Foreign site
        foreign = Index(2; tags=["foreign"])
        @test_throws ArgumentError TN_RESTRUCT.fuse_to(tt, [[sites[1]], [foreign]])

        # Negative truncation knobs
        @test_throws ArgumentError TN_RESTRUCT.split_to(tt, [[sites[1]], [sites[2]]]; rtol=-1.0)
        @test_throws ArgumentError TN_RESTRUCT.split_to(tt, [[sites[1]], [sites[2]]]; cutoff=-1.0)
        @test_throws ArgumentError TN_RESTRUCT.split_to(tt, [[sites[1]], [sites[2]]]; maxdim=-1)
        @test_throws ArgumentError TN_RESTRUCT.swap_site_indices(tt, Dict{Index, Int}(sites[1] => 2); rtol=-1.0)
        @test_throws ArgumentError TN_RESTRUCT.swap_site_indices(tt, Dict{Index, Int}(sites[1] => 2); maxdim=-1)
        @test_throws ArgumentError TN_RESTRUCT.restructure_to(tt, [[sites[1]], [sites[2]]]; swap_rtol=-1.0)

        # Out-of-range swap target
        @test_throws ArgumentError TN_RESTRUCT.swap_site_indices(tt, Dict{Index, Int}(sites[1] => 0))
        @test_throws ArgumentError TN_RESTRUCT.swap_site_indices(tt, Dict{Index, Int}(sites[1] => 5))

        # Foreign swap key
        @test_throws ArgumentError TN_RESTRUCT.swap_site_indices(tt, Dict{Index, Int}(foreign => 1))

        # Bad edge endpoints in restructure
        @test_throws ArgumentError TN_RESTRUCT.fuse_to(
            tt, [[sites[1]], [sites[2]]];
            edges=[(1, 5)],
        )
    end
end
