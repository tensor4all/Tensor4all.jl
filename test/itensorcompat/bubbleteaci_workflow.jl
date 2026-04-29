using Test
using Tensor4all

const IC_BTCI = Tensor4all.ITensorCompat
const TN_BTCI = Tensor4all.TensorNetworks

@testset "BubbleTeaCI-shaped ITensorCompat workflow" begin
    sites = [Index(2; tags=["x", "x=$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC_BTCI.MPS(blocks, sites)
    original_dense = IC_BTCI.to_dense(m)

    replacement = Index(2; tags=["y", "y=1"])
    original_first = m[1]
    original_data = copy_data(m[1])
    @test IC_BTCI.replace_siteinds!(m, [sites[1]], [replacement]) === m
    @test m[1] === original_first
    @test copy_data(m[1]) == original_data
    @test IC_BTCI.siteinds(m)[1] == replacement

    dummy = Index(2; tags=["dummy", "dummy=1"])
    @test TN_BTCI.insert_identity!(m.tt, dummy, 1) === m.tt
    @test dummy in IC_BTCI.siteinds(m)

    restored = TN_BTCI.suminds(m.tt, dummy)
    restored_dense = TN_BTCI.to_dense(restored)
    expanded_dense = Array(IC_BTCI.to_dense(m), replacement, dummy, sites[2])
    @test Array(restored_dense, replacement, sites[2]) ≈ dropdims(sum(expanded_dense; dims=2); dims=2)

    fixed = TN_BTCI.fixinds(restored, replacement => 2)
    fixed_dense = TN_BTCI.to_dense(fixed)
    @test Array(fixed_dense, sites[2]) ≈ Array(restored_dense, replacement, sites[2])[2, :]

    added = m + m
    @test IC_BTCI.to_dense(added) ≈ 2.0 * IC_BTCI.to_dense(m)

    @test IC_BTCI.truncate!(added; cutoff=1e-12) === added
    @test IC_BTCI.orthogonalize!(added, 1) === added
    dense = IC_BTCI.to_dense(added)
    value = IC_BTCI.evaluate(added, IC_BTCI.siteinds(added), [1, 1, 2])
    @test value ≈ Array(dense, IC_BTCI.siteinds(added)...)[1, 1, 2]

    scalar_mps = IC_BTCI.MPS(TN_BTCI.TensorTrain([Tensor(3.5)]))
    @test length(scalar_mps) == 0
    @test IC_BTCI.scalar(scalar_mps) == 3.5
end
