using Test
using Tensor4all

const IC = Tensor4all.ITensorCompat
const TN_IC = Tensor4all.TensorNetworks

function make_test_mps_for_itensorcompat()
    s1 = Index(2; tags=["s", "s=1"])
    s2 = Index(2; tags=["s", "s=2"])
    link = Index(2; tags=["Link", "l=1"])
    return TN_IC.TensorTrain([
        Tensor(reshape([1.0, 0.0, 0.0, 1.0], 2, 2), [s1, link]),
        Tensor(reshape([2.0, 0.5, -1.0, 1.0], 2, 2), [link, s2]),
    ])
end

@testset "ITensorCompat wrapper validation" begin
    s1 = Index(2; tags=["s", "s=1"])
    s2 = Index(2; tags=["s", "s=2"])
    link = Index(1; tags=["Link", "l=1"])
    tt = TN_IC.TensorTrain([
        Tensor(ones(2, 1), [s1, link]),
        Tensor(ones(1, 2), [link, s2]),
    ])

    m = IC.MPS(tt)
    @test length(m) == 2
    @test IC.siteinds(m) == [s1, s2]
    @test m[1] === tt[1]
    @test IC.linkinds(m) == TN_IC.linkinds(tt)
    @test IC.linkdims(m) == TN_IC.linkdims(tt)
    @test eltype(m) == Float64

    bad = TN_IC.TensorTrain([Tensor(ones(2, 2), [s1, s2])])
    @test_throws ArgumentError IC.MPS(bad)
end

@testset "ITensorCompat cutoff-only truncation" begin
    m = IC.MPS(make_test_mps_for_itensorcompat())
    @test IC.truncate!(m; cutoff=1e-10) === m
    @test_throws ArgumentError IC.truncate!(m; threshold=1e-10)
    @test_throws ArgumentError IC.truncate!(m; svd_policy=TN_IC.SvdTruncationPolicy())
    @test_throws ArgumentError IC.truncate!(m)
end

@testset "ITensorCompat scalar MPS" begin
    m = IC.MPS(TN_IC.TensorTrain([Tensor(3.5)]))
    @test length(m) == 0
    @test IC.siteinds(m) == Index[]
    @test IC.scalar(m) == 3.5
    @test Tensor4all.scalar(IC.to_dense(m)) == 3.5
    @test IC.evaluate(m) == 3.5
end

@testset "ITensorCompat MPS forwarding" begin
    m = IC.MPS(make_test_mps_for_itensorcompat())
    @test IC.norm(m) ≈ TN_IC.norm(m.tt)
    @test IC.inner(m, m) ≈ TN_IC.inner(m.tt, m.tt)
    @test IC.to_dense(IC.dag(m)) ≈ Tensor4all.dag(IC.to_dense(m))
    @test IC.to_dense(2.0 * m) ≈ 2.0 * IC.to_dense(m)
end
