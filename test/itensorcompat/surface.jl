using Test
using Tensor4all

const IC = Tensor4all.ITensorCompat
const TN = Tensor4all.TensorNetworks

@testset "ITensorCompat MPS surface" begin
    s1 = Index(2, "s=1")
    s2 = Index(3, "s=2")
    l1 = Index(4; tags=["Link", "l=1"])

    t1 = Tensor(randn(2, 4), [s1, l1])
    t2 = Tensor(randn(4, 3), [l1, s2])
    m = IC.MPS(TN.TensorTrain([t1, t2]))

    @test length(m) == 2
    @test collect(IC.siteinds(m)) == [s1, s2]
    @test IC.linkinds(m) == [l1]
    @test IC.linkdims(m) == [4]
    @test IC.rank(m) == 4
    @test eltype(m) == Float64
    @test m[1] == t1

    new_t1 = Tensor(randn(2, 4), [s1, l1])
    @test (m[1] = new_t1) == new_t1
    @test m[1] == new_t1
    @test (m.tt.llim, m.tt.rlim) == (0, length(m) + 1)
end

@testset "ITensorCompat MPS validation" begin
    s1 = Index(2, "s=1")
    s2 = Index(2, "s=2")
    t = Tensor(randn(2, 2), [s1, s2])
    @test_throws ArgumentError IC.MPS(TN.TensorTrain([t]))
end

@testset "ITensorCompat MPS algebra and dense operations" begin
    s1 = Index(2, "s=1")
    s2 = Index(2, "s=2")
    a = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))
    b = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))

    @test IC.dot(a, b) == TN.dot(a.tt, b.tt)
    @test IC.inner(a, b) == TN.inner(a.tt, b.tt)
    @test IC.norm(a) ≈ TN.norm(a.tt)
    @test IC.to_dense(a) ≈ TN.to_dense(a.tt)

    c = a + b
    @test c isa IC.MPS
    @test IC.to_dense(c) ≈ TN.to_dense(TN.add(a.tt, b.tt))

    scaled = 2.0 * a
    @test scaled isa IC.MPS
    @test IC.to_dense(scaled) ≈ TN.to_dense(2.0 * a.tt)

    value = IC.evaluate(a, IC.siteinds(a), [1, 2])
    @test value == TN.evaluate(a.tt, IC.siteinds(a), [1, 2])

    new_sites = [sim(s) for s in IC.siteinds(a)]
    replaced = IC.replace_siteinds(a, IC.siteinds(a), new_sites)
    @test IC.siteinds(replaced) == new_sites
    @test IC.siteinds(a) == [s1, s2]
    @test IC.replace_siteinds!(a, IC.siteinds(a), new_sites) === a
    @test IC.siteinds(a) == new_sites
end

@testset "ITensorCompat cutoff and mutating canonical operations" begin
    s1 = Index(2, "s=1")
    s2 = Index(2, "s=2")
    m = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))

    before_dense = IC.to_dense(m)
    @test IC.orthogonalize!(m, 1) === m
    @test IC.to_dense(m) ≈ before_dense

    @test IC.truncate!(m; maxdim=1) === m
    @test maximum(IC.linkdims(m); init=0) <= 1

    m2 = IC.MPS(TN.random_tt([s1, s2]; linkdims=2))
    @test IC.truncate!(m2; cutoff=1e-12) === m2
    @test_throws ArgumentError IC.truncate!(m2; cutoff=1e-12, threshold=1e-12)
    @test_throws ArgumentError IC.truncate!(m2; threshold=1e-12)
    @test_throws ArgumentError IC.truncate!(m2; svd_policy=nothing)
    @test_throws ArgumentError IC.truncate!(m2)
end

@testset "ITensorCompat scalar MPS" begin
    m = IC.MPS(TN.TensorTrain([Tensor(3.5)]))
    @test length(m) == 0
    @test IC.siteinds(m) == Index[]
    @test IC.scalar(m) == 3.5
    @test Tensor4all.scalar(IC.to_dense(m)) == 3.5
    @test IC.evaluate(m) == 3.5
end

@testset "ITensorCompat raw MPS constructors" begin
    sites = [Index(2, "s=1"), Index(3, "s=2")]
    blocks = [
        reshape(collect(1.0:4.0), 1, 2, 2),
        reshape(collect(1.0:6.0), 2, 3, 1),
    ]

    m = IC.MPS(blocks, sites)
    @test IC.siteinds(m) == sites
    @test IC.linkdims(m) == [2]

    dense = Array(IC.to_dense(m), sites...)
    @test size(dense) == (2, 3)

    inferred = IC.MPS(blocks)
    @test dim.(IC.siteinds(inferred)) == [2, 3]
end

@testset "ITensorCompat MPO surface" begin
    x1 = Index(2, "x=1")
    y1 = Index(2, "y=1")
    x2 = Index(3, "x=2")
    y2 = Index(3, "y=2")
    l1 = Index(2; tags=["Link", "l=1"])

    w1 = Tensor(randn(2, 2, 2), [x1, y1, l1])
    w2 = Tensor(randn(2, 3, 3), [l1, x2, y2])
    W = IC.MPO(TN.TensorTrain([w1, w2]))

    @test length(W) == 2
    @test IC.siteinds(W) == [[x1, y1], [x2, y2]]
    @test IC.linkdims(W) == [2]
    @test IC.rank(W) == 2
    @test W[1] == w1

    new_w1 = Tensor(randn(2, 2, 2), [x1, y1, l1])
    @test (W[1] = new_w1) == new_w1
    @test W[1] == new_w1

    bad = Tensor(randn(2), [x1])
    @test_throws ArgumentError IC.MPO(TN.TensorTrain([bad]))
end
