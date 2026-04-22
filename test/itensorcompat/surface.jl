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
