using Test
using Tensor4all

@testset "TensorTrain queries" begin
    s1 = Index(2; tags=["s1"])
    s2 = Index(2; tags=["s2"])
    s3 = Index(2; tags=["s3"])
    l1 = Index(3; tags=["l1"])
    l2 = Index(3; tags=["l2"])

    t1 = Tensor(randn(2, 3), [s1, l1])
    t2 = Tensor(randn(3, 2, 3), [l1, s2, l2])
    t3 = Tensor(randn(3, 2), [l2, s3])
    tt = TensorNetworks.TensorTrain([t1, t2, t3])

    @testset "linkinds(tt)" begin
        links = TensorNetworks.linkinds(tt)
        @test length(links) == 2
        @test links[1] == l1
        @test links[2] == l2
    end

    @testset "linkinds(tt, i)" begin
        @test TensorNetworks.linkinds(tt, 1) == l1
        @test TensorNetworks.linkinds(tt, 2) == l2
        @test_throws BoundsError TensorNetworks.linkinds(tt, 0)
        @test_throws BoundsError TensorNetworks.linkinds(tt, 3)
    end

    @testset "linkdims(tt)" begin
        @test TensorNetworks.linkdims(tt) == [3, 3]
    end

    @testset "siteinds(tt)" begin
        sites = TensorNetworks.siteinds(tt)
        @test length(sites) == 3
        @test s1 in sites[1]
        @test s2 in sites[2]
        @test s3 in sites[3]
    end

    @testset "siteinds(tt, i)" begin
        @test s1 in TensorNetworks.siteinds(tt, 1)
        @test s2 in TensorNetworks.siteinds(tt, 2)
        @test s3 in TensorNetworks.siteinds(tt, 3)
        @test_throws BoundsError TensorNetworks.siteinds(tt, 0)
        @test_throws BoundsError TensorNetworks.siteinds(tt, 4)
    end

    @testset "1-site TT (no links)" begin
        tt1 = TensorNetworks.TensorTrain([Tensor(randn(2), [s1])])
        @test TensorNetworks.linkinds(tt1) == Index[]
        @test length(TensorNetworks.siteinds(tt1)) == 1
    end

    @testset "dag" begin
        data_c = ComplexF64[1 + 2im, 3 + 4im, 5 + 6im, 7 + 8im, 9 + 10im, 11 + 12im]
        tc = Tensor(reshape(data_c, 2, 3), [s1, l1])
        tt_c = TensorNetworks.TensorTrain([tc])
        tt_d = TensorNetworks.dag(tt_c)
        @test inds(tt_d[1]) == inds(tc)
        @test tt_d[1].data ≈ conj(tc.data)
        @test length(tt_d) == 1
    end
end
