using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks

@testset "TensorTrain llim/rlim" begin
    @testset "setindex! widens ortho limits" begin
        i1 = Index(2; tags=["s1"])
        i2 = Index(2; tags=["s2"])
        i3 = Index(2; tags=["s3"])
        link1 = Index(2; tags=["l1"])
        link2 = Index(2; tags=["l2"])

        t1 = Tensor(randn(2, 2), [i1, link1])
        t2 = Tensor(randn(2, 2, 2), [link1, i2, link2])
        t3 = Tensor(randn(2, 2), [link2, i3])

        tt = TN.TensorTrain([t1, t2, t3], 1, 3)

        tt[1] = t1
        @test tt.llim == 0
        @test tt.rlim == 3
    end

    @testset "setindex! widens rlim" begin
        i1 = Index(2; tags=["s1"])
        i2 = Index(2; tags=["s2"])
        link = Index(2; tags=["l"])

        t1 = Tensor(randn(2, 2), [i1, link])
        t2 = Tensor(randn(2, 2), [link, i2])

        tt = TN.TensorTrain([t1, t2], 0, 2)

        tt[2] = t2
        @test tt.llim == 0
        @test tt.rlim == 3
    end

    @testset "default constructor has no ortho" begin
        i = Index(2; tags=["s"])
        t = Tensor(randn(2), [i])
        tt = TN.TensorTrain([t])
        @test tt.llim == 0
        @test tt.rlim == 2
    end

end
