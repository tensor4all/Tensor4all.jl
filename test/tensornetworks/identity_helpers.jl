using Test
using Tensor4all

const TN_ID = Tensor4all.TensorNetworks

@testset "insert_identity!" begin
    sites = [Index(2; tags=["x", "x=$n"]) for n in 1:2]
    link = Index(2; tags=["Link", "l=1"])
    tt = TN_ID.TensorTrain([
        Tensor([1.0 0.0; 0.0 1.0], [sites[1], link]),
        Tensor([2.0 0.0; 0.0 -1.0], [link, sites[2]]),
    ], 1, 2)
    original_dense = TN_ID.to_dense(tt)

    newsite = Index(2; tags=["x", "x=mid"])
    @test TN_ID.insert_identity!(tt, newsite, 1) === tt
    @test length(tt) == 3
    @test (tt.llim, tt.rlim) == (0, length(tt) + 1)
    @test newsite in only.(TN_ID.siteinds(tt)[2:2])
    @test Tensor4all.storage_kind(tt[2]) == :diagonal

    summed = TN_ID.suminds(tt, newsite)
    @test Array(TN_ID.to_dense(summed), sites...) ≈ Array(original_dense, sites...)
    @test_throws ArgumentError TN_ID.insert_identity!(tt, Index(2; tags=["bad"]), -1)
end
