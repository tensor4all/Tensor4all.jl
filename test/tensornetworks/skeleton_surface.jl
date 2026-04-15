using Test
using Tensor4all

@testset "TensorNetworks skeleton surface" begin
    TN = Tensor4all.TensorNetworks

    for name in (
        :findsite,
        :findsites,
        :findallsiteinds_by_tag,
        :findallsites_by_tag,
        Symbol("replace_siteinds!"),
        :replace_siteinds,
        Symbol("replace_siteinds_part!"),
        :rearrange_siteinds,
        :makesitediagonal,
        :extractdiagonal,
        :matchsiteinds,
        :save_as_mps,
        :load_tt,
    )
        @test isdefined(TN, name)
    end

    i1 = Index(2; tags=["x", "x=1"])
    i2 = Index(2; tags=["x", "x=2"])
    tt = TN.TensorTrain(
        Tensor[
            Tensor(ones(2), [i1]),
            Tensor(ones(2), [i2]),
        ],
        0,
        3,
    )

    @test_throws Tensor4all.SkeletonNotImplemented TN.replace_siteinds_part!(tt, [i1], [sim(i1)])
    @test_throws Tensor4all.SkeletonNotImplemented TN.rearrange_siteinds(tt, [[i1], [i2]])
    @test_throws Tensor4all.SkeletonNotImplemented TN.makesitediagonal(tt, "x")
    @test_throws Tensor4all.SkeletonNotImplemented TN.extractdiagonal(tt, "x")
end
