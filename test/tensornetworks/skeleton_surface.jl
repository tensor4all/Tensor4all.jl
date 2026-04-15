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
end
