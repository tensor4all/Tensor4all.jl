using Test
using Tensor4all

const TN_PC = Tensor4all.TensorNetworks

function _pc_two_site_tt(; tags_prefix="a")
    s1 = Index(2; tags=[tags_prefix, "s=1"])
    s2 = Index(2; tags=[tags_prefix, "s=2"])
    link = Index(2; tags=[tags_prefix, "link"])
    tt = TN_PC.TensorTrain([
        Tensor([1.0 0.0; 0.0 1.0], [s1, link]),
        Tensor([1.0 3.0; 2.0 4.0], [link, s2]),
    ])
    return tt, [s1, s2]
end

@testset "TensorNetworks partial_contract" begin
    @testset "diagonal pairs compute elementwise product" begin
        a, a_sites = _pc_two_site_tt(tags_prefix="a")
        b, b_sites = _pc_two_site_tt(tags_prefix="b")
        spec = TN_PC.PartialContractionSpec(
            Pair{Index,Index}[],
            [a_sites[1] => b_sites[1], a_sites[2] => b_sites[2]];
            output_order=a_sites,
        )

        result = TN_PC.partial_contract(a, b, spec)
        dense_result = Array(TN_PC.to_dense(result), a_sites...)
        dense_a = Array(TN_PC.to_dense(a), a_sites...)
        dense_b = Array(TN_PC.to_dense(b), b_sites...)

        @test dense_result ≈ dense_a .* dense_b
        @test TN_PC.siteinds(result) == [[a_sites[1]], [a_sites[2]]]
    end

    @testset "elementwise_product is a thin convenience wrapper" begin
        a, a_sites = _pc_two_site_tt(tags_prefix="aew")
        b, b_sites = _pc_two_site_tt(tags_prefix="bew")

        result = TN_PC.elementwise_product(a, b)
        @test Array(TN_PC.to_dense(result), a_sites...) ≈
              Array(TN_PC.to_dense(a), a_sites...) .* Array(TN_PC.to_dense(b), b_sites...)
    end

    @testset "validation" begin
        a, a_sites = _pc_two_site_tt(tags_prefix="av")
        b, b_sites = _pc_two_site_tt(tags_prefix="bv")
        wrong_dim = Index(3; tags=["wrong"])
        missing = Index(2; tags=["missing"])

        @test_throws DimensionMismatch TN_PC.PartialContractionSpec(
            Pair{Index,Index}[],
            [a_sites[1] => wrong_dim],
        )
        @test_throws ArgumentError TN_PC.partial_contract(
            a,
            b,
            TN_PC.PartialContractionSpec(Pair{Index,Index}[], [missing => b_sites[1]]),
        )
        @test_throws ArgumentError TN_PC.PartialContractionSpec(
            Pair{Index,Index}[],
            [a_sites[1] => b_sites[1], a_sites[1] => b_sites[2]],
        )
    end
end
