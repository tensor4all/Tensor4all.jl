using Test
using Tensor4all
using LinearAlgebra: norm

const TN_DENSE = Tensor4all.TensorNetworks

function _dense_two_site_mps_real()
    s1 = Index(2; tags=["k", "k=1"])
    s2 = Index(2; tags=["k", "k=2"])
    link = Index(2; tags=["Link", "k-link=1"])
    tt = TN_DENSE.TensorTrain([
        Tensor([1.0 0.0; 0.0 1.0], [s1, link]),
        Tensor([2.0 0.0; 0.0 -1.0], [link, s2]),
    ])
    expected = Tensor([2.0 0.0; 0.0 -1.0], [s1, s2])
    return tt, expected
end

function _dense_two_site_mps_complex()
    s1 = Index(2; tags=["k", "k=1"])
    s2 = Index(2; tags=["k", "k=2"])
    link = Index(2; tags=["Link", "k-link=1"])
    tt = TN_DENSE.TensorTrain([
        Tensor(ComplexF64[1.0 0.0; 0.0 1.0im], [s1, link]),
        Tensor(ComplexF64[2.0 0.0; 0.0 1.0], [link, s2]),
    ])
    expected = Tensor(ComplexF64[2.0 0.0; 0.0 1.0im], [s1, s2])
    return tt, expected
end

@testset "TensorTrain to_dense" begin
    @testset "real two-site" begin
        tt, expected = _dense_two_site_mps_real()
        result = TN_DENSE.to_dense(tt)
        @test result isa Tensor
        @test result ≈ expected
    end

    @testset "complex two-site" begin
        tt, expected = _dense_two_site_mps_complex()
        result = TN_DENSE.to_dense(tt)
        @test result isa Tensor
        @test result ≈ expected
    end

    @testset "norm round-trip" begin
        tt, expected = _dense_two_site_mps_real()
        dense = TN_DENSE.to_dense(tt)
        @test TN_DENSE.norm(tt) ≈ norm(dense)
    end

    @testset "to_dense + add equivalence" begin
        tt, _ = _dense_two_site_mps_real()
        dense_a = TN_DENSE.to_dense(tt)
        dense_b = TN_DENSE.to_dense(2.0 * tt)
        @test norm(dense_b - 2.0 * dense_a) < 1e-12
    end

    @testset "empty TensorTrain errors" begin
        empty_tt = TN_DENSE.TensorTrain(Tensor[])
        @test_throws ArgumentError TN_DENSE.to_dense(empty_tt)
    end

    @testset "three-site MPO" begin
        sites_in = [Index(2; tags=["s_in", "s=$n"]) for n in 1:3]
        sites_out = [Index(2; tags=["s_out", "s=$n"]) for n in 1:3]
        links = [Index(2; tags=["Link", "l=$n"]) for n in 1:2]

        tt = TN_DENSE.TensorTrain([
            Tensor(randn(2, 2, 2), [sites_in[1], sites_out[1], links[1]]),
            Tensor(randn(2, 2, 2, 2), [links[1], sites_in[2], sites_out[2], links[2]]),
            Tensor(randn(2, 2, 2), [links[2], sites_in[3], sites_out[3]]),
        ])

        dense = TN_DENSE.to_dense(tt)
        @test rank(dense) == 6
        @test all(dim(i) == 2 for i in inds(dense))
        @test TN_DENSE.norm(tt) ≈ norm(dense)
    end
end
