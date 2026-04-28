using Test
using Tensor4all
using LinearAlgebra: norm

@testset "Tensor arithmetic" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data_ij = reshape(collect(1.0:6.0), 2, 3)

    a = Tensor(data_ij, [i, j])

    @testset "addition same order" begin
        b = Tensor(data_ij, [i, j])
        c = a + b
        @test inds(c) == [i, j]
        @test copy_data(c) ≈ 2.0 .* data_ij
    end

    @testset "addition permuted indices" begin
        b = Tensor(permutedims(data_ij, (2, 1)), [j, i])
        c = a + b
        @test inds(c) == [i, j]
        @test copy_data(c) ≈ 2.0 .* data_ij
    end

    @testset "subtraction" begin
        b = Tensor(data_ij, [i, j])
        c = a - b
        @test copy_data(c) ≈ zeros(2, 3)
    end

    @testset "unary negation" begin
        c = -a
        @test copy_data(c) ≈ -data_ij
    end

    @testset "scalar multiply" begin
        c = 3.0 * a
        @test copy_data(c) ≈ 3.0 .* data_ij
        c2 = a * 3.0
        @test copy_data(c2) ≈ 3.0 .* data_ij
    end

    @testset "scalar divide" begin
        c = a / 2.0
        @test copy_data(c) ≈ data_ij ./ 2.0
    end

    @testset "norm" begin
        @test norm(a) ≈ norm(data_ij)
    end

    @testset "isapprox same order" begin
        b = Tensor(data_ij .+ 1e-15, [i, j])
        @test isapprox(a, b; atol=1e-10)
    end

    @testset "isapprox permuted" begin
        b = Tensor(permutedims(data_ij, (2, 1)), [j, i])
        @test isapprox(a, b)
    end

    @testset "error: mismatched indices" begin
        k = Index(4; tags=["k"])
        b = Tensor(reshape(collect(1.0:8.0), 2, 4), [i, k])
        @test_throws ArgumentError a + b
    end

    @testset "error: different rank" begin
        k = Index(4; tags=["k"])
        b = Tensor(reshape(collect(1.0:24.0), 2, 3, 4), [i, j, k])
        @test_throws DimensionMismatch a + b
    end

    @testset "complex scalars" begin
        c = (2.0 + 1.0im) * a
        @test copy_data(c) ≈ (2.0 + 1.0im) .* data_ij
    end
end
