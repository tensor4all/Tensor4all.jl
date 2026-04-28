using Test
using Tensor4all
using LinearAlgebra: norm
using Random: MersenneTwister

const TN_RANDOM = Tensor4all.TensorNetworks

@testset "TensorTrain random_tt" begin
    @testset "single site real" begin
        s = Index(3; tags=["s", "s=1"])
        tt = TN_RANDOM.random_tt([s])
        @test length(tt) == 1
        @test rank(tt[1]) == 1
        @test TN_RANDOM.norm(tt) ≈ 1.0
    end

    @testset "uniform link dim, real" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:5]
        tt = TN_RANDOM.random_tt(sites; linkdims=4)
        @test length(tt) == 5
        @test all(d == 2 for d in dim.(TN_RANDOM.siteinds(tt)[1]))
        link_dims = TN_RANDOM.linkdims(tt)
        # Cap by min(linkdims, cumulative product) — for d=2, 5 sites, linkdims=4:
        # right boundary cap = min(4, 2) = 2
        # then bonds grow ≤ 4 internally
        @test all(d <= 4 for d in link_dims)
        @test TN_RANDOM.norm(tt) ≈ 1.0
    end

    @testset "non-uniform link dims" begin
        sites = [Index(3; tags=["s", "s=$i"]) for i in 1:4]
        tt = TN_RANDOM.random_tt(sites; linkdims=[2, 5, 3])
        link_dims = TN_RANDOM.linkdims(tt)
        @test length(link_dims) == 3
        @test link_dims[1] <= 2
        @test link_dims[2] <= 5
        @test link_dims[3] <= 3
        @test TN_RANDOM.norm(tt) ≈ 1.0
    end

    @testset "complex eltype" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:4]
        tt = TN_RANDOM.random_tt(ComplexF64, sites; linkdims=3)
        @test eltype(Tensor4all.copy_data(tt[2])) == ComplexF64
        @test TN_RANDOM.norm(tt) ≈ 1.0
    end

    @testset "rng reproducibility" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        tt1 = TN_RANDOM.random_tt(MersenneTwister(42), sites; linkdims=2)
        tt2 = TN_RANDOM.random_tt(MersenneTwister(42), sites; linkdims=2)
        for j in 1:length(tt1)
            @test Tensor4all.copy_data(tt1[j]) == Tensor4all.copy_data(tt2[j])
        end
    end

    @testset "argument validation" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        @test_throws ArgumentError TN_RANDOM.random_tt(Index[])
        @test_throws ArgumentError TN_RANDOM.random_tt(sites; linkdims=0)
        @test_throws ArgumentError TN_RANDOM.random_tt(sites; linkdims=-1)
        @test_throws DimensionMismatch TN_RANDOM.random_tt(sites; linkdims=[2, 2, 2])  # too many
        @test_throws DimensionMismatch TN_RANDOM.random_tt(sites; linkdims=[2])         # too few
        @test_throws ArgumentError TN_RANDOM.random_tt(sites; linkdims=[0, 1])         # zero entry
    end
end
