using Test
using Tensor4all
using LinearAlgebra: norm

const TN = Tensor4all.TensorNetworks

function make_chain_indices()
    sites = [Index(2; tags=["s", "s=$n"]) for n in 1:3]
    links = [Index(2; tags=["Link", "l=$n"]) for n in 1:2]
    return sites, links
end

function make_test_mps(; sites::Union{Nothing, Vector{Index}}=nothing, links::Union{Nothing, Vector{Index}}=nothing, variant::Symbol=:a)
    sites === nothing && ((sites, links) = make_chain_indices())
    links === nothing && throw(ArgumentError("links must be provided when sites are provided"))

    if variant === :a
        t1_data = reshape([1.0, -0.5, 0.25, 2.0], 2, 2)
        t2_data = reshape([1.0, -1.0, 0.5, 0.25, 0.75, -0.5, 1.5, 2.0], 2, 2, 2)
        t3_data = reshape([1.0, 0.5, -0.75, 2.0], 2, 2)
    elseif variant === :b
        t1_data = reshape([0.5, 1.0, -1.5, 0.25], 2, 2)
        t2_data = reshape([0.25, 1.5, -0.5, 0.75, 2.0, -1.0, 0.5, 1.25], 2, 2, 2)
        t3_data = reshape([1.5, -0.5, 0.25, 1.0], 2, 2)
    else
        throw(ArgumentError("unknown MPS variant $variant"))
    end

    return TN.TensorTrain(
        Tensor[
            Tensor(t1_data, [sites[1], links[1]]),
            Tensor(t2_data, [links[1], sites[2], links[2]]),
            Tensor(t3_data, [links[2], sites[3]]),
        ],
    )
end

@testset "TensorTrain arithmetic" begin
    @testset "scalar multiply" begin
        tt = make_test_mps()
        scaled = 2.0 * tt

        @test length(scaled) == 3
        @test 0 <= scaled.llim <= length(scaled)
        @test 1 <= scaled.rlim <= length(scaled) + 1
    end

    @testset "scalar multiply from right" begin
        tt = make_test_mps()
        scaled = tt * 3.0
        @test length(scaled) == 3
    end

    @testset "scalar divide" begin
        tt = make_test_mps()
        divided = tt / 2.0
        @test length(divided) == 3
    end

    @testset "unary negation" begin
        tt = make_test_mps()
        negated = -tt
        @test length(negated) == 3
    end

    @testset "empty TensorTrain errors" begin
        empty_tt = TN.TensorTrain(Tensor[])
        @test_throws ArgumentError 2.0 * empty_tt
    end
end
