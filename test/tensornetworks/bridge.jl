using Test
using Tensor4all
using Random: MersenneTwister

const TN_BRIDGE = Tensor4all.TensorNetworks
const STT_BRIDGE = Tensor4all.SimpleTT

@testset "TensorTrain ↔ SimpleTT bridge" begin
    @testset "MPS-like (N=3) round trip" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:4]
        tt = TN_BRIDGE.random_tt(MersenneTwister(11), sites; linkdims=3)
        before_dense = TN_BRIDGE.to_dense(tt)

        stt = STT_BRIDGE.TensorTrain(tt)
        @test stt isa STT_BRIDGE.TensorTrain{Float64,3}
        @test length(stt) == 4
        # Each tensor must be (left_link, site, right_link) with boundary dim-1.
        @test size(stt.sitetensors[1], 1) == 1
        @test size(stt.sitetensors[end], 3) == 1
        @test size(stt.sitetensors[2], 2) == 2  # site dim

        tt2 = TN_BRIDGE.TensorTrain(stt, sites)
        @test length(tt2) == 4
        # Round-trip preserves the underlying tensor.
        @test TN_BRIDGE.to_dense(tt2) ≈ before_dense
    end

    @testset "MPO-like (N=4) round trip" begin
        s_in = [Index(2; tags=["in", "s=$i"]) for i in 1:3]
        s_out = [Index(2; tags=["out", "s=$i"]) for i in 1:3]
        links = [Index(3; tags=["Link", "l=$i"]) for i in 1:2]
        rng = MersenneTwister(12)

        # Construct a 3-site MPO directly so each tensor has 2 site indices.
        tt = TN_BRIDGE.TensorTrain([
            Tensor(randn(rng, 2, 2, 3), [s_in[1], s_out[1], links[1]]),
            Tensor(randn(rng, 3, 2, 2, 3), [links[1], s_in[2], s_out[2], links[2]]),
            Tensor(randn(rng, 3, 2, 2), [links[2], s_in[3], s_out[3]]),
        ])
        before_dense = TN_BRIDGE.to_dense(tt)

        stt = STT_BRIDGE.TensorTrain(tt)
        @test stt isa STT_BRIDGE.TensorTrain{Float64,4}
        @test length(stt) == 3
        @test size(stt.sitetensors[1]) == (1, 2, 2, 3)  # left_link padded to 1
        @test size(stt.sitetensors[2]) == (3, 2, 2, 3)
        @test size(stt.sitetensors[3]) == (3, 2, 2, 1)  # right_link padded to 1

        tt2 = TN_BRIDGE.TensorTrain(stt, s_in, s_out)
        @test length(tt2) == 3
        @test TN_BRIDGE.to_dense(tt2) ≈ before_dense
    end

    @testset "generic site_groups overload" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        tt = TN_BRIDGE.random_tt(MersenneTwister(13), sites; linkdims=2)
        stt = STT_BRIDGE.TensorTrain(tt)
        # Generic groups: same as the MPS convenience.
        groups = [[s] for s in sites]
        tt2 = TN_BRIDGE.TensorTrain(stt, groups)
        @test TN_BRIDGE.to_dense(tt2) ≈ TN_BRIDGE.to_dense(tt)
    end

    @testset "compress! round trip via SimpleTT" begin
        # MPS compress via SimpleTT, then re-lift.
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:5]
        tt = TN_BRIDGE.random_tt(MersenneTwister(14), sites; linkdims=4)
        stt = STT_BRIDGE.TensorTrain(tt)
        STT_BRIDGE.compress!(stt, :SVD; tolerance=0.0, maxbonddim=2)
        tt2 = TN_BRIDGE.TensorTrain(stt, sites)
        @test length(tt2) == 5
        # Compression bounded the bond dim.
        @test all(d <= 2 for d in TN_BRIDGE.linkdims(tt2))
    end

    @testset "argument validation" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        tt = TN_BRIDGE.random_tt(MersenneTwister(15), sites; linkdims=2)
        stt = STT_BRIDGE.TensorTrain(tt)

        @test_throws DimensionMismatch TN_BRIDGE.TensorTrain(stt, sites[1:2])
        @test_throws ArgumentError TN_BRIDGE.TensorTrain(stt, [[sites[1], sites[2]] for _ in 1:3])
        @test_throws DimensionMismatch TN_BRIDGE.TensorTrain(
            stt,
            [[Index(99; tags=["bogus"])] for _ in 1:3],
        )

        empty_tt = TN_BRIDGE.TensorTrain(Tensor[])
        @test_throws ArgumentError STT_BRIDGE.TensorTrain(empty_tt)
    end
end
