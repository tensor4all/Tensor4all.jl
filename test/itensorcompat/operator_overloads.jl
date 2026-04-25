using Test
using Tensor4all

const IC = Tensor4all.ITensorCompat
const TN = Tensor4all.TensorNetworks

@testset "maxlinkdim" begin
    sites = [Index(2; tags=["s$n"]) for n in 1:3]
    blocks = [
        reshape(collect(1.0:8.0), 1, 2, 4),
        reshape(collect(1.0:24.0), 4, 2, 3),
        reshape(collect(1.0:6.0), 3, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    @test IC.maxlinkdim(m) == 4

    blocks_mpo = [
        reshape(collect(1.0:12.0), 1, 2, 2, 3),
        reshape(collect(1.0:12.0), 3, 2, 2, 1),
    ]
    input = [Index(2; tags=["in$n"]) for n in 1:2]
    output = [Index(2; tags=["out$n"]) for n in 1:2]
    w = IC.MPO(blocks_mpo, input, output)
    @test IC.maxlinkdim(w) == 3
end

@testset "data accessor" begin
    sites = [Index(2; tags=["s$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    d = IC.data(m)
    @test d isa Vector{Tensor4all.Tensor}
    @test length(d) == 2

    input = [Index(2; tags=["in$n"]) for n in 1:1]
    output = [Index(2; tags=["out$n"]) for n in 1:1]
    blocks_mpo = [reshape([1.0, 0.0, 0.0, -1.0], 1, 2, 2, 1)]
    w = IC.MPO(blocks_mpo, input, output)
    dw = IC.data(w)
    @test dw isa Vector{Tensor4all.Tensor}
    @test length(dw) == 1

    tt = TN.TensorTrain([Tensor(1.0)])
    dtt = IC.data(tt)
    @test dtt isa Vector{Tensor4all.Tensor}
    @test length(dtt) == 1
end
