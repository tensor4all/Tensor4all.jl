using Test
using Tensor4all

const IC_RAW = Tensor4all.ITensorCompat

@testset "raw MPS block constructor" begin
    sites = [Index(2; tags=["s", "s=$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC_RAW.MPS(blocks, sites)
    @test IC_RAW.siteinds(m) == sites
    dense = Array(IC_RAW.to_dense(m), sites...)
    @test dense == [2.0 0.0; 0.0 -1.0]

    generated = IC_RAW.MPS(blocks)
    @test tags(IC_RAW.siteinds(generated)[1]) == ["Site", "n=1"]
end

@testset "raw MPO block constructor" begin
    input = [Index(2; tags=["in", "in=1"])]
    output = [Index(2; tags=["out", "out=1"])]
    blocks = [reshape([1.0, 0.0, 0.0, -1.0], 1, 2, 2, 1)]

    mpo = IC_RAW.MPO(blocks, input, output)
    @test IC_RAW.siteinds(mpo) == [[input[1], output[1]]]
end
