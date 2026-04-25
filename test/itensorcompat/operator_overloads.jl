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

@testset "MPO from Tensor vector" begin
    i1 = Index(2; tags=["i1"]); o1 = Index(2; tags=["o1"])
    i2 = Index(2; tags=["i2"]); o2 = Index(2; tags=["o2"])
    l1 = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i1, o1, l1])
    t2 = Tensor(reshape(collect(1.0:12.0), 3, 2, 2), [l1, i2, o2])
    w = IC.MPO([t1, t2])
    @test length(w) == 2
    @test length(IC.siteinds(w)) == 2
    @test IC.siteinds(w)[1] == [i1, o1]
    @test IC.siteinds(w)[2] == [i2, o2]
    @test IC.data(w)[1] === t1

    # error case: tensor with wrong number of site indices
    t_bad = Tensor(reshape(collect(1.0:6.0), 3, 2), [l1, i2])
    @test_throws ArgumentError IC.MPO([t1, t_bad])
end

@testset "prime/prime! on MPS" begin
    sites = [Index(2; tags=["s$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)

    mp = IC.prime(m)
    @test IC.siteinds(mp)[1].plev == 1
    @test IC.siteinds(m)[1].plev == 0  # original unchanged
    @test mp[1].data === m[1].data     # tensor data shared

    IC.prime!(m)
    @test IC.siteinds(m)[1].plev == 1

    mp2 = IC.prime(m, 2)
    @test IC.siteinds(mp2)[1].plev == 3
end

@testset "prime/prime! on MPO" begin
    i1 = Index(2; tags=["i1"]); o1 = Index(2; tags=["o1"])
    i2 = Index(2; tags=["i2"]); o2 = Index(2; tags=["o2"])
    l1 = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i1, o1, l1])
    t2 = Tensor(reshape(collect(1.0:12.0), 3, 2, 2), [l1, i2, o2])
    w = IC.MPO([t1, t2])

    wp = IC.prime(w)
    @test all(idx.plev == 1 for idx in Iterators.flatten(IC.siteinds(wp)))
    @test all(idx.plev == 0 for idx in Iterators.flatten(IC.siteinds(w)))

    IC.prime!(w)
    @test all(idx.plev == 1 for idx in Iterators.flatten(IC.siteinds(w)))
end

@testset "replaceprime on MPS" begin
    sites = [Index(2; tags=["s$n"], plev=2) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    mr = IC.replaceprime(m, 2 => 0)
    @test IC.siteinds(mr)[1].plev == 0
    @test IC.siteinds(m)[1].plev == 2   # original unchanged
end

@testset "replaceprime on MPO" begin
    i1 = Index(2; tags=["i1"], plev=1); o1 = Index(2; tags=["o1"], plev=1)
    i2 = Index(2; tags=["i2"], plev=1); o2 = Index(2; tags=["o2"], plev=1)
    l1 = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i1, o1, l1])
    t2 = Tensor(reshape(collect(1.0:12.0), 3, 2, 2), [l1, i2, o2])
    w = IC.MPO([t1, t2])
    wr = IC.replaceprime(w, 1 => 0)
    @test all(idx.plev == 0 for idx in Iterators.flatten(IC.siteinds(wr)))
end
