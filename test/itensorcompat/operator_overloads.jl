using Test
using Tensor4all
using Random

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

@testset "ITensor-style constructors and random helpers" begin
    sites = IC.siteinds(2, 3)
    @test length(sites) == 3
    @test dim.(sites) == [2, 2, 2]
    @test all(Tensor4all.hastag.(sites, "Site"))
    @test Tensor4all.hasplev(sites[1], 0)
    @test !Tensor4all.hasplev(prime(sites[1]), 0)

    filled = ITensor(2.5, sites[1], sites[2])
    @test Tensor4all.inds(filled) == sites[1:2]
    @test filled.data == fill(2.5, 2, 2)

    rng = MersenneTwister(7)
    r = IC.random_itensor(rng, ComplexF64, sites[1], sites[2])
    @test r isa Tensor4all.Tensor
    @test eltype(r) == ComplexF64
    @test Tensor4all.inds(r) == sites[1:2]
    @test Tensor4all.inds(IC.random_itensor(sites[1:2])) == sites[1:2]

    m = IC.random_mps(MersenneTwister(11), Float64, sites; linkdims=2)
    @test m isa IC.MPS
    @test IC.siteinds(m) == sites
    @test maximum(IC.linkdims(m); init=0) <= 2
end

@testset "ITensor-style apply overloads" begin
    input = [Index(2; tags=["in", "n=$n"]) for n in 1:2]
    output = [Index(2; tags=["out", "n=$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1, 2, 2, 2),
        reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], 2, 2, 2, 1),
    ]
    op = IC.MPO(blocks, input, output)
    psi = IC.MPS([
        reshape([1.0, 2.0, 0.0, 1.0], 1, 2, 2),
        reshape([0.5, -1.0, 2.0, 1.0], 2, 2, 1),
    ], input)

    got = IC.apply(op, psi; alg="naive", cutoff=1e-12)
    @test got isa IC.MPS
    @test IC.siteinds(got) == output
    @test IC.to_dense(got) ≈ TN.to_dense(TN.contract(op.tt, psi.tt; method=:naive, threshold=1e-12))

    middle = [Index(2; tags=["mid", "n=$n"]) for n in 1:2]
    left = IC.MPO(blocks, input, middle)
    right = IC.MPO(blocks, middle, output)
    composed = IC.apply(left, right; alg=:naive)
    @test composed isa IC.MPO
    @test IC.siteinds(composed) == [[input[1], output[1]], [input[2], output[2]]]
end

@testset "MPO arithmetic overloads" begin
    input = [Index(2; tags=["in", "n=$n"]) for n in 1:2]
    output = [Index(2; tags=["out", "n=$n"]) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1, 2, 2, 2),
        reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], 2, 2, 2, 1),
    ]
    A = IC.MPO(blocks, input, output)
    B = IC.MPO(blocks, input, output)

    @test IC.to_dense(A + B) ≈ 2 * IC.to_dense(A)
    @test IC.norm(A - B) <= 1e-12
    @test IC.to_dense(3 * A) ≈ 3 * IC.to_dense(A)

    input3 = [Index(2; tags=["in3", "n=$n"]) for n in 1:3]
    output3 = [Index(2; tags=["out3", "n=$n"]) for n in 1:3]
    blocks3 = [
        reshape(collect(1.0:8.0), 1, 2, 2, 2),
        reshape(collect(1.0:16.0), 2, 2, 2, 2),
        reshape(collect(1.0:8.0), 2, 2, 2, 1),
    ]
    C = IC.MPO(blocks3, input3, output3)
    D = IC.MPO(blocks3, input3, output3)
    @test C + D isa IC.MPO
    @test IC.siteinds(C + D) == [[input3[i], output3[i]] for i in eachindex(input3)]
end

@testset "non-chain MPO operations are rejected" begin
    # This is accepted by the strict MPO wrapper because each tensor has two
    # non-adjacent-bond site indices under TensorNetworks.siteinds, but it has
    # an extra T1-T3 edge and is not a chain.
    x1 = Index(2; tags=["x1"])
    x2 = Index(2; tags=["x2"])
    x3 = Index(2; tags=["x3"])
    y1 = Index(2; tags=["y1"])
    y2 = Index(2; tags=["y2"])
    y3 = Index(2; tags=["y3"])
    l12 = Index(2; tags=["Link", "l12"])
    l23 = Index(2; tags=["Link", "l23"])
    l13 = Index(2; tags=["Link", "l13"])

    @test_throws ArgumentError IC.MPO(TN.TensorTrain([
        Tensor(ones(2, 2, 2), [x1, l12, l13]),
        Tensor(ones(2, 2, 2, 2), [l12, x2, y2, l23]),
        Tensor(ones(2, 2, 2), [l23, x3, l13]),
    ]))
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

@testset "sim(siteinds, ...) on MPS" begin
    sites = [Index(2; tags=["s$n"], plev=n) for n in 1:2]
    blocks = [
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        reshape([2.0, 0.0, 0.0, -1.0], 2, 2, 1),
    ]
    m = IC.MPS(blocks, sites)
    new_sites = IC.sim(IC.siteinds, m)
    @test new_sites isa Vector{Tensor4all.Index}
    @test length(new_sites) == 2
    @test dim.(new_sites) == dim.(sites)
    @test Tensor4all.tags.(new_sites) == Tensor4all.tags.(sites)
    @test plev.(new_sites) == plev.(sites)
    @test id.(new_sites) != id.(sites)
end

@testset "sim(siteinds, ...) on MPO" begin
    i1 = Index(2; tags=["i1"]); o1 = Index(2; tags=["o1"])
    i2 = Index(2; tags=["i2"]); o2 = Index(2; tags=["o2"])
    l1 = Index(3; tags=["Link", "l=1"])
    t1 = Tensor(reshape(collect(1.0:12.0), 2, 2, 3), [i1, o1, l1])
    t2 = Tensor(reshape(collect(1.0:12.0), 3, 2, 2), [l1, i2, o2])
    w = IC.MPO([t1, t2])
    new_sites = IC.sim(IC.siteinds, w)
    @test new_sites isa Vector{Vector{Tensor4all.Index}}
    @test length(new_sites) == 2
    @test all(length(g) == 2 for g in new_sites)
    @test id.(vcat(new_sites...)) != id.(vcat(i1, o1, i2, o2))
end

@testset "Index prime via ' operator" begin
    idx = Index(4; tags=["x"], plev=0)
    idx1 = idx'
    @test Tensor4all.plev(idx1) == 1
    @test Tensor4all.dim(idx1) == 4
    @test Tensor4all.tags(idx1) == ["x"]
    @test Tensor4all.id(idx1) == Tensor4all.id(idx)

    idx2 = idx''
    @test Tensor4all.plev(idx2) == 2

    @test Tensor4all.plev(idx) == 0  # original unchanged
end
