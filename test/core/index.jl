using Test
using Tensor4all

@testset "Index backend wrapper" begin
    i = Tensor4all.Index(4; tags=["x", "site", "x"], plev=1, id=42)
    j = Tensor4all.sim(i)

    @test Tensor4all.dim(i) == 4
    @test Tensor4all.tags(i) == ["site", "x"]
    @test Tensor4all.plev(i) == 1
    @test Tensor4all.id(i) == 42
    @test Tensor4all.hastag(i, "x")
    @test sprint(show, i) == "Index(4|site,x; plev=1)"

    same = Tensor4all.Index(4; tags=["site", "x"], plev=1, id=42)
    @test same == i
    @test hash(same) == hash(i)

    @test Tensor4all.id(j) != Tensor4all.id(i)
    @test Tensor4all.dim(j) == Tensor4all.dim(i)
    @test Tensor4all.tags(j) == Tensor4all.tags(i)
    @test Tensor4all.plev(j) == Tensor4all.plev(i)

    ip = Tensor4all.prime(i, 2)
    @test Tensor4all.plev(ip) == 3
    @test Tensor4all.id(ip) == Tensor4all.id(i)
    @test Tensor4all.plev(Tensor4all.noprime(ip)) == 0
    @test Tensor4all.plev(Tensor4all.setprime(i, 7)) == 7

    xs = [i, j, ip]
    ys = [j, ip]
    @test Tensor4all.commoninds(xs, ys) == [j, ip]
    @test Tensor4all.uniqueinds(xs, ys) == [i]

    @test_throws ArgumentError Tensor4all.Index(0)
    @test_throws ArgumentError Tensor4all.Index(2; plev=-1)
end
