using Test
using Tensor4all

@testset "Index skeleton" begin
    i = Tensor4all.Index(4; tags=["x", "site"], plev=1)
    j = Tensor4all.sim(i)

    @test Tensor4all.dim(i) == 4
    @test Tensor4all.tags(i) == ["x", "site"]
    @test Tensor4all.plev(i) == 1
    @test Tensor4all.hastag(i, "x")
    @test Tensor4all.id(i) != Tensor4all.id(j)
    @test Tensor4all.dim(j) == Tensor4all.dim(i)

    ip = Tensor4all.prime(i, 2)
    @test Tensor4all.plev(ip) == 3
    @test Tensor4all.id(ip) == Tensor4all.id(i)
    @test Tensor4all.plev(Tensor4all.noprime(ip)) == 0
    @test Tensor4all.plev(Tensor4all.setprime(i, 7)) == 7

    xs = [i, j, ip]
    ys = [j, ip]
    @test Tensor4all.commoninds(xs, ys) == [j, ip]
    @test Tensor4all.uniqueinds(xs, ys) == [i]
end

@testset "ITensors-style Index constructor" begin
    i = Tensor4all.Index(3, "x")
    @test Tensor4all.dim(i) == 3
    @test Tensor4all.tags(i) == ["x"]

    j = Tensor4all.Index(4, "x,y"; plev=2)
    @test Tensor4all.dim(j) == 4
    @test Tensor4all.tags(j) == ["x", "y"]
    @test Tensor4all.plev(j) == 2

    k = Tensor4all.Index(5; tags="site, n=1")
    @test Tensor4all.tags(k) == ["site", "n=1"]
end

@testset "Index replacement compatibility" begin
    i = Tensor4all.Index(2; tags=["i"])
    j = Tensor4all.Index(3; tags=["j"])
    ip = Tensor4all.Index(2; tags=["ip"])
    jp = Tensor4all.Index(3; tags=["jp"])
    bad = Tensor4all.Index(5; tags=["bad"])
    missing = Tensor4all.Index(2; tags=["missing"])

    xs = [i, j]
    @test Tensor4all.replaceind(xs, i, ip) == [ip, j]
    @test Tensor4all.replaceind(xs, i => ip) == [ip, j]
    @test Tensor4all.replaceinds(xs, i => ip, j => jp) == [ip, jp]
    @test Tensor4all.replaceinds(xs) == xs
    @test Tensor4all.replaceinds(xs, ()) == xs
    @test Tensor4all.replaceinds(xs, [i], [ip]) == [ip, j]
    @test Tensor4all.replaceinds(xs, [missing], [ip]) == xs
    @test_throws ArgumentError Tensor4all.replaceinds(xs, [i], [bad])

    a = Tensor4all.Index(2; tags=["a"])
    b = Tensor4all.Index(2; tags=["b"])
    ap = Tensor4all.Index(2; tags=["ap"])
    @test Tensor4all.replaceinds([a, b], a => b, b => ap) == [b, ap]
end
