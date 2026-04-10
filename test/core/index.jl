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
