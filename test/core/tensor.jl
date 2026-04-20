using Test
using Tensor4all

@testset "Tensor skeleton" begin
    i = Tensor4all.Index(2; tags=["i"])
    j = Tensor4all.Index(3; tags=["j"])
    data = reshape(collect(1.0:6.0), 2, 3)
    tensor = Tensor4all.Tensor(data, [i, j])

    @test Tensor4all.rank(tensor) == 2
    @test Tensor4all.dims(tensor) == (2, 3)
    @test Tensor4all.inds(tensor) == [i, j]
    @test Tensor4all.inds(Tensor4all.prime(tensor)) == [Tensor4all.prime(i), Tensor4all.prime(j)]

    bad = PermutedDimsArray(reshape(collect(1.0:8.0), 2, 2, 2), (2, 1, 3))
    k = Tensor4all.Index(2; tags=["k"])
    @test_throws ArgumentError Tensor4all.Tensor(bad, [i, j, k])
    @test_throws DimensionMismatch Tensor4all.Tensor(data, [i])
    contracted = Tensor4all.contract(tensor, tensor)
    @test Tensor4all.rank(contracted) == 0
    @test Tensor4all.dims(contracted) == ()
    @test contracted.data[] == 91.0
end

@testset "Tensor replaceind compatibility" begin
    i = Tensor4all.Index(2; tags=["i"])
    j = Tensor4all.Index(3; tags=["j"])
    ip = Tensor4all.Index(2; tags=["ip"])
    jp = Tensor4all.Index(3; tags=["jp"])
    bad = Tensor4all.Index(5; tags=["bad"])
    missing = Tensor4all.Index(2; tags=["missing"])

    data = reshape(collect(1.0:6.0), 2, 3)
    handle = Ptr{Cvoid}(1)
    tensor = Tensor4all.Tensor(data, [i, j]; backend_handle=handle)

    replaced = Tensor4all.replaceind(tensor, i, ip)
    @test Tensor4all.inds(replaced) == [ip, j]
    @test Tensor4all.inds(tensor) == [i, j]
    @test replaced.data == tensor.data
    @test replaced.backend_handle == handle

    @test Tensor4all.inds(Tensor4all.replaceind(tensor, i => ip)) == [ip, j]
    @test Tensor4all.inds(Tensor4all.replaceinds(tensor, (i, j), (ip, jp))) == [ip, jp]
    @test Tensor4all.inds(Tensor4all.replaceinds(tensor, i => ip, j => jp)) == [ip, jp]
    @test Tensor4all.inds(Tensor4all.replaceind(tensor, missing, ip)) == [i, j]
    @test Tensor4all.inds(Tensor4all.replaceinds(tensor, [missing], [ip])) == [i, j]
    @test_throws ArgumentError Tensor4all.replaceind(tensor, i, bad)
    @test_throws ArgumentError Tensor4all.replaceinds(tensor, (i,), (bad,))

    mut = Tensor4all.Tensor(data, [i, j]; backend_handle=handle)
    @test Tensor4all.replaceind!(mut, i, ip) === mut
    @test Tensor4all.inds(mut) == [ip, j]
    @test mut.data == data
    @test mut.backend_handle == handle

    mut2 = Tensor4all.Tensor(data, [i, j]; backend_handle=handle)
    @test Tensor4all.replaceinds!(mut2, (i, j), (ip, jp)) === mut2
    @test Tensor4all.inds(mut2) == [ip, jp]
    @test mut2.data == data
    @test mut2.backend_handle == handle
end
