using Test
using Tensor4all
using LinearAlgebra: I

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
    @test only(Array(contracted)) == sum(data .* data)
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

@testset "structured diagonal tensor payload metadata" begin
    i = Tensor4all.Index(3; tags=["i"])
    j = Tensor4all.sim(i)

    d = Tensor4all.delta(i, j)
    @test Tensor4all.isdiag(d)

    info = Tensor4all.structured_storage_info(d)
    @test info.kind == :diagonal
    @test info.dtype == Float64
    @test info.logical_dims == (3, 3)
    @test info.payload_dims == (3,)
    @test info.payload_strides == (1,)
    @test info.payload_length == 3
    @test info.axis_classes == (1, 1)

    @test Tensor4all.structured_payload(d) == ones(Float64, 3)
    @test Array(d, i, j) == Matrix{Float64}(I, 3, 3)

    handle = Tensor4all.TensorNetworks._new_tensor_handle(d, :f64)
    try
        roundtrip = Tensor4all.TensorNetworks._tensor_from_handle(handle)
        @test Tensor4all.isdiag(roundtrip)
        @test Tensor4all.structured_storage_info(roundtrip) == info
        @test Tensor4all.structured_payload(roundtrip) == ones(Float64, 3)
    finally
        Tensor4all.TensorNetworks._release_tensor_handle(handle)
    end
end

@testset "ITensors-style Tensor primitives" begin
    i = Tensor4all.Index(2, "i")
    j = Tensor4all.Index(3, "j")
    k = Tensor4all.Index(5, "k")
    i2 = Tensor4all.sim(i)

    @test Tensor4all.ITensor === Tensor4all.Tensor

    a = Tensor4all.Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
    a2 = Tensor4all.ITensor(reshape(collect(1.0:6.0), 2, 3), i, j)
    @test Array(a2, i, j) == Array(a, i, j)
    @test Tensor4all.scalar(Tensor4all.ITensor(3.5)) == 3.5

    b = Tensor4all.Tensor(reshape(collect(1.0:15.0), 3, 5), [j, k])

    @test Tensor4all.commoninds(a, b) == [j]
    @test Tensor4all.uniqueinds(a, b) == [i]
    @test Tensor4all.hasinds(a, i)
    @test Tensor4all.hasinds(a, i, j)
    @test !Tensor4all.hasinds(a, k)
    @test eltype(a) == Float64

    c = a * b
    @test c ≈ Tensor4all.contract(a, b)

    scalar_tensor = Tensor4all.Tensor(fill(3.5), Index[])
    @test Tensor4all.scalar(scalar_tensor) == 3.5

    replaced = Tensor4all.replaceind(a, i, i2)
    @test Tensor4all.inds(replaced) == [i2, j]
    @test Tensor4all.inds(a) == [i, j]
    @test Array(replaced, i2, j) == Array(a, i, j)

    missing = Tensor4all.replaceind(a, k, Tensor4all.Index(5, "newk"))
    @test Tensor4all.inds(missing) == Tensor4all.inds(a)

    bad = Tensor4all.Index(4, "bad")
    @test_throws ArgumentError Tensor4all.replaceind(a, i, bad)

    mutable_replaced = Tensor4all.Tensor(Array(a, i, j), [i, j])
    @test Tensor4all.replaceind!(mutable_replaced, i, i2) === mutable_replaced
    @test Tensor4all.inds(mutable_replaced) == [i2, j]
    @test Array(mutable_replaced, i2, j) == Array(a, i, j)

    oh = Tensor4all.onehot(i => 2)
    @test Tensor4all.inds(oh) == [i]
    @test Array(Tensor4all.Tensor(oh), i) == [0.0, 1.0]
    @test Array(Tensor4all.contract(a, oh), j) == Array(a, i, j)[2, :]
    @test Array(Tensor4all.contract(oh, a), j) == Array(a, i, j)[2, :]

    ident = Tensor4all.delta(i, i2)
    @test Array(ident, i, i2) == Matrix{Float64}(I, 2, 2)
end
