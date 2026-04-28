using Test
using Tensor4all
using LinearAlgebra: I

@testset "Tensor skeleton" begin
    i = Tensor4all.Index(2; tags=["i"])
    j = Tensor4all.Index(3; tags=["j"])
    data = reshape(collect(1.0:6.0), 2, 3)
    tensor = Tensor4all.Tensor(data, [i, j])

    @test fieldnames(typeof(tensor)) == (:handle, :inds)
    @test tensor.backend_handle !== nothing
    @test Tensor4all.rank(tensor) == 2
    @test Tensor4all.dims(tensor) == (2, 3)
    @test Tensor4all.inds(tensor) == [i, j]
    @test Tensor4all.inds(Tensor4all.prime(tensor)) == [Tensor4all.prime(i), Tensor4all.prime(j)]
    filled = Tensor4all.Tensor(4.0, i, j)
    @test Tensor4all.inds(filled) == [i, j]
    @test fieldnames(typeof(filled)) == (:handle, :inds)
    @test filled.backend_handle !== nothing
    @test Tensor4all.copy_data(filled) == fill(4.0, 2, 3)
    filled_copy = Tensor4all.copy_data(filled)
    @test filled_copy == fill(4.0, 2, 3)
    filled_copy[1, 1] = -1.0
    @test Tensor4all.copy_data(filled) == fill(4.0, 2, 3)

    bad = PermutedDimsArray(reshape(collect(1.0:8.0), 2, 2, 2), (2, 1, 3))
    k = Tensor4all.Index(2; tags=["k"])
    @test_throws ArgumentError Tensor4all.Tensor(bad, [i, j, k])
    @test_throws DimensionMismatch Tensor4all.Tensor(data, [i])
    contracted = Tensor4all.contract(tensor, tensor)
    @test Tensor4all.rank(contracted) == 0
    @test Tensor4all.dims(contracted) == ()
    @test Tensor4all.copy_data(contracted)[] == 91.0
    @test only(Array(contracted)) == sum(data .* data)
    @test only(Tensor4all.copy_data(contracted)) == sum(data .* data)
end

@testset "Tensor scalar and ITensor conveniences" begin
    i = Index(2; tags=["i"])
    t = Tensor([1.0, 2.0], i)
    @test inds(t) == [i]
    @test eltype(t) == Float64
    @test Tensor4all.ITensor === Tensor4all.Tensor
    @test Tensor4all.scalar(Tensor(3.5)) == 3.5
    @test_throws ArgumentError Tensor4all.scalar(t)
end

@testset "Tensor explicit materialization copies" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data = reshape(collect(1.0:6.0), 2, 3)
    tensor = Tensor(data, [i, j])

    copied = Tensor4all.copy_data(tensor)
    @test copied == data
    copied[1, 1] = -100.0
    @test Tensor4all.copy_data(tensor) == data
    @test Array(tensor) == data

    reordered = Tensor4all.copy_data(tensor, j, i)
    @test reordered == permutedims(data, (2, 1))
    reordered[1, 1] = -200.0
    @test Tensor4all.copy_data(tensor, j, i) == permutedims(data, (2, 1))
    @test Tensor4all.copy_data(tensor, [j, i]) == permutedims(data, (2, 1))
end

@testset "Tensor copy ownership" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data = reshape(collect(1.0:6.0), 2, 3)
    tensor = Tensor(data, [i, j])

    ptr(t) = getfield(getfield(t, :handle), :ptr)

    copied = copy(tensor)
    @test copied !== tensor
    @test ptr(copied) != ptr(tensor)
    @test inds(copied) == inds(tensor)
    @test copy_data(copied) == data

    deepcopied = deepcopy(tensor)
    @test deepcopied !== tensor
    @test ptr(deepcopied) != ptr(tensor)
    @test inds(deepcopied) == inds(tensor)
    @test copy_data(deepcopied) == data

    tt = Tensor4all.TensorNetworks.TensorTrain([tensor])
    deep_tt = deepcopy(tt)
    @test deep_tt !== tt
    @test deep_tt.data[1] !== tt.data[1]
    @test ptr(deep_tt.data[1]) != ptr(tt.data[1])
    @test copy_data(deep_tt.data[1]) == data

    shared_tt = Tensor4all.TensorNetworks.TensorTrain([tensor, tensor])
    deep_shared_tt = deepcopy(shared_tt)
    @test deep_shared_tt.data[1] === deep_shared_tt.data[2]
    @test deep_shared_tt.data[1] !== tensor
    @test ptr(deep_shared_tt.data[1]) != ptr(tensor)

    op = Tensor4all.TensorNetworks.LinearOperator(;
        mpo=tt,
        input_indices=[i],
        output_indices=[j],
    )
    deep_op = deepcopy(op)
    @test deep_op !== op
    @test deep_op.mpo !== op.mpo
    @test ptr(deep_op.mpo.data[1]) != ptr(op.mpo.data[1])
    @test copy_data(deep_op.mpo.data[1]) == data

    scalar_mps = Tensor4all.ITensorCompat.MPS(
        Tensor4all.TensorNetworks.TensorTrain([Tensor(2.0)]),
    )
    deep_mps = deepcopy(scalar_mps)
    @test deep_mps !== scalar_mps
    @test deep_mps.tt !== scalar_mps.tt
    @test ptr(deep_mps.tt.data[1]) != ptr(scalar_mps.tt.data[1])
    @test copy_data(deep_mps.tt.data[1])[] == 2.0

    diag_tensor = delta(i, sim(i))
    copied_diag = copy(diag_tensor)
    deep_diag = deepcopy(diag_tensor)
    @test isdiag(copied_diag)
    @test isdiag(deep_diag)
    @test ptr(copied_diag) != ptr(diag_tensor)
    @test ptr(deep_diag) != ptr(diag_tensor)
    @test structured_storage_info(copied_diag) == structured_storage_info(diag_tensor)
    @test structured_storage_info(deep_diag) == structured_storage_info(diag_tensor)
    @test structured_payload(copied_diag) == structured_payload(diag_tensor)
    @test structured_payload(deep_diag) == structured_payload(diag_tensor)

    @test_throws ArgumentError deepcopy(getfield(tensor, :handle))
end

@testset "Tensor replaceind compatibility" begin
    i = Tensor4all.Index(2; tags=["i"])
    j = Tensor4all.Index(3; tags=["j"])
    ip = Tensor4all.Index(2; tags=["ip"])
    jp = Tensor4all.Index(3; tags=["jp"])
    bad = Tensor4all.Index(5; tags=["bad"])
    missing = Tensor4all.Index(2; tags=["missing"])

    data = reshape(collect(1.0:6.0), 2, 3)
    tensor = Tensor4all.Tensor(data, [i, j])
    handle = tensor.backend_handle

    replaced = Tensor4all.replaceind(tensor, i, ip)
    @test Tensor4all.inds(replaced) == [ip, j]
    @test Tensor4all.inds(tensor) == [i, j]
    @test Tensor4all.copy_data(replaced) == Tensor4all.copy_data(tensor)
    @test replaced.backend_handle !== nothing
    @test replaced.backend_handle != handle

    @test Tensor4all.inds(Tensor4all.replaceind(tensor, i => ip)) == [ip, j]
    @test Tensor4all.inds(Tensor4all.replaceinds(tensor, (i, j), (ip, jp))) == [ip, jp]
    @test Tensor4all.inds(Tensor4all.replaceinds(tensor, i => ip, j => jp)) == [ip, jp]
    @test Tensor4all.inds(Tensor4all.replaceind(tensor, missing, ip)) == [i, j]
    @test Tensor4all.inds(Tensor4all.replaceinds(tensor, [missing], [ip])) == [i, j]
    @test_throws ArgumentError Tensor4all.replaceind(tensor, i, bad)
    @test_throws ArgumentError Tensor4all.replaceinds(tensor, (i,), (bad,))

    mut = Tensor4all.Tensor(data, [i, j])
    mut_handle = mut.backend_handle
    @test Tensor4all.replaceind!(mut, i, ip) === mut
    @test Tensor4all.inds(mut) == [ip, j]
    @test Tensor4all.copy_data(mut) == data
    @test mut.backend_handle !== nothing
    @test mut.backend_handle != mut_handle

    mut2 = Tensor4all.Tensor(data, [i, j])
    mut2_handle = mut2.backend_handle
    @test Tensor4all.replaceinds!(mut2, (i, j), (ip, jp)) === mut2
    @test Tensor4all.inds(mut2) == [ip, jp]
    @test Tensor4all.copy_data(mut2) == data
    @test mut2.backend_handle !== nothing
    @test mut2.backend_handle != mut2_handle
end

@testset "structured diagonal tensor payload metadata" begin
    i = Tensor4all.Index(3; tags=["i"])
    j = Tensor4all.sim(i)

    d = Tensor4all.delta(i, j)
    @test Tensor4all.isdiag(d)
    @test fieldnames(typeof(d)) == (:handle, :inds)
    @test d.backend_handle !== nothing

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
