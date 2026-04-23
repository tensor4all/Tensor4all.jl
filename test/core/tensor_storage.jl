using Test
using Tensor4all
using LinearAlgebra: I

@testset "structured tensor storage C API" begin
    i = Index(3; tags=["i"])
    j = Index(3; tags=["j"])
    d = Tensor4all.diagtensor([1.0, 2.0, 4.0], [i, j])

    @test Tensor4all.storage_kind(d) == :diagonal
    @test Tensor4all.payload_rank(d) == 1
    @test Tensor4all.payload_dims(d) == [3]
    @test Tensor4all.axis_classes(d) == [0, 0]
    @test Tensor4all.payload(d) == [1.0, 2.0, 4.0]
    @test Array(d, i, j) == [
        1.0 0.0 0.0
        0.0 2.0 0.0
        0.0 0.0 4.0
    ]
end

@testset "structured handle survives backend clone" begin
    i = Index(2; tags=["i"])
    j = Index(2; tags=["j"])
    d = Tensor4all.delta(i, j)

    handle = C_NULL
    try
        handle = Tensor4all.TensorNetworks._new_tensor_handle(d, :f64)
        @test Tensor4all.TensorNetworks._storage_kind_from_handle(handle) ==
            Tensor4all.TensorNetworks._T4A_STORAGE_KIND_DIAGONAL
        @test Tensor4all.TensorNetworks._payload_from_handle(handle) == [1.0, 1.0]
    finally
        Tensor4all.TensorNetworks._release_tensor_handle(handle)
    end
end

@testset "delta and identity constructors" begin
    i = Index(3; tags=["i"])
    j = Index(3; tags=["j"])
    k = Index(3; tags=["k"])
    bad = Index(2; tags=["bad"])

    d2 = Tensor4all.delta(i, j)
    @test Tensor4all.storage_kind(d2) == :diagonal
    @test Tensor4all.axis_classes(d2) == [0, 0]
    @test Tensor4all.payload(d2) == ones(3)

    d3 = Tensor4all.delta(i, j, k)
    @test Tensor4all.axis_classes(d3) == [0, 0, 0]
    @test Tensor4all.payload(d3) == ones(3)

    @test Array(Tensor4all.identity_tensor(i, j), i, j) == Matrix{Float64}(I, 3, 3)
    @test_throws DimensionMismatch Tensor4all.delta(i, bad)
end
