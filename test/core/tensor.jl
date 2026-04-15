using Test
using Tensor4all

@testset "Tensor backend wrapper" begin
    i = Tensor4all.Index(2; tags=["i"])
    j = Tensor4all.Index(3; tags=["j"])
    k = Tensor4all.Index(4; tags=["k"])

    data = reshape(collect(1.0:6.0), 2, 3)
    tensor = Tensor4all.Tensor(data, [i, j])
    @test Tensor4all.rank(tensor) == 2
    @test Tensor4all.dims(tensor) == (2, 3)
    @test Tensor4all.inds(tensor) == [i, j]

    primed = Tensor4all.prime(tensor)
    @test Tensor4all.inds(primed) == [Tensor4all.prime(i), Tensor4all.prime(j)]

    swapped = Tensor4all.swapinds(tensor, i, j)
    swapped_data, swapped_inds = Tensor4all._dense_array(swapped)
    @test swapped_inds == [j, i]
    @test swapped_data == permutedims(data, (2, 1))

    a = Tensor4all.Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
    b = Tensor4all.Tensor(reshape(collect(1.0:12.0), 3, 4), [j, k])
    c = Tensor4all.contract(a, b)
    c_data, c_inds = Tensor4all._dense_array(c)
    @test c_inds == [i, k]
    @test c_data == reshape(collect(1.0:6.0), 2, 3) * reshape(collect(1.0:12.0), 3, 4)

    zdata = ComplexF64[1 + 2im 3 + 4im 5 + 6im; 7 + 8im 9 + 10im 11 + 12im]
    ztensor = Tensor4all.Tensor(zdata, [i, j])
    zback, zinds = Tensor4all._dense_array(ztensor)
    @test zinds == [i, j]
    @test zback == zdata

    bad = PermutedDimsArray(reshape(collect(1.0:8.0), 2, 2, 2), (2, 1, 3))
    @test_throws ArgumentError Tensor4all.Tensor(bad, [i, j, k])
    @test_throws DimensionMismatch Tensor4all.Tensor(data, [i])
end
