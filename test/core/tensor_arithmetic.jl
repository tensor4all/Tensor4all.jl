using Test
using Tensor4all

@testset "Tensor index permutation matching" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data_a = reshape(collect(1.0:6.0), 2, 3)
    data_b = permutedims(data_a, (2, 1))

    a = Tensor(data_a, [i, j])
    b = Tensor(data_b, [j, i])

    c = a + b
    @test inds(c) == [i, j]
    @test c.data == 2.0 .* data_a
end
