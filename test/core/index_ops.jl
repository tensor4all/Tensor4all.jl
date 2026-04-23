using Test
using Tensor4all

@testset "Tensor index fixing and summation" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    t = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])

    @test inds(Tensor4all.onehot(i => 2)) == [i]
    @test Array(Tensor(Tensor4all.onehot(i => 2)), i) == [0.0, 1.0]
    @test Array(Tensor4all.fixinds(t, i => 2), j) == Array(t, i, j)[2, :]
    @test Array(Tensor4all.suminds(t, i), j) == vec(sum(Array(t, i, j); dims=1))

    p = Tensor4all.projectinds(t, i => [2])
    @test dims(p) == (1, 3)
    @test dim(inds(p)[1]) == 1
end
