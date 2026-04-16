using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks

function _tt_linkdims(tt::TN.TensorTrain)
    dims = Int[]
    for position in 1:(length(tt) - 1)
        shared = commoninds(inds(tt[position]), inds(tt[position + 1]))
        links = [index for index in shared if hastag(index, "Link")]
        isempty(links) && continue
        push!(dims, dim(only(links)))
    end
    return dims
end

@testset "TensorTrain orthogonalize and truncate" begin
    s1 = Index(2; tags=["s1"])
    s2 = Index(2; tags=["s2"])
    s3 = Index(2; tags=["s3"])
    l1 = Index(4; tags=["Link", "l1"])
    l2 = Index(4; tags=["Link", "l2"])

    t1 = Tensor(randn(2, 4), [s1, l1])
    t2 = Tensor(randn(4, 2, 4), [l1, s2, l2])
    t3 = Tensor(randn(4, 2), [l2, s3])
    tt = TN.TensorTrain([t1, t2, t3])

    tt2 = TN.orthogonalize(tt, 2)
    @test tt2.llim == 1
    @test tt2.rlim == 3
    @test tt.llim == 0
    @test tt.rlim == 4

    n1 = TN.norm(tt)
    n2 = TN.norm(TN.orthogonalize(tt, 1))
    @test n1 ≈ n2 rtol=1e-10

    tt_trunc = TN.truncate(tt; maxdim=2)
    @test all(d -> d <= 2, _tt_linkdims(tt_trunc))

    @test_throws ArgumentError TN.orthogonalize(TN.TensorTrain(Tensor[]), 1)
    @test_throws ArgumentError TN.orthogonalize(tt, 0)
    @test_throws ArgumentError TN.truncate(TN.TensorTrain(Tensor[]); maxdim=2)
    @test_throws ArgumentError TN.truncate(tt)
end
