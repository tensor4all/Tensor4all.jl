using Test
using Tensor4all

@testset "Tensor dag" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data = reshape(
        ComplexF64[1 + 2im, 3 + 4im, 5 + 6im, 7 + 8im, 9 + 10im, 11 + 12im],
        2,
        3,
    )
    t = Tensor(data, [i, j])
    d = dag(t)

    @test inds(d) == [i, j]
    @test d.data ≈ conj(data)

    t_real = Tensor(reshape(collect(1.0:6.0), 2, 3), [i, j])
    @test dag(t_real).data ≈ t_real.data
end

@testset "Tensor Array with index reordering" begin
    i = Index(2; tags=["i"])
    j = Index(3; tags=["j"])
    data = reshape(collect(1.0:6.0), 2, 3)
    t = Tensor(data, [i, j])

    @test Array(t, i, j) ≈ data
    @test Array(t, j, i) ≈ permutedims(data, (2, 1))
    @test size(Array(t, j, i)) == (3, 2)
end

@testset "Tensor SVD" begin
    i = Index(3; tags=["i"])
    j = Index(4; tags=["j"])
    data = reshape(collect(1.0:12.0), 3, 4)
    t = Tensor(data, [i, j])

    U, S, V = svd(t, [i])
    reconstructed = contract(contract(U, S), dag(V))
    @test isapprox(t, reconstructed; atol=1e-12)

    U2, S2, V2 = svd(t, i)
    @test isapprox(contract(contract(U2, S2), dag(V2)), t; atol=1e-12)

    v1 = collect(1.0:3.0)
    v2 = collect(1.0:4.0)
    data_r1 = v1 * v2' + 1e-10 * randn(3, 4)
    t_r1 = Tensor(data_r1, [i, j])
    _, S3, _ = svd(t_r1, [i]; maxdim=1)
    @test dims(S3) == (1, 1)

    k = Index(2; tags=["k"])
    t3 = Tensor(reshape(collect(1.0:24.0), 3, 4, 2), [i, j, k])
    U4, S4, V4 = svd(t3, [i, j])
    @test isapprox(t3, contract(contract(U4, S4), dag(V4)); atol=1e-10)

    @test_throws ArgumentError svd(t, Index[])
    @test_throws ArgumentError svd(t, [Index(5; tags=["x"])])
    @test_throws ArgumentError svd(t, [i, j])
end

@testset "Tensor SVD with svd_policy" begin
    TN = Tensor4all.TensorNetworks
    i = Index(3; tags=["i"])
    j = Index(4; tags=["j"])
    data = reshape(collect(1.0:12.0), 3, 4)
    t = Tensor(data, [i, j])

    # convenience rtol path and full-policy path with matching threshold
    # must produce identical singular values.
    _, S_rtol, _ = svd(t, [i]; rtol=1e-6)
    _, S_pol, _ = svd(t, [i]; svd_policy=TN.SvdTruncationPolicy(threshold=1e-6))
    @test dims(S_rtol) == dims(S_pol)
    @test isapprox(S_rtol.data, S_pol.data; atol=1e-14)

    # full-policy strategies not reachable from rtol/cutoff: absolute scale
    # with squared-value measure and discarded-tail-sum rule.
    pol = TN.SvdTruncationPolicy(
        threshold=1e-4,
        scale=:absolute,
        measure=:squared_value,
        rule=:discarded_tail_sum,
    )
    U, S, V = svd(t, [i]; svd_policy=pol)
    @test isapprox(t, contract(contract(U, S), dag(V)); atol=5e-2)

    # Ambiguity rejection
    @test_throws ArgumentError svd(t, [i];
        rtol=1e-6, svd_policy=TN.SvdTruncationPolicy(threshold=1e-6))
end

@testset "Tensor QR" begin
    i = Index(3; tags=["i"])
    j = Index(4; tags=["j"])
    t = Tensor(reshape(collect(1.0:12.0), 3, 4), [i, j])

    Q, R = qr(t, [i])
    @test isapprox(t, contract(Q, R); atol=1e-12)

    Q2, R2 = qr(t, i)
    @test isapprox(t, contract(Q2, R2); atol=1e-12)

    k = Index(2; tags=["k"])
    t3 = Tensor(reshape(collect(1.0:24.0), 3, 4, 2), [i, j, k])
    Q3, R3 = qr(t3, [i, j])
    @test isapprox(t3, contract(Q3, R3); atol=1e-10)

    @test_throws ArgumentError qr(t, Index[])
    @test_throws ArgumentError qr(t, [Index(5; tags=["x"])])
    @test_throws ArgumentError qr(t, [i, j])
end
