using Test
using Tensor4all
using LinearAlgebra: norm

const TN_CONTRACT = Tensor4all.TensorNetworks

function _make_mps(sites::Vector{Index}, links::Vector{Index}, eltype::Type=Float64; seed::Integer=1)
    n = length(sites)
    @assert length(links) == n - 1
    rng_state = seed
    rand_array(dims::Tuple) = begin
        rng_state += 1
        if eltype <: Complex
            return ComplexF64.(randn(MersenneTwister(rng_state), Float64, dims), randn(MersenneTwister(rng_state + 100), Float64, dims))
        else
            return randn(MersenneTwister(rng_state), eltype, dims)
        end
    end
    tensors = Tensor[]
    for i in 1:n
        if n == 1
            push!(tensors, Tensor(rand_array((dim(sites[i]),)), [sites[i]]))
        elseif i == 1
            push!(tensors, Tensor(rand_array((dim(sites[i]), dim(links[i]))), [sites[i], links[i]]))
        elseif i == n
            push!(tensors, Tensor(rand_array((dim(links[i-1]), dim(sites[i]))), [links[i-1], sites[i]]))
        else
            push!(tensors, Tensor(rand_array((dim(links[i-1]), dim(sites[i]), dim(links[i]))), [links[i-1], sites[i], links[i]]))
        end
    end
    return TN_CONTRACT.TensorTrain(tensors)
end

using Random: MersenneTwister

@testset "TensorTrain contract" begin
    @testset "MPS · MPS shared sites = scalar" begin
        sites = [Index(2; tags=["s", "s=$n"]) for n in 1:3]
        links_a = [Index(2; tags=["LinkA", "l=$n"]) for n in 1:2]
        links_b = [Index(3; tags=["LinkB", "l=$n"]) for n in 1:2]
        a = _make_mps(sites, links_a; seed=10)
        b = _make_mps(sites, links_b; seed=20)

        result = TN_CONTRACT.contract(a, b)
        # All sites contracted out — result is a scalar TT (one rank-0 tensor).
        @test length(result.data) == 1
        @test rank(result.data[1]) == 0
        # Compare the scalar value against tensor-tensor contraction of the
        # dense forms (this avoids the rank-0 to_dense path which currently
        # mishandles rank-0 inputs in _new_tensor_handle).
        expected = Tensor4all.contract(TN_CONTRACT.to_dense(a), TN_CONTRACT.to_dense(b))
        @test result.data[1].data[] ≈ expected.data[]
    end

    @testset "MPS · MPO leaves output sites" begin
        n = 3
        s_in = [Index(2; tags=["s_in", "s=$i"]) for i in 1:n]
        s_out = [Index(2; tags=["s_out", "s=$i"]) for i in 1:n]
        links_psi = [Index(2; tags=["LinkPsi", "l=$i"]) for i in 1:n-1]
        links_op = [Index(3; tags=["LinkOp", "l=$i"]) for i in 1:n-1]

        psi = _make_mps(s_in, links_psi; seed=30)

        op = TN_CONTRACT.TensorTrain([
            Tensor(randn(MersenneTwister(40), 2, 2, 3), [s_in[1], s_out[1], links_op[1]]),
            Tensor(randn(MersenneTwister(41), 3, 2, 2, 3), [links_op[1], s_in[2], s_out[2], links_op[2]]),
            Tensor(randn(MersenneTwister(42), 3, 2, 2), [links_op[2], s_in[3], s_out[3]]),
        ])

        result = TN_CONTRACT.contract(op, psi)
        result_dense = TN_CONTRACT.to_dense(result)
        expected = Tensor4all.contract(TN_CONTRACT.to_dense(op), TN_CONTRACT.to_dense(psi))
        @test result_dense ≈ expected
    end

    @testset "complex scalars" begin
        sites = [Index(2; tags=["s", "s=$n"]) for n in 1:2]
        links = [Index(2; tags=["L", "l=1"])]
        a = TN_CONTRACT.TensorTrain([
            Tensor(ComplexF64[1.0+0im 0.0+1im; 0.0 1.0], [sites[1], links[1]]),
            Tensor(ComplexF64[1.0 0.0+0.5im; 0.0 1.0], [links[1], sites[2]]),
        ])
        b = TN_CONTRACT.TensorTrain([
            Tensor(ComplexF64[2.0 0.0; 0.0 2.0+1im], [sites[1], links[1]]),
            Tensor(ComplexF64[1.0+0.1im 0.0; 0.0 1.0], [links[1], sites[2]]),
        ])

        result = TN_CONTRACT.contract(a, b)
        result_dense = TN_CONTRACT.to_dense(result)
        expected = Tensor4all.contract(TN_CONTRACT.to_dense(a), TN_CONTRACT.to_dense(b))
        @test result_dense ≈ expected
    end

    @testset "argument validation" begin
        sites = [Index(2; tags=["s", "s=$n"]) for n in 1:2]
        links = [Index(2; tags=["L", "l=1"])]
        a = _make_mps(sites, links; seed=50)
        empty_tt = TN_CONTRACT.TensorTrain(Tensor[])

        @test_throws ArgumentError TN_CONTRACT.contract(empty_tt, a)
        @test_throws ArgumentError TN_CONTRACT.contract(a, empty_tt)
        @test_throws ArgumentError TN_CONTRACT.contract(a, a; threshold=-1.0)
        @test_throws ArgumentError TN_CONTRACT.contract(a, a; maxdim=-1)
        @test_throws ArgumentError TN_CONTRACT.contract(a, a; method=:bogus)
    end

    @testset "method choice runs" begin
        # MPS · MPO so the result has free site indices and we can compare
        # via to_dense without hitting the rank-0 path.
        n = 3
        s_in = [Index(2; tags=["s_in", "s=$i"]) for i in 1:n]
        s_out = [Index(2; tags=["s_out", "s=$i"]) for i in 1:n]
        links_psi = [Index(2; tags=["LinkPsi", "l=$i"]) for i in 1:n-1]
        links_op = [Index(3; tags=["LinkOp", "l=$i"]) for i in 1:n-1]

        psi = _make_mps(s_in, links_psi; seed=60)
        op = TN_CONTRACT.TensorTrain([
            Tensor(randn(MersenneTwister(70), 2, 2, 3), [s_in[1], s_out[1], links_op[1]]),
            Tensor(randn(MersenneTwister(71), 3, 2, 2, 3), [links_op[1], s_in[2], s_out[2], links_op[2]]),
            Tensor(randn(MersenneTwister(72), 3, 2, 2), [links_op[2], s_in[3], s_out[3]]),
        ])

        ref = TN_CONTRACT.to_dense(TN_CONTRACT.contract(op, psi; method=:zipup))
        for m in (:fit, :naive)
            got = TN_CONTRACT.to_dense(TN_CONTRACT.contract(op, psi; method=m))
            @test got ≈ ref
        end
    end

    @testset "svd_policy and qr_rtol" begin
        n = 3
        s_in = [Index(2; tags=["s_in", "sp=$i"]) for i in 1:n]
        s_out = [Index(2; tags=["s_out", "sp=$i"]) for i in 1:n]
        links_psi = [Index(2; tags=["LinkPsi", "sp_l=$i"]) for i in 1:n-1]
        links_op = [Index(3; tags=["LinkOp", "sp_l=$i"]) for i in 1:n-1]
        psi = _make_mps(s_in, links_psi; seed=80)
        op = TN_CONTRACT.TensorTrain([
            Tensor(randn(MersenneTwister(90), 2, 2, 3), [s_in[1], s_out[1], links_op[1]]),
            Tensor(randn(MersenneTwister(91), 3, 2, 2, 3), [links_op[1], s_in[2], s_out[2], links_op[2]]),
            Tensor(randn(MersenneTwister(92), 3, 2, 2), [links_op[2], s_in[3], s_out[3]]),
        ])

        ref = TN_CONTRACT.to_dense(TN_CONTRACT.contract(op, psi; method=:zipup))
        pol = TN_CONTRACT.SvdTruncationPolicy()
        got_pol = TN_CONTRACT.to_dense(
            TN_CONTRACT.contract(op, psi;
                method=:zipup, threshold=1e-12, svd_policy=pol),
        )
        @test got_pol ≈ ref

        # qr_rtol accepted with factorize_alg=:qr.
        got_qr = TN_CONTRACT.to_dense(
            TN_CONTRACT.contract(op, psi; method=:zipup, factorize_alg=:qr, qr_rtol=1e-10),
        )
        @test got_qr ≈ ref

        # Negative threshold rejected.
        @test_throws ArgumentError TN_CONTRACT.contract(op, psi; threshold=-1.0)
    end
end
