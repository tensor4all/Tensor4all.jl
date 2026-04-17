using Test
using Tensor4all
using LinearAlgebra
using LinearAlgebra: norm
using Random: MersenneTwister

const TN_LINSOLVE = Tensor4all.TensorNetworks

# Build a chain MPO that acts as `coeff * I` on `sites`. Each site tensor is
# `coeff^(1/n) * δ(s_in, s_out)` with a dim-1 internal link separating
# adjacent positions, so the global action is `coeff * I`.
function _scaled_identity_mpo(sites::Vector{Index}, coeff::Real=1.0)
    n = length(sites)
    sites_in = sites
    sites_out = [sim(s) for s in sites]
    links = [Index(1; tags=["Link", "l=$i"]) for i in 1:(n - 1)]
    per_site_scale = coeff^(1.0 / n)

    tensors = Tensor[]
    for i in 1:n
        d = dim(sites[i])
        identity_block = per_site_scale * Matrix{Float64}(LinearAlgebra.I, d, d)
        if i == 1 && n == 1
            push!(tensors, Tensor(identity_block, [sites_in[1], sites_out[1]]))
        elseif i == 1
            data = reshape(identity_block, d, d, 1)
            push!(tensors, Tensor(data, [sites_in[1], sites_out[1], links[1]]))
        elseif i == n
            data = reshape(identity_block, 1, d, d)
            push!(tensors, Tensor(data, [links[n - 1], sites_in[n], sites_out[n]]))
        else
            data = reshape(identity_block, 1, d, d, 1)
            push!(tensors, Tensor(data, [links[i - 1], sites_in[i], sites_out[i], links[i]]))
        end
    end
    mpo = TN_LINSOLVE.TensorTrain(tensors, 0, n + 1)

    op = TN_LINSOLVE.LinearOperator(;
        mpo=mpo,
        input_indices=sites_in,
        output_indices=sites_out,
        true_input=Union{Index, Nothing}[s for s in sites_in],
        true_output=Union{Index, Nothing}[s for s in sites_in],
    )
    return op
end

@testset "TensorTrain linsolve" begin
    @testset "identity operator: solution = rhs" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        op = _scaled_identity_mpo(sites, 1.0)
        rhs = TN_LINSOLVE.random_tt(MersenneTwister(1), sites; linkdims=4)

        x = TN_LINSOLVE.linsolve(op, rhs; nfullsweeps=8, krylov_tol=1e-12)
        @test TN_LINSOLVE.to_dense(x) ≈ TN_LINSOLVE.to_dense(rhs)
    end

    @testset "scalar 2 * I: solution = rhs / 2" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:3]
        op = _scaled_identity_mpo(sites, 2.0)
        rhs = TN_LINSOLVE.random_tt(MersenneTwister(2), sites; linkdims=4)

        x = TN_LINSOLVE.linsolve(op, rhs; nfullsweeps=8, krylov_tol=1e-12)
        # Solving (2 I) x = b  =>  x = b / 2.
        expected = 0.5 * TN_LINSOLVE.to_dense(rhs)
        @test TN_LINSOLVE.to_dense(x) ≈ expected
    end

    @testset "argument validation" begin
        sites = [Index(2; tags=["s", "s=$i"]) for i in 1:2]
        op = _scaled_identity_mpo(sites, 1.0)
        rhs = TN_LINSOLVE.random_tt(MersenneTwister(3), sites; linkdims=2)
        empty_tt = TN_LINSOLVE.TensorTrain(Tensor[])

        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, empty_tt)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; nfullsweeps=0)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; krylov_tol=0.0)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; krylov_maxiter=0)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; krylov_dim=0)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; rtol=-1.0)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; convergence_tol=-1.0)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; center_vertex=0)
        @test_throws ArgumentError TN_LINSOLVE.linsolve(op, rhs; center_vertex=99)

        bare_op = TN_LINSOLVE.LinearOperator()
        @test_throws ArgumentError TN_LINSOLVE.linsolve(bare_op, rhs)
    end
end
