using Test
using Tensor4all
using Tensor4all.SimpleTT
import Tensor4all.SimpleTT: sitedims, linkdims, rank, evaluate, sitetensor, fulltensor, scale!
using LinearAlgebra: dot, norm

@testset "SimpleTT" begin
    @testset "Float64" begin
        @testset "Construction" begin
            @testset "constant tensor train" begin
                tt = SimpleTensorTrain([2, 3, 4], 1.5)
                @test length(tt) == 3
                @test sitedims(tt) == [2, 3, 4]
                @test rank(tt) == 1
            end

            @testset "zeros tensor train" begin
                tt = zeros(SimpleTensorTrain, [2, 3])
                @test length(tt) == 2
                @test sitedims(tt) == [2, 3]
                @test sum(tt) == 0.0
            end

            @testset "from site tensors" begin
                t1 = randn(1, 2, 3)
                t2 = randn(3, 4, 1)
                tt = SimpleTensorTrain([t1, t2])
                @test length(tt) == 2
                @test sitedims(tt) == [2, 4]
                @test linkdims(tt) == [3]
            end
        end

        @testset "Accessors" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.0)
            @test length(tt) == 3
            @test sitedims(tt) == [2, 3, 4]

            ldims = linkdims(tt)
            @test length(ldims) == 2  # n_sites - 1
            @test all(d -> d == 1, ldims)  # rank-1 constant

            @test rank(tt) == 1
        end

        @testset "Evaluation (1-indexed)" begin
            tt = SimpleTensorTrain([2, 3, 4], 2.0)

            # 1-indexed evaluation
            @test evaluate(tt, [1, 1, 1]) ≈ 2.0
            @test evaluate(tt, [2, 3, 4]) ≈ 2.0
            @test evaluate(tt, [1, 2, 3]) ≈ 2.0

            # Callable interface (1-indexed)
            @test tt([1, 1, 1]) ≈ 2.0
            @test tt(1, 2, 3) ≈ 2.0
        end

        @testset "1-indexing verification" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.5)

            # evaluate with [1,1,1] should work (first element)
            @test evaluate(tt, [1, 1, 1]) ≈ 1.5

            # sitetensor(tt, 1) returns the first site tensor
            t1 = sitetensor(tt, 1)
            @test size(t1, 2) == 2  # first site has dim 2
        end

        @testset "Site tensor" begin
            tt = SimpleTensorTrain([2, 3], 1.0)

            # First site (1-indexed)
            t1 = sitetensor(tt, 1)
            @test size(t1, 1) == 1  # left dim
            @test size(t1, 2) == 2  # site dim
            @test size(t1, 3) == 1  # right dim (rank-1, single site link)

            # Last site (1-indexed)
            t2 = sitetensor(tt, 2)
            @test size(t2, 1) == 1  # left dim
            @test size(t2, 2) == 3  # site dim
            @test size(t2, 3) == 1  # right dim
        end

        @testset "Arithmetic" begin
            tt1 = SimpleTensorTrain([2, 3], 1.0)
            tt2 = SimpleTensorTrain([2, 3], 2.0)

            # Addition
            tt3 = tt1 + tt2
            @test sum(tt3) ≈ sum(tt1) + sum(tt2)

            # Subtraction
            tt4 = tt1 - tt2
            @test sum(tt4) ≈ sum(tt1) - sum(tt2)

            # Scalar multiplication
            tt5 = 3.0 * tt1
            @test sum(tt5) ≈ 3.0 * sum(tt1)

            tt6 = tt1 * 3.0
            @test sum(tt6) ≈ 3.0 * sum(tt1)

            # Dot product
            @test dot(tt1, tt1) ≈ norm(tt1)^2
            @test dot(tt1, tt2) ≈ 2.0 * dot(tt1, tt1)
        end

        @testset "In-place scale!" begin
            tt = SimpleTensorTrain([2, 3], 1.0)
            original_sum = sum(tt)
            scale!(tt, 2.5)
            @test sum(tt) ≈ 2.5 * original_sum
        end

        @testset "reverse" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.5)
            tt_rev = reverse(tt)
            @test length(tt_rev) == 3
            @test sitedims(tt_rev) == [4, 3, 2]
            @test sum(tt_rev) ≈ sum(tt)
        end

        @testset "fulltensor" begin
            tt = SimpleTensorTrain([2, 3], 1.5)
            arr = fulltensor(tt)
            @test size(arr) == (2, 3)
            @test all(x -> x ≈ 1.5, arr)
        end

        @testset "copy" begin
            tt1 = SimpleTensorTrain([2, 3], 3.0)
            tt2 = copy(tt1)
            @test length(tt2) == length(tt1)
            @test sitedims(tt2) == sitedims(tt1)
            @test sum(tt2) ≈ sum(tt1)

            # Ensure deep copy: modifying tt2 doesn't affect tt1
            scale!(tt2, 0.0)
            @test sum(tt1) ≈ 3.0 * 2 * 3
            @test sum(tt2) ≈ 0.0
        end

        @testset "norm" begin
            tt = SimpleTensorTrain([2, 3], 1.5)
            # norm^2 = sum of squares = 1.5^2 * 2 * 3 = 2.25 * 6 = 13.5
            @test norm(tt) ≈ sqrt(1.5^2 * 2 * 3)
        end

        @testset "sum" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.5)
            @test sum(tt) ≈ 1.5 * 2 * 3 * 4
        end

        @testset "compress!" begin
            # Create a TT by adding two rank-1 TTs (result has rank 2)
            tt1 = SimpleTensorTrain([2, 3, 4], 1.0)
            tt2 = SimpleTensorTrain([2, 3, 4], 2.0)
            tt = tt1 + tt2
            @test rank(tt) == 2

            original_sum = sum(tt)
            compress!(tt; method=:SVD, tolerance=1e-12)
            @test rank(tt) == 1  # Should compress back to rank 1
            @test sum(tt) ≈ original_sum
        end

        @testset "show" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.0)

            io = IOBuffer()
            show(io, tt)
            s = String(take!(io))
            @test occursin("SimpleTensorTrain", s)
            @test occursin("3", s)  # sites

            show(io, MIME"text/plain"(), tt)
            s = String(take!(io))
            @test occursin("Sites:", s)
        end

        @testset "from site tensors - evaluation" begin
            # Create site tensors and verify evaluate matches direct contraction
            t1 = randn(1, 2, 3)
            t2 = randn(3, 4, 1)
            tt = SimpleTensorTrain([t1, t2])

            arr = fulltensor(tt)
            # Check a few evaluations match the full tensor
            for i in 1:2, j in 1:4
                @test evaluate(tt, [i, j]) ≈ arr[i, j]
            end
        end
    end

    @testset "ComplexF64" begin
        @testset "Construction" begin
            @testset "constant tensor train" begin
                tt = SimpleTensorTrain([2, 3], 1.0 + 2.0im)
                @test length(tt) == 2
                @test sitedims(tt) == [2, 3]
                @test rank(tt) == 1
            end

            @testset "zeros tensor train" begin
                tt = zeros(SimpleTensorTrain{ComplexF64}, [2, 3])
                @test length(tt) == 2
                @test sitedims(tt) == [2, 3]
                @test sum(tt) == 0.0 + 0.0im
            end

            @testset "from site tensors" begin
                t1 = randn(ComplexF64, 1, 2, 3)
                t2 = randn(ComplexF64, 3, 4, 1)
                tt = SimpleTensorTrain([t1, t2])
                @test length(tt) == 2
                @test sitedims(tt) == [2, 4]
                @test linkdims(tt) == [3]
            end
        end

        @testset "Accessors" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.0 + 0.0im)
            @test length(tt) == 3
            @test sitedims(tt) == [2, 3, 4]

            ldims = linkdims(tt)
            @test length(ldims) == 2
            @test all(d -> d == 1, ldims)

            @test rank(tt) == 1
        end

        @testset "Evaluation (1-indexed)" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.0 + 2.0im)

            @test evaluate(tt, [1, 1, 1]) ≈ 1.0 + 2.0im
            @test evaluate(tt, [2, 3, 4]) ≈ 1.0 + 2.0im

            # Callable interface
            @test tt([1, 1, 1]) ≈ 1.0 + 2.0im
            @test tt(1, 2, 3) ≈ 1.0 + 2.0im
        end

        @testset "1-indexing verification" begin
            tt = SimpleTensorTrain([2, 3], 1.0 + 2.0im)

            @test evaluate(tt, [1, 1]) ≈ 1.0 + 2.0im

            t1 = sitetensor(tt, 1)
            @test size(t1, 2) == 2  # first site has dim 2
        end

        @testset "Site tensor" begin
            tt = SimpleTensorTrain([2, 3], 1.0 + 0.0im)

            t1 = sitetensor(tt, 1)
            @test size(t1, 1) == 1
            @test size(t1, 2) == 2
            @test size(t1, 3) == 1
            @test eltype(t1) == ComplexF64

            t2 = sitetensor(tt, 2)
            @test size(t2, 1) == 1
            @test size(t2, 2) == 3
            @test size(t2, 3) == 1
            @test eltype(t2) == ComplexF64
        end

        @testset "Arithmetic" begin
            tt1 = SimpleTensorTrain([2, 3], 1.0 + 1.0im)
            tt2 = SimpleTensorTrain([2, 3], 2.0 + 0.5im)

            # Addition
            tt3 = tt1 + tt2
            @test sum(tt3) ≈ sum(tt1) + sum(tt2)

            # Subtraction
            tt4 = tt1 - tt2
            @test sum(tt4) ≈ sum(tt1) - sum(tt2)

            # Scalar multiplication (complex scalar)
            tt5 = (2.0 + 1.0im) * tt1
            @test sum(tt5) ≈ (2.0 + 1.0im) * sum(tt1)

            tt6 = tt1 * (2.0 + 1.0im)
            @test sum(tt6) ≈ (2.0 + 1.0im) * sum(tt1)

            # Dot product: dot(a, b) = sum(conj(a) .* b)
            @test dot(tt1, tt1) ≈ norm(tt1)^2
        end

        @testset "In-place scale!" begin
            tt = SimpleTensorTrain([2, 3], 1.0 + 1.0im)
            original_sum = sum(tt)
            scale!(tt, 2.0 + 0.5im)
            @test sum(tt) ≈ (2.0 + 0.5im) * original_sum
        end

        @testset "reverse" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.0 + 2.0im)
            tt_rev = reverse(tt)
            @test length(tt_rev) == 3
            @test sitedims(tt_rev) == [4, 3, 2]
            @test sum(tt_rev) ≈ sum(tt)
        end

        @testset "fulltensor" begin
            tt = SimpleTensorTrain([2, 3], 1.0 + 2.0im)
            arr = fulltensor(tt)
            @test size(arr) == (2, 3)
            @test eltype(arr) == ComplexF64
            @test all(x -> x ≈ 1.0 + 2.0im, arr)
        end

        @testset "copy" begin
            tt1 = SimpleTensorTrain([2, 3], 1.0 + 2.0im)
            tt2 = copy(tt1)
            @test length(tt2) == length(tt1)
            @test sitedims(tt2) == sitedims(tt1)
            @test sum(tt2) ≈ sum(tt1)

            # Ensure deep copy
            scale!(tt2, 0.0 + 0.0im)
            @test sum(tt1) ≈ (1.0 + 2.0im) * 2 * 3
            @test sum(tt2) ≈ 0.0 + 0.0im
        end

        @testset "norm" begin
            tt = SimpleTensorTrain([2, 3], 1.0 + 2.0im)
            # norm^2 = sum of |z|^2 = |1+2i|^2 * 2 * 3 = 5 * 6 = 30
            @test norm(tt) ≈ sqrt(abs2(1.0 + 2.0im) * 2 * 3)
        end

        @testset "sum" begin
            tt = SimpleTensorTrain([2, 3], 1.0 + 2.0im)
            @test sum(tt) ≈ (1.0 + 2.0im) * 2 * 3
        end

        @testset "compress!" begin
            tt1 = SimpleTensorTrain([2, 3, 4], 1.0 + 0.0im)
            tt2 = SimpleTensorTrain([2, 3, 4], 0.0 + 2.0im)
            tt = tt1 + tt2
            @test rank(tt) == 2

            original_sum = sum(tt)
            compress!(tt; method=:SVD, tolerance=1e-12)
            @test rank(tt) == 1
            @test sum(tt) ≈ original_sum
        end

        @testset "show" begin
            tt = SimpleTensorTrain([2, 3], 1.0 + 2.0im)

            io = IOBuffer()
            show(io, tt)
            s = String(take!(io))
            @test occursin("SimpleTensorTrain", s)
            @test occursin("ComplexF64", s)

            show(io, MIME"text/plain"(), tt)
            s = String(take!(io))
            @test occursin("Sites:", s)
        end

        @testset "from site tensors - evaluation" begin
            t1 = randn(ComplexF64, 1, 2, 3)
            t2 = randn(ComplexF64, 3, 4, 1)
            tt = SimpleTensorTrain([t1, t2])

            arr = fulltensor(tt)
            for i in 1:2, j in 1:4
                @test evaluate(tt, [i, j]) ≈ arr[i, j]
            end
        end
    end
end
