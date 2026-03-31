using Test
using Tensor4all
using Tensor4all.QuanticsGrids
using Tensor4all.QuanticsTCI
using Tensor4all.SimpleTT: SimpleTensorTrain
import Tensor4all: evaluate, linkdims, maxbonderror, maxrank
import Tensor4all.QuanticsTCI: integral, to_tensor_train

@testset "QuanticsTCI" begin
    @testset "Float64 - continuous grid" begin
        # Create a 1D grid with 8 bits (256 points) over [0, 1)
        grid = DiscretizedGrid(1, 8, [0.0], [1.0])

        # Interpolate f(x) = x^2
        qtci, ranks, errors = quanticscrossinterpolate(Float64, x -> x^2, grid;
            tolerance=1e-8, maxiter=50)

        @testset "return types" begin
            @test qtci isa QuanticsTensorCI2{Float64}
            @test ranks isa Vector{Int}
            @test errors isa Vector{Float64}
            @test length(ranks) > 0
            @test length(errors) > 0
            @test length(ranks) == length(errors)
        end

        @testset "evaluate" begin
            # Evaluate at a known point using quantics indices
            qi = origcoord_to_quantics(grid, [0.5])
            val = evaluate(qtci, qi)
            @test val isa Float64
            @test val ≈ 0.25 atol=1e-4
        end

        @testset "callable interface" begin
            qi = origcoord_to_quantics(grid, [0.25])
            val = qtci(qi...)
            @test val isa Float64
            @test val ≈ 0.0625 atol=1e-4
        end

        @testset "sum" begin
            s = sum(qtci)
            @test s isa Float64
            @test isfinite(s)
        end

        @testset "integral" begin
            # integral of x^2 from 0 to 1 = 1/3
            val = integral(qtci)
            @test val isa Float64
            @test val ≈ 1/3 atol=1e-3
        end

        @testset "maxbonderror" begin
            err = maxbonderror(qtci)
            @test err isa Float64
            @test err >= 0.0
        end

        @testset "maxrank" begin
            r = maxrank(qtci)
            @test r isa Int
            @test r >= 1
        end

        @testset "linkdims" begin
            ld = linkdims(qtci)
            @test ld isa Vector{Int}
            @test all(d -> d >= 1, ld)
        end

        @testset "to_tensor_train" begin
            tt = to_tensor_train(qtci)
            @test tt isa SimpleTensorTrain{Float64}
        end

        @testset "display" begin
            buf = IOBuffer()
            show(buf, qtci)
            s = String(take!(buf))
            @test occursin("QuanticsTensorCI2{Float64}", s)
        end
    end

    @testset "Float64 - discrete (size tuple)" begin
        # Interpolate f(i, j) = i + j on an 8x8 grid
        qtci, ranks, errors = quanticscrossinterpolate(
            Float64, (i, j) -> Float64(i + j), (8, 8);
            tolerance=1e-10, maxiter=50)

        @test qtci isa QuanticsTensorCI2{Float64}
        @test length(ranks) > 0
        @test length(errors) > 0

        # The function i+j should be exactly representable with low rank
        @test maxrank(qtci) <= 4

        @testset "linkdims" begin
            ld = linkdims(qtci)
            @test ld isa Vector{Int}
        end
    end

    @testset "Float64 - from Array" begin
        # Create a simple 8x8 array
        F = Float64[i + j for i in 1:8, j in 1:8]
        qtci, ranks, errors = quanticscrossinterpolate(F;
            tolerance=1e-10, maxiter=50)

        @test qtci isa QuanticsTensorCI2{Float64}
        @test length(ranks) > 0
    end

    @testset "ComplexF64 - continuous grid" begin
        grid = DiscretizedGrid(1, 8, [0.0], [1.0])

        # Interpolate f(x) = exp(im * x)
        qtci, ranks, errors = quanticscrossinterpolate(
            ComplexF64, x -> exp(im * x), grid;
            tolerance=1e-8, maxiter=50)

        @test qtci isa QuanticsTensorCI2{ComplexF64}
        @test ranks isa Vector{Int}
        @test errors isa Vector{Float64}

        @testset "evaluate" begin
            qi = origcoord_to_quantics(grid, [0.5])
            val = evaluate(qtci, qi)
            @test val isa ComplexF64
            @test val ≈ exp(im * 0.5) atol=1e-4
        end

        @testset "sum" begin
            s = sum(qtci)
            @test s isa ComplexF64
        end

        @testset "integral" begin
            val = integral(qtci)
            @test val isa ComplexF64
        end

        @testset "to_tensor_train" begin
            tt = to_tensor_train(qtci)
            @test tt isa SimpleTensorTrain{ComplexF64}
        end

        @testset "maxbonderror and maxrank" begin
            @test maxbonderror(qtci) isa Float64
            @test maxrank(qtci) isa Int
        end

        @testset "display" begin
            buf = IOBuffer()
            show(buf, qtci)
            s = String(take!(buf))
            @test occursin("QuanticsTensorCI2{ComplexF64}", s)
        end
    end

    @testset "ComplexF64 - discrete (size tuple)" begin
        qtci, ranks, errors = quanticscrossinterpolate(
            ComplexF64, (i, j) -> ComplexF64(i + im * j), (8, 8);
            tolerance=1e-8, maxiter=50)

        @test qtci isa QuanticsTensorCI2{ComplexF64}
        @test length(ranks) > 0
    end

    @testset "kwargs: options are passed" begin
        grid = DiscretizedGrid(1, 4, [0.0], [1.0])

        # Test that verbosity and other kwargs don't error
        qtci, ranks, errors = quanticscrossinterpolate(
            Float64, x -> x, grid;
            tolerance=1e-4,
            maxbonddim=10,
            maxiter=5,
            nrandominitpivot=2,
            verbosity=0,
            nsearchglobalpivot=2,
            nsearch=10,
            normalizeerror=false)

        @test qtci isa QuanticsTensorCI2{Float64}
        # With maxiter=5, we should have at most 5 iterations
        @test length(ranks) <= 5
    end
end
