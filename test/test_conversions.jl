using Test
using Tensor4all
using Tensor4all.SimpleTT
using Tensor4all.TreeTN

import Tensor4all.SimpleTT: sitedims, linkdims, evaluate, sitetensor, fulltensor
import Tensor4all.TreeTN: MPS, nv, inner, linkdims as ttn_linkdims
using LinearAlgebra: norm

@testset "SimpleTT <-> TreeTN Conversions" begin
    @testset "SimpleTT -> MPS -> SimpleTT round-trip" begin
        @testset "rank-1 constant" begin
            tt = SimpleTensorTrain([2, 3, 4], 1.5)
            mps = MPS(tt)
            @test nv(mps) == 3

            tt2 = SimpleTensorTrain(mps)
            @test length(tt2) == 3
            @test sitedims(tt2) == [2, 3, 4]

            # Check values match
            arr1 = fulltensor(tt)
            arr2 = fulltensor(tt2)
            @test arr1 ≈ arr2
        end

        @testset "higher rank" begin
            t1 = randn(1, 2, 3)
            t2 = randn(3, 4, 2)
            t3 = randn(2, 3, 1)
            tt = SimpleTensorTrain([t1, t2, t3])
            mps = MPS(tt)
            @test nv(mps) == 3

            tt2 = SimpleTensorTrain(mps)
            @test length(tt2) == 3
            @test sitedims(tt2) == [2, 4, 3]
            @test linkdims(tt2) == [3, 2]

            arr1 = fulltensor(tt)
            arr2 = fulltensor(tt2)
            @test arr1 ≈ arr2
        end

        @testset "single site" begin
            tt = SimpleTensorTrain([3], 2.0)
            mps = MPS(tt)
            @test nv(mps) == 1

            tt2 = SimpleTensorTrain(mps)
            @test length(tt2) == 1
            @test sitedims(tt2) == [3]

            arr1 = fulltensor(tt)
            arr2 = fulltensor(tt2)
            @test arr1 ≈ arr2
        end

        @testset "two sites" begin
            tt = SimpleTensorTrain([2, 5], 3.0)
            mps = MPS(tt)
            @test nv(mps) == 2

            tt2 = SimpleTensorTrain(mps)
            @test length(tt2) == 2
            @test sitedims(tt2) == [2, 5]

            arr1 = fulltensor(tt)
            arr2 = fulltensor(tt2)
            @test arr1 ≈ arr2
        end
    end

    # ComplexF64 Tensor creation via C API is not yet supported (returns null pointer).
    # Skipping until the Rust C API implements complex tensor creation.
    @testset "ComplexF64 round-trip (skipped: C API lacks c64 Tensor)" begin
        @test_skip tt = SimpleTensorTrain([2, 3, 4], 1.0 + 2.0im)
    end

    @testset "MPS -> SimpleTT -> MPS" begin
        sites = [Tensor4all.Index(2) for _ in 1:4]
        mps = TreeTN.random_mps(sites; linkdims=3)

        tt = SimpleTensorTrain(mps)
        @test length(tt) == 4
        @test sitedims(tt) == [2, 2, 2, 2]

        mps2 = MPS(tt)
        @test nv(mps2) == 4

        # Check that the dense tensor representation matches
        dense1 = TreeTN.to_dense(mps)
        arr1 = data(dense1)
        arr2 = fulltensor(tt)
        @test arr1 ≈ arr2
    end
end
