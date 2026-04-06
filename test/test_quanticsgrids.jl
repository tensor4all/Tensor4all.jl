using Test
using Tensor4all

@testset "QuanticsGrids" begin
    @testset "top-level exports" begin
        @test isdefined(Tensor4all, :DiscretizedGrid)
        @test isdefined(Tensor4all, :InherentDiscreteGrid)
        @test isdefined(Tensor4all, :localdimensions)
    end

    @testset "grouped unfolding" begin
        grid = Tensor4all.QuanticsGrids.DiscretizedGrid(
            2, [2, 2], [0.0, 0.0], [1.0, 1.0]; unfolding=:grouped)

        q = Tensor4all.QuanticsGrids.origcoord_to_quantics(grid, [0.25, 0.75])
        x = Tensor4all.QuanticsGrids.quantics_to_origcoord(grid, q)

        @test length(q) == 4
        @test x ≈ [0.25, 0.75] atol=0.3
        @test Tensor4all.QuanticsGrids.localdimensions(grid) == fill(2, 4)
    end
end
