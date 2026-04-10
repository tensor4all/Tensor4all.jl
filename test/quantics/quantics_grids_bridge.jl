using Test
using Tensor4all
using QuanticsGrids

@testset "QuanticsGrids re-export" begin
    @test Tensor4all.DiscretizedGrid === QuanticsGrids.DiscretizedGrid
    @test Tensor4all.InherentDiscreteGrid === QuanticsGrids.InherentDiscreteGrid

    grid = Tensor4all.DiscretizedGrid((3, 5); unfoldingscheme=:interleaved)
    @test Tensor4all.quantics_to_grididx(grid, [1, 2, 1, 2, 1, 2, 1, 2]) == (1, 30)
    @test Tensor4all.grididx_to_quantics(grid, (1, 30)) == [1, 2, 1, 2, 1, 2, 1, 2]
end
