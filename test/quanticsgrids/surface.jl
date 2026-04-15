using Test
using Tensor4all

@testset "QuanticsGrids surface" begin
    @test isdefined(Tensor4all, :QuanticsGrids)
    @test :QuanticsGrids ∉ Tensor4all.QuanticsGrids._reexportable_symbols()
end
