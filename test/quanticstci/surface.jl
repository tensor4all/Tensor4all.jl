using Test
using Tensor4all

@testset "QuanticsTCI surface" begin
    @test isdefined(Tensor4all, :QuanticsTCI)
    @test :QuanticsTCI ∉ Tensor4all.QuanticsTCI._reexportable_symbols()
end
