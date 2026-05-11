using Test
using Tensor4all
import TensorCrossInterpolation as TCI

@testset "InterpolativeQTT surface" begin
    @test isdefined(Tensor4all, :InterpolativeQTT)
    @test :InterpolativeQTT ∉ Tensor4all.InterpolativeQTT._reexportable_symbols()

    for sym in [
        :LagrangePolynomials, :getChebyshevGrid,
        :interpolatesinglescale, :interpolatemultiscale, :interpolateadaptive,
        :interpolatesinglescale_sparse, :invertqtt, :estimate_interpolation_error,
    ]
        @test isdefined(Tensor4all.InterpolativeQTT, sym)
    end

    for sym in [:Interval, :NInterval, :angular_local_lagrange]
        @test !isdefined(Tensor4all.InterpolativeQTT, sym)
    end
end

@testset "InterpolativeQTT functional" begin
    IQTT = Tensor4all.InterpolativeQTT

    f = x -> exp(-x^2)
    tt = IQTT.interpolatesinglescale(f, -2.0, 2.0, 8, 20)
    @test tt isa TCI.TensorTrain{Float64, 3}
    @test length(tt) == 8

    P = IQTT.getChebyshevGrid(5)
    @test P isa IQTT.LagrangePolynomials{Float64}
    @test length(P.grid) == 6
end
