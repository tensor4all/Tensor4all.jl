using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks
const QT = Tensor4all.QuanticsTransform

@testset "QuanticsTransform materialization" begin
    @testset "shift_operator" begin
        @testset "periodic" begin
            op = QT.shift_operator(3, 1; bc=:periodic)
            @test op.mpo !== nothing
            @test length(op.mpo) == 3
            @test length(op.input_indices) == 3
            @test length(op.output_indices) == 3
        end

        @testset "negative offset" begin
            op = QT.shift_operator(3, -1; bc=:periodic)
            @test op.mpo !== nothing
        end

        @testset "open BC" begin
            op = QT.shift_operator(3, 1; bc=:open)
            @test op.mpo !== nothing
        end
    end

    @testset "flip_operator" begin
        @testset "periodic" begin
            op = QT.flip_operator(3; bc=:periodic)
            @test op.mpo !== nothing
            @test length(op.mpo) == 3
        end

        @testset "open" begin
            op = QT.flip_operator(3; bc=:open)
            @test op.mpo !== nothing
        end
    end

    @testset "cumsum_operator" begin
        op = QT.cumsum_operator(3)
        @test op.mpo !== nothing
        @test length(op.mpo) == 3
    end

    @testset "phase_rotation_operator" begin
        for theta in [0.0, pi / 4, pi / 2, pi]
            op = QT.phase_rotation_operator(3, theta)
            @test op.mpo !== nothing
        end
    end

    @testset "fourier_operator" begin
        @testset "forward" begin
            op = QT.fourier_operator(3; forward=true)
            @test op.mpo !== nothing
        end

        @testset "inverse" begin
            op = QT.fourier_operator(3; forward=false)
            @test op.mpo !== nothing
        end

        @testset "with maxbonddim" begin
            op = QT.fourier_operator(3; forward=true, maxbonddim=4)
            @test op.mpo !== nothing
        end
    end

    @testset "affine_operator" begin
        op = QT.affine_operator(3, 1, 1, 1, 1; bc=:periodic)
        @test op.mpo !== nothing
        @test length(op.mpo) == 3
    end

    @testset "multivar operators" begin
        @testset "shift_operator_multivar" begin
            op = QT.shift_operator_multivar(2, 1, 2, 1; bc=:periodic)
            @test op.mpo !== nothing
            @test length(op.mpo) > 0
        end

        @testset "flip_operator_multivar" begin
            op = QT.flip_operator_multivar(2, 2, 2; bc=:periodic)
            @test op.mpo !== nothing
        end

        @testset "phase_rotation_operator_multivar" begin
            op = QT.phase_rotation_operator_multivar(2, pi / 4, 2, 1)
            @test op.mpo !== nothing
        end
    end

    @testset "shift_operator maps 00 to 01 on output indices" begin
        op = QT.shift_operator(2, 1; bc=:periodic)
        TN.set_iospaces!(op, op.input_indices, op.output_indices)

        s_in = op.input_indices
        link = Index(1; tags=["Link", "state-l=1"])
        t1 = Tensor(reshape([1.0, 0.0], 2, 1), [s_in[1], link])
        t2 = Tensor(reshape([1.0, 0.0], 1, 2), [link, s_in[2]])
        state = TN.TensorTrain([t1, t2])
        @test map(only, TN.siteinds(state)) == op.input_indices

        result = TN.apply(op, state)
        @test map(only, TN.siteinds(result)) == op.output_indices
        @test Array(contract(result[1], result[2]), op.output_indices...) ≈ ComplexF64[
            0.0 1.0
            0.0 0.0
        ]
    end

    @testset "deferred operators remain placeholder" begin
        op = QT.binaryop_operator(3, 1, 0, 1, 0)
        @test op.mpo === nothing

        op2 = QT.affine_pullback_operator(3, (;))
        @test op2.mpo === nothing
    end
end
