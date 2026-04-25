using Test
using Tensor4all

const TN = Tensor4all.TensorNetworks
const QT = Tensor4all.QuanticsTransform

function _operator_dense(op::TN.LinearOperator)
    return Array(TN.to_dense(op.mpo), op.output_indices..., op.input_indices...)
end

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

        transposed = transpose(op)
        @test transposed.input_indices == op.output_indices
        @test transposed.output_indices == op.input_indices
        dense = _operator_dense(op)
        nsites = length(op.input_indices)
        @test _operator_dense(transposed) ≈ permutedims(
            dense,
            Tuple([collect((nsites + 1):(2 * nsites)); collect(1:nsites)]),
        )
    end

    @testset "affine_operator_multivar" begin
        # 2D swap matrix A = [[0, 1], [1, 0]], b = 0. Forward swap of variables.
        op_swap = QT.affine_operator_multivar(
            1,
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 0],
            [1, 1],
            2,
            2;
            bc=[:periodic, :periodic],
        )
        @test op_swap.mpo !== nothing
        @test length(op_swap.input_indices) == length(op_swap.output_indices)

        # Argument validation
        @test_throws DimensionMismatch QT.affine_operator_multivar(
            1, [0, 1], [1, 1], [0], [1], 2, 2,
        )  # a wrong length
        @test_throws ArgumentError QT.affine_operator_multivar(
            1, [0], [1], [0], [1], 0, 1,
        )  # m == 0
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

    @testset "binaryop_operator" begin
        op = QT.binaryop_operator(3, 1, 0, 1, 0)
        @test op.mpo !== nothing

        op_multi = QT.binaryop_operator_multivar(3, 1, 0, 1, 0, 3, 1, 2)
        @test op_multi.mpo !== nothing

        @test_throws ArgumentError QT.binaryop_operator_multivar(3, 1, 0, 1, 0, 2, 1, 1)
        # Coefficient out of Int8 range
        @test_throws ArgumentError QT.binaryop_operator(3, 200, 0, 1, 0)

        @testset "2-var shape matches affine_pullback_operator_multivar" begin
            r = 3
            op_bin = QT.binaryop_operator(r, 0, 1, 1, 0)
            op_aff = QT.affine_pullback_operator_multivar(
                r, [0, 1, 1, 0], [1, 1, 1, 1], [0, 0], [1, 1], 2, 2;
                bc=[:periodic, :periodic],
            )
            @test length(op_bin.mpo) == length(op_aff.mpo)
            @test length(op_bin.input_indices) == length(op_aff.input_indices)
            @test length(op_bin.output_indices) == length(op_aff.output_indices)
            bin_dims = [dim(i) for i in op_bin.input_indices]
            aff_dims = [dim(i) for i in op_aff.input_indices]
            @test bin_dims == aff_dims
        end

        @testset "multivar shape" begin
            r = 2
            op_bin = QT.binaryop_operator_multivar(r, 0, 1, 1, 0, 3, 1, 3)
            # 3-variable Fused layout: r sites, each input/output dim 2^3 = 8.
            @test length(op_bin.mpo) == r
            @test length(op_bin.input_indices) == r
            @test all(dim(i) == 8 for i in op_bin.input_indices)
            @test all(dim(i) == 8 for i in op_bin.output_indices)
        end
    end

    @testset "affine_pullback_operator" begin
        # 1D pullback (a = 1, b = 0) — identity-like, layout single-var.
        op = QT.affine_pullback_operator(3, 1, 1, 0, 1)
        @test op.mpo !== nothing

        # 2D pullback with swap matrix A = [[0,1],[1,0]], b = 0.
        op_swap = QT.affine_pullback_operator_multivar(
            1,
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 0],
            [1, 1],
            2,
            2;
            bc=[:periodic, :periodic],
        )
        @test op_swap.mpo !== nothing

        # Argument validation
        @test_throws DimensionMismatch QT.affine_pullback_operator_multivar(
            1, [0, 1], [1, 1], [0], [1], 2, 2,
        )  # a wrong length
        @test_throws ArgumentError QT.affine_pullback_operator_multivar(
            1, [0], [1], [0], [1], 0, 1,
        )  # m == 0

        @testset "pullback == transpose(forward) via field swap" begin
            # Simple 1D forward y = x + 3, r=2 (dim 4).
            r = 2
            a_num, a_den, b_num, b_den = 1, 1, 3, 1
            forward = QT.affine_operator(r, a_num, a_den, b_num, b_den; bc=:periodic)
            pullback = QT.affine_pullback_operator(r, a_num, a_den, b_num, b_den; bc=:periodic)

            # pullback.input_indices should correspond to forward.output_indices (swap).
            @test length(pullback.input_indices) == length(forward.output_indices)
            @test length(pullback.output_indices) == length(forward.input_indices)
            # Per-site dimensions consistent with transpose.
            @test [dim(i) for i in pullback.input_indices] ==
                  [dim(i) for i in forward.output_indices]
            @test [dim(i) for i in pullback.output_indices] ==
                  [dim(i) for i in forward.input_indices]
        end
    end

    @testset "unfuse_quantics_operator" begin
        r = 2
        nvars = 2
        op = QT.shift_operator_multivar(r, 1, nvars, 1; bc=:periodic)
        input_sites = [Index(2; tags=["qin", "bit=$bit", "var=$var"]) for bit in 1:r for var in 1:nvars]
        output_sites = [Index(2; tags=["qout", "bit=$bit", "var=$var"]) for bit in 1:r for var in 1:nvars]

        unfused = QT.unfuse_quantics_operator(op, input_sites, output_sites; base=2)

        @test unfused isa TN.LinearOperator
        @test length(unfused.mpo) == r * nvars
        @test unfused.input_indices == input_sites
        @test unfused.output_indices == output_sites
        @test length(unfused.input_indices) == length(unfused.output_indices) == length(unfused.true_input) == length(unfused.true_output)
        @test all(isnothing, unfused.true_input)
        @test all(dim(index) == 2 for index in unfused.input_indices)
        @test all(dim(index) == 2 for index in unfused.output_indices)

        expected = reshape(_operator_dense(op), ntuple(_ -> 2, 2 * r * nvars))
        @test _operator_dense(unfused) ≈ expected
    end

    @testset "unfuse_quantics_operator documents fused-index encoding" begin
        fused_input = Index(4; tags=["fused-in"])
        fused_output = Index(4; tags=["fused-out"])
        data = zeros(Float64, 4, 4)
        data[3, 2] = 1.0
        op = TN.LinearOperator(;
            mpo=TN.TensorTrain([Tensor(data, [fused_output, fused_input])]),
            input_indices=[fused_input],
            output_indices=[fused_output],
        )
        input_sites = [Index(2; tags=["in", "var=$var"]) for var in 1:2]
        output_sites = [Index(2; tags=["out", "var=$var"]) for var in 1:2]

        unfused = QT.unfuse_quantics_operator(op, input_sites, output_sites; base=2)
        dense = _operator_dense(unfused)

        @test dense[1, 2, 2, 1] == 1.0
        @test count(!iszero, dense) == 1
    end
end
