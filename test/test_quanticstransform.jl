using Test
using Tensor4all
using Tensor4all: dim
using Tensor4all.SimpleTT: SimpleTensorTrain
using Tensor4all.TreeTN: MPS, siteinds
using Tensor4all.QuanticsTransform:
    LinearOperator,
    affine_operator,
    apply,
    binaryop_operator,
    flip_operator_multivar,
    phase_rotation_operator_multivar,
    set_iospaces!,
    shift_operator,
    shift_operator_multivar

const CAPI = Tensor4all.C_API

@testset "QuanticsTransform C API bindings" begin
    @testset "multivar constructors" begin
        out = Ref{Ptr{Cvoid}}(C_NULL)

        status = CAPI.t4a_qtransform_shift_multivar(
            Csize_t(4), Int64(1), Cint(0), Csize_t(3), Csize_t(1), out)
        @test status == 0
        @test out[] != C_NULL
        CAPI.t4a_linop_release(out[])

        out[] = C_NULL
        status = CAPI.t4a_qtransform_flip_multivar(
            Csize_t(4), Cint(1), Csize_t(3), Csize_t(2), out)
        @test status == 0
        @test out[] != C_NULL
        CAPI.t4a_linop_release(out[])

        out[] = C_NULL
        status = CAPI.t4a_qtransform_phase_rotation_multivar(
            Csize_t(4), Cdouble(pi / 3), Csize_t(3), Csize_t(0), out)
        @test status == 0
        @test out[] != C_NULL
        CAPI.t4a_linop_release(out[])
    end

    @testset "affine and binaryop constructors" begin
        out = Ref{Ptr{Cvoid}}(C_NULL)

        a_num = Int64[1, 1, 0, 0, 1, 1]
        a_den = fill(Int64(1), 6)
        b_num = Int64[0, 0, 0]
        b_den = fill(Int64(1), 3)
        bc = Cint[1, 1, 0]

        status = CAPI.t4a_qtransform_affine(
            Csize_t(4), a_num, a_den, b_num, b_den, Csize_t(3), Csize_t(2), bc, out)
        @test status == 0
        @test out[] != C_NULL
        CAPI.t4a_linop_release(out[])

        out[] = C_NULL
        status = CAPI.t4a_qtransform_binaryop(
            Csize_t(4), Int8(1), Int8(1), Int8(1), Int8(-1), Cint(1), Cint(0), out)
        @test status == 0
        @test out[] != C_NULL
        CAPI.t4a_linop_release(out[])
    end

    @testset "mapping rewrite enables high-level apply" begin
        tt = SimpleTensorTrain([2, 2, 2], 1.0)
        mps = MPS(tt)
        op = shift_operator(3, 1)

        set_iospaces!(op, mps)
        result = apply(op, mps; method=:naive)

        @test result isa Tensor4all.TreeTN.TreeTensorNetwork
    end

    @testset "high-level multivar wrappers construct operators" begin
        @test shift_operator_multivar(3, 1, 2, 0) isa LinearOperator
        @test flip_operator_multivar(3, 2, 1; bc=Tensor4all.QuanticsTransform.Open) isa LinearOperator
        @test phase_rotation_operator_multivar(3, pi / 4, 2, 1) isa LinearOperator
        @test binaryop_operator(3, 1, 1, 1, -1) isa LinearOperator
    end

    @testset "affine wrapper supports explicit output space" begin
        input_mps = MPS(SimpleTensorTrain(fill(4, 3), 1.0))
        output_mps = MPS(SimpleTensorTrain(fill(8, 3), 0.0))

        a_num = Int64[
            1  -1
            1   0
            0   1
        ]
        a_den = ones(Int64, 3, 2)
        b_num = Int64[0, 0, 0]
        b_den = ones(Int64, 3)
        bc = [
            Tensor4all.QuanticsTransform.Open,
            Tensor4all.QuanticsTransform.Periodic,
            Tensor4all.QuanticsTransform.Periodic,
        ]

        op = affine_operator(3, a_num, a_den, b_num, b_den; bc=bc)
        set_iospaces!(op, input_mps, output_mps)
        result = apply(op, input_mps; method=:naive)

        @test result isa Tensor4all.TreeTN.TreeTensorNetwork
        @test dim(siteinds(result, 1)[1]) == 8
    end
end
