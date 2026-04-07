using Test
using Tensor4all
using Tensor4all: dim
using Tensor4all.SimpleTT: SimpleTensorTrain
using Tensor4all.TreeTN: MPS, siteinds, to_dense
using Tensor4all.QuanticsTransform:
    AffineParams,
    LinearOperator,
    affine_operator,
    affine_pullback_operator,
    apply,
    binaryop_operator,
    flip_operator_multivar,
    phase_rotation_operator_multivar,
    set_iospaces!,
    shift_operator,
    shift_operator_multivar

const CAPI = Tensor4all.C_API

function _product_mps(vectors::Vector{<:AbstractVector{ComplexF64}})
    nsites = length(vectors)
    site_inds = [Tensor4all.Index(length(vector)) for vector in vectors]
    arrays = Array{ComplexF64}[]

    for (site, vector) in enumerate(vectors)
        if nsites == 1
            push!(arrays, reshape(collect(vector), length(vector)))
        elseif site == 1
            push!(arrays, reshape(collect(vector), length(vector), 1))
        elseif site == nsites
            push!(arrays, reshape(collect(vector), 1, length(vector)))
        else
            push!(arrays, reshape(collect(vector), 1, length(vector), 1))
        end
    end

    return MPS(arrays, site_inds)
end

_dense_state(mps) = ComplexF64.(Tensor4all.data(to_dense(mps)))

function _decode_coords(level_digits::Vector{Int}, nvars::Int)
    r = length(level_digits)
    coords = zeros(Int, nvars)
    for var in 1:nvars
        value = 0
        for digit in level_digits
            value = (value << 1) | ((digit >> (var - 1)) & 1)
        end
        coords[var] = value
    end
    return coords
end

function _encode_coords(coords::Vector{Int}, r::Int)
    digits = zeros(Int, r)
    for level in 1:r
        bitpos = r - level
        digit = 0
        for (var, coord) in enumerate(coords)
            digit |= ((coord >> bitpos) & 1) << (var - 1)
        end
        digits[level] = digit
    end
    return digits
end

function _expected_affine_pullback(source_dense, a::AbstractMatrix{<:Integer},
                                   b::AbstractVector{<:Integer}, bc::Vector)
    source_ndims, output_ndims = size(a)
    r = ndims(source_dense)
    output_dim = 1 << output_ndims
    source_size = 1 << r
    expected = zeros(ComplexF64, ntuple(_ -> output_dim, r))

    for output_index in CartesianIndices(expected)
        output_digits = Int[index - 1 for index in Tuple(output_index)]
        output_coords = _decode_coords(output_digits, output_ndims)
        source_coords = vec(a * output_coords .+ b)

        valid = true
        for i in eachindex(source_coords)
            if bc[i] == Tensor4all.QuanticsTransform.Periodic
                source_coords[i] = mod(source_coords[i], source_size)
            elseif source_coords[i] < 0 || source_coords[i] >= source_size
                valid = false
                break
            end
        end

        if valid
            source_digits = _encode_coords(source_coords, r)
            expected[output_index] = source_dense[CartesianIndex((source_digits .+ 1)...)]
        end
    end

    return expected
end

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
        status = CAPI.t4a_qtransform_affine_pullback(
            Csize_t(4),
            Csize_t(1),
            Csize_t(2),
            Int64[1, 0],
            Int64[1, 1],
            Int64[0],
            Int64[1],
            Cint[1],
            out,
        )
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
        @test affine_pullback_operator(2, AffineParams([1 0; 0 1], [0, 0])) isa LinearOperator
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

    @testset "affine pullback semantics" begin
        @testset "identity" begin
            a = Int64[1 0; 0 1]
            b = Int64[0, 0]
            bc = [Tensor4all.QuanticsTransform.Periodic, Tensor4all.QuanticsTransform.Periodic]

            op = affine_pullback_operator(2, AffineParams(a, b); bc=bc)
            state = _product_mps([
                ComplexF64[1, 2, 3, 4],
                ComplexF64[5, 6, 7, 8],
            ])
            set_iospaces!(op, state)
            result = apply(op, state)

            @test _dense_state(result) ≈ _dense_state(state)
        end

        @testset "2d shear" begin
            a = Int64[1 0; 1 1]
            b = Int64[0, 0]
            bc = [Tensor4all.QuanticsTransform.Periodic, Tensor4all.QuanticsTransform.Periodic]

            op = affine_pullback_operator(2, AffineParams(a, b); bc=bc)
            state = _product_mps([
                ComplexF64[1, 3, 5, 7],
                ComplexF64[2, 4, 6, 8],
            ])
            set_iospaces!(op, state)
            result = apply(op, state)

            source_dense = _dense_state(state)
            expected = _expected_affine_pullback(source_dense, a, b, bc)
            @test _dense_state(result) ≈ expected
        end

        @testset "embedding" begin
            a = reshape(Int64[1, 0], 1, 2)
            b = Int64[0]
            bc = [Tensor4all.QuanticsTransform.Open]

            op = affine_pullback_operator(2, AffineParams(a, b); bc=bc)
            input_state = _product_mps([
                ComplexF64[1, 2],
                ComplexF64[3, 4],
            ])
            output_state = MPS(SimpleTensorTrain(fill(4, 2), 0.0))
            set_iospaces!(op, input_state, output_state)
            result = apply(op, input_state)

            source_dense = _dense_state(input_state)
            expected = _expected_affine_pullback(source_dense, a, b, bc)
            @test _dense_state(result) ≈ expected
        end

        @testset "open shift" begin
            a = reshape(Int64[1], 1, 1)
            b = Int64[1]
            bc = [Tensor4all.QuanticsTransform.Open]

            op = affine_pullback_operator(2, AffineParams(a, b); bc=bc)
            state = _product_mps([
                ComplexF64[1, 2],
                ComplexF64[3, 4],
            ])
            set_iospaces!(op, state)
            result = apply(op, state)

            source_dense = _dense_state(state)
            expected = _expected_affine_pullback(source_dense, a, b, bc)
            @test _dense_state(result) ≈ expected
        end
    end
end
