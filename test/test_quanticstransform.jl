using Test
using Tensor4all
using Tensor4all.QuanticsTransform
using Tensor4all.TreeTN: MPS, to_dense

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

function _expected_pullback(source_dense, a::AbstractMatrix{<:Integer}, b::AbstractVector{<:Integer}, bc::Vector{BoundaryCondition})
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
            if bc[i] == Periodic
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

@testset "QuanticsTransform" begin
    @testset "affine pullback identity" begin
        params = AffineParams([1 0; 0 1], [0, 0])
        op = affine_pullback_operator(2, params; bc=[Periodic, Periodic])
        state = _product_mps([
            ComplexF64[1, 2, 3, 4],
            ComplexF64[5, 6, 7, 8],
        ])
        result = apply(op, state)
        @test _dense_state(result) ≈ _dense_state(state)
    end

    @testset "affine pullback 2d shear" begin
        a = [1 0; 1 1]
        b = [0, 0]
        bc = [Periodic, Periodic]
        params = AffineParams(a, b)
        op = affine_pullback_operator(2, params; bc=bc)
        state = _product_mps([
            ComplexF64[1, 3, 5, 7],
            ComplexF64[2, 4, 6, 8],
        ])
        result = apply(op, state)

        source_dense = _dense_state(state)
        expected = _expected_pullback(source_dense, a, b, bc)
        @test _dense_state(result) ≈ expected
    end

    @testset "affine pullback embedding" begin
        a = reshape([1, 0], 1, 2)
        b = [0]
        bc = [Open]
        params = AffineParams(a, b)
        op = affine_pullback_operator(2, params; bc=bc)
        state = _product_mps([
            ComplexF64[1, 2],
            ComplexF64[3, 4],
        ])
        result = apply(op, state)

        source_dense = _dense_state(state)
        expected = _expected_pullback(source_dense, a, b, bc)
        @test _dense_state(result) ≈ expected
    end

    @testset "affine pullback open shift" begin
        a = reshape([1], 1, 1)
        b = [1]
        bc = [Open]
        params = AffineParams(a, b)
        op = affine_pullback_operator(2, params; bc=bc)
        state = _product_mps([
            ComplexF64[1, 2],
            ComplexF64[3, 4],
        ])
        result = apply(op, state)

        source_dense = _dense_state(state)
        expected = _expected_pullback(source_dense, a, b, bc)
        @test _dense_state(result) ≈ expected
    end
end