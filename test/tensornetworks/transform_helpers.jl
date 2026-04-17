using Test
using Tensor4all
using Random: MersenneTwister

const TN = Tensor4all.TensorNetworks

_prod_dims(xs) = isempty(xs) ? 1 : prod(xs)

function _dense_contract(
    a::AbstractArray,
    ainds::Vector{Index},
    b::AbstractArray,
    binds::Vector{Index},
)
    common = [index for index in ainds if index in binds]
    a_common_axes = [findfirst(==(index), ainds) for index in common]
    b_common_axes = [findfirst(==(index), binds) for index in common]
    a_rest_axes = [axis for axis in eachindex(ainds) if axis ∉ a_common_axes]
    b_rest_axes = [axis for axis in eachindex(binds) if axis ∉ b_common_axes]

    amat = reshape(
        permutedims(a, (a_rest_axes..., a_common_axes...)),
        _prod_dims(size(a)[a_rest_axes]),
        _prod_dims(size(a)[a_common_axes]),
    )
    bmat = reshape(
        permutedims(b, (b_common_axes..., b_rest_axes...)),
        _prod_dims(size(b)[b_common_axes]),
        _prod_dims(size(b)[b_rest_axes]),
    )

    data = reshape(
        amat * bmat,
        size(a)[a_rest_axes]...,
        size(b)[b_rest_axes]...,
    )
    return data, [ainds[a_rest_axes]..., binds[b_rest_axes]...]
end

function dense_tensor(tt::TN.TensorTrain, target_inds::Vector{Index})
    data = copy(tt[1].data)
    current_inds = inds(tt[1])
    for n in 2:length(tt)
        data, current_inds = _dense_contract(data, current_inds, tt[n].data, inds(tt[n]))
    end

    boundary_axes = [axis for (axis, index) in pairs(current_inds) if hastag(index, "Link")]
    for axis in boundary_axes
        dim(current_inds[axis]) == 1 || error("Uncontracted nontrivial link index $(current_inds[axis])")
    end
    if !isempty(boundary_axes)
        data = dropdims(data; dims=Tuple(boundary_axes))
        current_inds = [index for index in current_inds if !hastag(index, "Link")]
    end

    permutation = map(target_inds) do index
        axis = findfirst(==(index), current_inds)
        axis === nothing && error("Target index $index not found in dense tensor")
        axis
    end
    return permutedims(data, Tuple(permutation))
end

function siteinds_by_tensor(tt::TN.TensorTrain)
    counts = Dict{Index, Int}()
    for tensor in tt
        for index in inds(tensor)
            counts[index] = get(counts, index, 0) + 1
        end
    end
    return [
        [index for index in inds(tensor) if get(counts, index, 0) == 1 && !hastag(index, "Link")]
        for tensor in tt
    ]
end

function simple_mps(sites::Vector{Index})
    # Mirrors the ITensorMPS layout: rank-2 boundary tensors (site + one
    # internal link) and rank-3 interior tensors (left link, site, right
    # link). No dim-1 boundary links — those are not part of the
    # ITensorMPS MPS data structure.
    n = length(sites)
    if n == 1
        return TN.TensorTrain([Tensor(collect(1.0:dim(sites[1])), [sites[1]])], 0, 2)
    end

    links = [Index(2; tags=["Link", "l=$i"]) for i in 1:(n - 1)]
    cursor = 1.0
    tensors = Tensor[]
    for i in 1:n
        if i == 1
            elements = dim(sites[1]) * dim(links[1])
            data = reshape(collect(cursor:(cursor + elements - 1)), dim(sites[1]), dim(links[1]))
            push!(tensors, Tensor(data, [sites[1], links[1]]))
        elseif i == n
            elements = dim(links[n - 1]) * dim(sites[n])
            data = reshape(collect(cursor:(cursor + elements - 1)), dim(links[n - 1]), dim(sites[n]))
            push!(tensors, Tensor(data, [links[n - 1], sites[n]]))
        else
            elements = dim(links[i - 1]) * dim(sites[i]) * dim(links[i])
            data = reshape(collect(cursor:(cursor + elements - 1)), dim(links[i - 1]), dim(sites[i]), dim(links[i]))
            push!(tensors, Tensor(data, [links[i - 1], sites[i], links[i]]))
        end
        cursor += length(tensors[end].data)
    end
    return TN.TensorTrain(tensors, 0, n + 1)
end

@testset "TensorNetworks transform helpers" begin
    @testset "replace_siteinds_part! mutates only the requested subset" begin
        sites = [Index(2; tags=["x", "x=$n"]) for n in 1:3]
        tt = simple_mps(sites)
        newsite = Index(2; tags=["y", "y=2"])

        @test TN.replace_siteinds_part!(tt, [sites[2]], [newsite]) === tt
        @test siteinds_by_tensor(tt) == [[sites[1]], [newsite], [sites[3]]]

        @test_throws DimensionMismatch TN.replace_siteinds_part!(tt, [sites[1]], [newsite, sim(newsite)])
        @test_throws ArgumentError TN.replace_siteinds_part!(tt, [Index(2; tags=["missing", "missing=1"])], [newsite])
    end

    @testset "rearrange_siteinds delegates to restructure_to" begin
        sitesx = [Index(2; tags=["x", "x=$n"]) for n in 1:3]
        sitesy = [Index(2; tags=["y", "y=$n"]) for n in 1:3]
        sitesxy = collect(Iterators.flatten(zip(sitesx, sitesy)))
        psi = simple_mps(sitesxy)

        before = TN.to_dense(psi)
        # Interleaved -> fused (each adjacent pair becomes one node).
        fused_groups = [[x, y] for (x, y) in zip(sitesx, sitesy)]
        fused = TN.rearrange_siteinds(psi, fused_groups)
        @test length(fused) == 3
        @test TN.to_dense(fused) ≈ before

        # Round trip back to interleaved.
        interleaved_groups = [[s] for s in sitesxy]
        round_trip = TN.rearrange_siteinds(fused, interleaved_groups)
        @test length(round_trip) == 6
        @test TN.to_dense(round_trip) ≈ before
    end

    @testset "makesitediagonal and extractdiagonal roundtrip a tagged site family" begin
        sites = [Index(2; tags=["x", "x=$n"]) for n in 1:2]
        psi = simple_mps(sites)
        dense_original = dense_tensor(psi, sites)

        mpo = TN.makesitediagonal(psi, "x")
        diag_sites = prime.(sites)
        dense_diagonal = dense_tensor(
            mpo,
            [diag_sites[1], sites[1], diag_sites[2], sites[2]],
        )

        for i1 in 1:2, j1 in 1:2, i2 in 1:2, j2 in 1:2
            if i1 == j1 && i2 == j2
                @test dense_diagonal[i1, j1, i2, j2] == dense_original[j1, j2]
            else
                @test iszero(dense_diagonal[i1, j1, i2, j2])
            end
        end

        @test TN.findallsiteinds_by_tag(mpo; tag="x") == sites

        recovered = TN.extractdiagonal(mpo, "x")
        @test siteinds_by_tensor(recovered) == [[site] for site in sites]
        @test dense_tensor(recovered, sites) ≈ dense_original
    end

    @testset "operator space setters require explicit Index vectors" begin
        input_template = [Index(2; tags=["xin", "xin=$n"]) for n in 1:2]
        output_template = [Index(2; tags=["xout", "xout=$n"]) for n in 1:2]
        input_bound = [Index(3; tags=["u", "u=$n"]) for n in 1:2]
        output_bound = [Index(3; tags=["v", "v=$n"]) for n in 1:2]

        op = TN.LinearOperator(; input_indices=input_template, output_indices=output_template)

        @test TN.set_input_space!(op, input_bound) === op
        @test op.true_input == input_bound
        @test all(isnothing, op.true_output)

        @test TN.set_iospaces!(op, input_bound, output_bound) === op
        @test op.true_input == input_bound
        @test op.true_output == output_bound

        @test_throws ArgumentError TN.set_output_space!(op, output_bound[1:1])
        @test_throws MethodError TN.set_input_space!(op, simple_mps(input_template))
        @test_throws MethodError TN.set_output_space!(op, simple_mps(output_template))
        @test_throws MethodError TN.set_iospaces!(op, simple_mps(input_template))
    end
end
