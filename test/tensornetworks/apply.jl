using Test
using Tensor4all

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
    if !isempty(boundary_axes)
        data = dropdims(data; dims=Tuple(boundary_axes))
        current_inds = [index for index in current_inds if !hastag(index, "Link")]
    end

    permutation = map(target_inds) do index
        axis = findfirst(==(index), current_inds)
        axis === nothing && error("Target index $index not found")
        axis
    end
    return permutedims(data, Tuple(permutation))
end

function simple_state(sites::Vector{Index})
    links = [
        Index(n in (1, length(sites) + 1) ? 1 : 2; tags=["Link", "state-l=$(n - 1)"])
        for n in 1:(length(sites) + 1)
    ]
    tensors = Tensor[
        Tensor(reshape([1.0, 2.0, 3.0, 4.0], 2, 2), [sites[1], links[2]]),
        Tensor(reshape([1.0, -1.0, 0.5, 2.0], 2, 2), [links[2], sites[2]]),
    ]
    return TN.TensorTrain(tensors, 0, 3)
end

function identity_operator(input_sites::Vector{Index}, internal_output::Vector{Index})
    link = Index(1; tags=["Link", "op-l=1"])
    t1 = Tensor(
        reshape([1.0, 0.0, 0.0, 1.0], 2, 2, 1),
        [internal_output[1], input_sites[1], link],
    )
    t2 = Tensor(
        reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),
        [link, internal_output[2], input_sites[2]],
    )
    mpo = TN.TensorTrain([t1, t2], 0, 3)
    return TN.LinearOperator(; mpo, input_indices=copy(input_sites), output_indices=copy(internal_output))
end

function single_site_operator(internal_input::Index, internal_output::Index, matrix::Matrix{Float64})
    mpo = TN.TensorTrain([Tensor(matrix, [internal_output, internal_input])], 0, 2)
    return TN.LinearOperator(; mpo, input_indices=[internal_input], output_indices=[internal_output])
end

function apply_matrix_on_last_axis(data::AbstractMatrix, matrix::AbstractMatrix)
    expected = similar(data)
    for row in axes(data, 1), out in axes(matrix, 1)
        value = zero(eltype(expected))
        for input in axes(matrix, 2)
            value += matrix[out, input] * data[row, input]
        end
        expected[row, out] = value
    end
    return expected
end

@testset "TensorNetworks apply" begin
    input_sites = [Index(2; tags=["x", "x=$n"]) for n in 1:2]
    state = simple_state(input_sites)
    output_internal = [Index(2; tags=["tmpout", "tmpout=$n"]) for n in 1:2]
    output_true = [Index(2; tags=["y", "y=$n"]) for n in 1:2]

    op = identity_operator(input_sites, output_internal)
    TN.set_iospaces!(op, input_sites, output_true)

    result = TN.apply(op, state)

    @test TN.findallsiteinds_by_tag(result; tag="y") == output_true
    @test dense_tensor(result, output_true) ≈ dense_tensor(state, input_sites)

    internal_in = Index(2; tags=["ain", "ain=2"])
    internal_out = Index(2; tags=["aout", "aout=2"])
    local_matrix = [0.0 1.0; 2.0 0.0]
    partial = single_site_operator(internal_in, internal_out, local_matrix)
    partial_output = Index(2; tags=["z", "z=2"])
    TN.set_iospaces!(partial, [input_sites[2]], [partial_output])

    partial_result = TN.apply(partial, state)
    expected_partial = apply_matrix_on_last_axis(dense_tensor(state, input_sites), local_matrix)

    @test dense_tensor(partial_result, [input_sites[1], partial_output]) ≈ expected_partial

    pol = TN.SvdTruncationPolicy()
    result_pol = TN.apply(op, state; threshold=1e-12, svd_policy=pol)
    @test dense_tensor(result_pol, output_true) ≈ dense_tensor(state, input_sites)
end
