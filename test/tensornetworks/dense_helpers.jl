if !isdefined(@__MODULE__, :_tn_test_prod_dims)
    _tn_test_prod_dims(xs) = isempty(xs) ? 1 : prod(xs)
end

if !isdefined(@__MODULE__, :_tn_test_dense_contract)
    function _tn_test_dense_contract(
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
            _tn_test_prod_dims(size(a)[a_rest_axes]),
            _tn_test_prod_dims(size(a)[a_common_axes]),
        )
        bmat = reshape(
            permutedims(b, (b_common_axes..., b_rest_axes...)),
            _tn_test_prod_dims(size(b)[b_common_axes]),
            _tn_test_prod_dims(size(b)[b_rest_axes]),
        )

        data = reshape(
            amat * bmat,
            size(a)[a_rest_axes]...,
            size(b)[b_rest_axes]...,
        )
        return data, [ainds[a_rest_axes]..., binds[b_rest_axes]...]
    end
end

if !isdefined(@__MODULE__, :tn_test_dense_tensor)
    function tn_test_dense_tensor(tt::TN.TensorTrain, target_inds::Vector{Index})
        data = copy_data(tt[1])
        current_inds = inds(tt[1])
        for n in 2:length(tt)
            data, current_inds = _tn_test_dense_contract(data, current_inds, copy_data(tt[n]), inds(tt[n]))
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
end
