function _require_unfuse_base(base::Integer)
    base > 1 || throw(ArgumentError("base must be greater than 1, got $base"))
    return Int(base)
end

function _infer_unfuse_nvariables(
    op::TensorNetworks.LinearOperator,
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index},
    base::Int,
)
    mpo = op.mpo
    mpo === nothing && throw(ArgumentError("LinearOperator.mpo must be set before unfuse_quantics_operator"))
    length(op.input_indices) == length(op.output_indices) == length(mpo) || throw(
        ArgumentError(
            "unfuse_quantics_operator requires synchronized operator metadata; got mpo=$(length(mpo)), input=$(length(op.input_indices)), output=$(length(op.output_indices))",
        ),
    )
    length(input_sites) == length(output_sites) || throw(
        DimensionMismatch(
            "input_sites and output_sites must have the same length, got $(length(input_sites)) and $(length(output_sites))",
        ),
    )
    length(input_sites) % length(op.input_indices) == 0 || throw(
        DimensionMismatch(
            "target site count $(length(input_sites)) must be a multiple of fused operator length $(length(op.input_indices))",
        ),
    )

    nvariables = div(length(input_sites), length(op.input_indices))
    nvariables > 0 || throw(ArgumentError("unfuse_quantics_operator requires at least one target variable"))
    expected_fused_dim = base^nvariables

    for (name, sites) in (("input_sites", input_sites), ("output_sites", output_sites))
        for (position, index) in pairs(sites)
            dim(index) == base || throw(
                DimensionMismatch("$name[$position] has dimension $(dim(index)); expected base $base"),
            )
        end
    end

    for (position, index) in pairs(op.input_indices)
        dim(index) == expected_fused_dim || throw(
            DimensionMismatch(
                "fused input index $position has dimension $(dim(index)); expected base^nvariables = $expected_fused_dim",
            ),
        )
    end
    for (position, index) in pairs(op.output_indices)
        dim(index) == expected_fused_dim || throw(
            DimensionMismatch(
                "fused output index $position has dimension $(dim(index)); expected base^nvariables = $expected_fused_dim",
            ),
        )
    end

    return nvariables
end

function _unfuse_tensor_indices(tensor::Tensor, replacements::Dict{UInt64, Vector{Index}})
    new_indices = Index[]
    new_dims = Int[]
    for index in inds(tensor)
        replacement = get(replacements, id(index), nothing)
        if replacement === nothing
            push!(new_indices, index)
            push!(new_dims, dim(index))
        else
            prod(dim.(replacement)) == dim(index) || throw(
                DimensionMismatch(
                    "replacement dimensions $(dim.(replacement)) do not multiply to fused index dimension $(dim(index))",
                ),
            )
            append!(new_indices, replacement)
            append!(new_dims, dim.(replacement))
        end
    end
    return Tensor(reshape(tensor.data, Tuple(new_dims)), new_indices)
end

function _restore_requested_site_indices!(
    tt::TensorNetworks.TensorTrain,
    requested::AbstractVector{<:Index},
)
    actual_by_id = Dict{UInt64, Index}()
    for group in TensorNetworks.siteinds(tt), index in group
        actual_by_id[id(index)] = index
    end

    oldsites = Index[]
    newsites = Index[]
    for index in requested
        actual = get(actual_by_id, id(index), nothing)
        actual === nothing && throw(
            ArgumentError("restructured operator is missing requested site index $index"),
        )
        if actual != index
            push!(oldsites, actual)
            push!(newsites, index)
        end
    end
    isempty(oldsites) || TensorNetworks.replace_siteinds!(tt, oldsites, newsites)
    return tt
end

"""
    unfuse_quantics_operator(op, input_sites, output_sites; base=2, ...)

Convert a fused-QTT `LinearOperator` into a per-bit/per-variable operator.

`input_sites` and `output_sites` are flat site-major vectors: for each bit
layer, list variables in the desired variable order. A fused physical index
is decoded with Julia column-major order, so the first variable in each bit
layer is the least-significant digit of that fused index.

Truncation keywords are forwarded to `TensorNetworks.restructure_to`.
"""
function unfuse_quantics_operator(
    op::TensorNetworks.LinearOperator,
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index};
    base::Integer=2,
    split_threshold::Real=0.0,
    split_maxdim::Integer=0,
    split_svd_policy::Union{Nothing, TensorNetworks.SvdTruncationPolicy}=nothing,
    split_final_sweep::Bool=false,
    final_threshold::Real=0.0,
    final_maxdim::Integer=0,
    final_svd_policy::Union{Nothing, TensorNetworks.SvdTruncationPolicy}=nothing,
)
    resolved_base = _require_unfuse_base(base)
    nvariables = _infer_unfuse_nvariables(op, input_sites, output_sites, resolved_base)

    expanded_tensors = Tensor[]
    for position in eachindex(op.mpo.data)
        offset = (position - 1) * nvariables
        replacements = Dict{UInt64, Vector{Index}}(
            id(op.output_indices[position]) => collect(output_sites[(offset + 1):(offset + nvariables)]),
            id(op.input_indices[position]) => collect(input_sites[(offset + 1):(offset + nvariables)]),
        )
        push!(expanded_tensors, _unfuse_tensor_indices(op.mpo[position], replacements))
    end

    expanded = TensorNetworks.TensorTrain(expanded_tensors, op.mpo.llim, op.mpo.rlim)
    target_groups = [
        Index[output_sites[position], input_sites[position]]
        for position in eachindex(input_sites)
    ]
    mpo = TensorNetworks.restructure_to(
        expanded,
        target_groups;
        split_threshold,
        split_maxdim,
        split_svd_policy,
        split_final_sweep,
        final_threshold,
        final_maxdim,
        final_svd_policy,
    )
    _restore_requested_site_indices!(mpo, [collect(output_sites); collect(input_sites)])

    return TensorNetworks.LinearOperator(;
        mpo,
        input_indices=collect(input_sites),
        output_indices=collect(output_sites),
    )
end
