using LinearAlgebra: Diagonal, svd

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
    return Array(data), [ainds[a_rest_axes]..., binds[b_rest_axes]...]
end

function _dense_tensor(tt::TensorTrain)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must be non-empty"))

    data = copy(tt[1].data)
    current_inds = inds(tt[1])
    for n in 2:length(tt)
        data, current_inds = _dense_contract(data, current_inds, tt[n].data, inds(tt[n]))
    end

    boundary_axes = Int[]
    dense_inds = Index[]
    for (axis, index) in pairs(current_inds)
        if hastag(index, _LINK_TAG)
            dim(index) == 1 || throw(
                ArgumentError("Dense reconstruction left an uncontracted nontrivial link index $index"),
            )
            push!(boundary_axes, axis)
        else
            push!(dense_inds, index)
        end
    end

    if !isempty(boundary_axes)
        data = Array(dropdims(data; dims=Tuple(boundary_axes)))
    end
    return data, dense_inds
end

function _flatten_site_groups(site_groups::AbstractVector{<:AbstractVector{<:Index}})
    flattened = Index[]
    for group in site_groups
        isempty(group) && throw(ArgumentError("Each target site group must contain at least one index"))
        append!(flattened, group)
    end
    return flattened
end

function _validate_rearranged_sites(
    tt::TensorTrain,
    site_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    current_sites = reduce(vcat, _siteinds_by_tensor(tt); init=Index[])
    target_sites = _flatten_site_groups(site_groups)

    length(target_sites) == length(current_sites) || throw(
        ArgumentError(
            "Target site groups contain $(length(target_sites)) site indices, expected $(length(current_sites))",
        ),
    )
    length(Set(target_sites)) == length(target_sites) || throw(
        ArgumentError("Target site groups must not contain duplicate site indices"),
    )
    Set(target_sites) == Set(current_sites) || throw(
        ArgumentError("Target site groups must match the TensorTrain site index set"),
    )
    return current_sites, target_sites
end

function _factor_dense_tensor(
    data::AbstractArray,
    site_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    length(site_groups) == 1 && return Tensor[
        Tensor(
            Array(reshape(data, 1, map(dim, only(site_groups))..., 1)),
            [Index(1; tags=[_LINK_TAG, "l=0"]), only(site_groups)..., Index(1; tags=[_LINK_TAG, "l=1"])],
        ),
    ]

    tensors = Tensor[]
    rest = Array(reshape(copy(data), 1, size(data)...))
    left_link = Index(1; tags=[_LINK_TAG, "l=0"])
    left_dim = 1

    for group_position in 1:(length(site_groups) - 1)
        group = site_groups[group_position]
        group_dims = map(dim, group)
        rest_matrix = reshape(rest, left_dim * _prod_dims(group_dims), :)
        factorization = svd(rest_matrix)
        bond_dim = length(factorization.S)
        right_link = Index(bond_dim; tags=[_LINK_TAG, "l=$group_position"])

        tensor_data = Array(reshape(factorization.U[:, 1:bond_dim], left_dim, group_dims..., bond_dim))
        push!(tensors, Tensor(tensor_data, [left_link, group..., right_link]))

        remainder = Diagonal(factorization.S[1:bond_dim]) * factorization.Vt[1:bond_dim, :]
        trailing_shape = size(rest)[(length(group_dims) + 2):end]
        rest = Array(reshape(remainder, bond_dim, trailing_shape...))

        left_link = right_link
        left_dim = bond_dim
    end

    final_group = site_groups[end]
    final_data = Array(reshape(rest, left_dim, map(dim, final_group)..., 1))
    right_boundary = Index(1; tags=[_LINK_TAG, "l=$(length(site_groups))"])
    push!(tensors, Tensor(final_data, [left_link, final_group..., right_boundary]))
    return tensors
end

function _tensor_axis(indices::AbstractVector{<:Index}, target::Index)
    axis = findfirst(==(target), indices)
    axis === nothing && throw(ArgumentError("Index $target not found in tensor"))
    return axis
end

function _diagonalize_tensor_site(tensor::Tensor, site::Index)
    tensor_indices = inds(tensor)
    site_axis = _tensor_axis(tensor_indices, site)
    diagsite = prime(site)
    diagsite in tensor_indices && throw(
        ArgumentError("Tensor already contains diagonal partner $diagsite for site $site"),
    )

    new_dims = (
        size(tensor.data)[1:site_axis-1]...,
        dim(site),
        dim(site),
        size(tensor.data)[site_axis+1:end]...,
    )
    diagonalized = zeros(eltype(tensor.data), new_dims...)

    for position in CartesianIndices(tensor.data)
        index_tuple = Tuple(position)
        diagonal_position = (
            index_tuple[1:site_axis-1]...,
            index_tuple[site_axis],
            index_tuple[site_axis],
            index_tuple[site_axis+1:end]...,
        )
        diagonalized[diagonal_position...] = tensor.data[position]
    end

    return Tensor(
        diagonalized,
        [tensor_indices[1:site_axis-1]..., diagsite, site, tensor_indices[site_axis+1:end]...];
        backend_handle=tensor.backend_handle,
    )
end

function _extract_diagonal_tensor(tensor::Tensor, site::Index, diagsite::Index=prime(site))
    tensor_indices = inds(tensor)
    diag_axis = _tensor_axis(tensor_indices, diagsite)
    site_axis = _tensor_axis(tensor_indices, site)
    dim(site) == dim(diagsite) || throw(
        DimensionMismatch("Diagonal site pair $diagsite and $site must have matching dimensions"),
    )
    diag_axis == site_axis && throw(ArgumentError("Diagonal partner must differ from the base site index"))

    remaining_axes = [axis for axis in eachindex(tensor_indices) if axis != diag_axis]
    remaining_indices = tensor_indices[remaining_axes]
    extracted = zeros(eltype(tensor.data), map(dim, remaining_indices)...)

    for position in CartesianIndices(extracted)
        old_position = Vector{Int}(undef, ndims(tensor.data))
        cursor = 1
        site_value = 0
        for axis in remaining_axes
            old_position[axis] = position[cursor]
            if axis == site_axis
                site_value = position[cursor]
            end
            cursor += 1
        end
        old_position[diag_axis] = site_value
        extracted[position] = tensor.data[Tuple(old_position)...]
    end

    return Tensor(extracted, remaining_indices; backend_handle=tensor.backend_handle)
end

"""
    rearrange_siteinds(tt, site_groups)

Return a new `TensorTrain` whose tensors carry the requested `site_groups`
while preserving the represented dense indexed tensor.
"""
function rearrange_siteinds(
    tt::TensorTrain,
    site_groups::AbstractVector{<:AbstractVector{<:Index}},
)
    _, target_sites = _validate_rearranged_sites(tt, site_groups)
    dense_data, current_sites = _dense_tensor(tt)
    permutation = [findfirst(==(site), current_sites) for site in target_sites]
    permuted = permutedims(dense_data, Tuple(permutation))
    tensors = _factor_dense_tensor(permuted, site_groups)
    return TensorTrain(tensors, tt.llim, tt.llim + length(tensors) + 1)
end

"""
    makesitediagonal(tt, tag)

Return a copy of `tt` where every numbered site family matching `tag` gains a
paired `prime(site)` leg and is diagonalized on that pair.
"""
function makesitediagonal(tt::TensorTrain, tag::AbstractString)
    copied = _copy_train(tt)
    for site in findallsiteinds_by_tag(copied; tag=tag)
        position = findsite(copied, site)
        position === nothing && throw(ArgumentError("Site index $site not found in TensorTrain"))
        copied.data[position] = _diagonalize_tensor_site(copied[position], site)
    end
    return copied
end

"""
    extractdiagonal(tt, tag)

Return a copy of `tt` with each numbered site family matching `tag` collapsed
from a `(prime(site), site)` diagonal pair back to `site`.
"""
function extractdiagonal(tt::TensorTrain, tag::AbstractString)
    copied = _copy_train(tt)
    for site in findallsiteinds_by_tag(copied; tag=tag)
        position = findsite(copied, site)
        position === nothing && throw(ArgumentError("Site index $site not found in TensorTrain"))
        copied.data[position] = _extract_diagonal_tensor(copied[position], site)
    end
    return copied
end
