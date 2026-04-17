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
    rearrange_siteinds(tt, site_groups; kwargs...)

Regroup the site indices of `tt` so that each new node holds the indices in
`site_groups[i]`. Equivalent to [`restructure_to`](@ref) with the same
target groups; this name is kept for compatibility with the
`Quantics.jl` / `ITensorMPS.jl` API surface.

`kwargs...` are forwarded verbatim to `restructure_to` and let the caller
control split/swap truncation knobs and an optional final truncation pass.
"""
function rearrange_siteinds(
    tt::TensorTrain,
    site_groups::AbstractVector{<:AbstractVector{<:Index}};
    kwargs...,
)
    return restructure_to(tt, site_groups; kwargs...)
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
