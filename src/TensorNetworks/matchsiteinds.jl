function _site_arity(tt::TensorTrain)
    siteinds = _siteinds_by_tensor(tt)
    isempty(siteinds) && throw(ArgumentError("matchsiteinds requires a non-empty TensorTrain"))
    arities = unique(length.(siteinds))
    length(arities) == 1 || throw(
        ArgumentError("matchsiteinds expects a consistent MPS-like or MPO-like TensorTrain"),
    )
    arity = only(arities)
    arity in (1, 2) || throw(ArgumentError("matchsiteinds supports only MPS-like or MPO-like TensorTrains"))
    return arity, siteinds
end

function _validate_target_positions(current_sites::Vector{Index}, target_sites::AbstractVector{<:Index})
    positions = Int[]
    seen = Set{Int}()
    for site in current_sites
        position = findfirst(==(site), target_sites)
        position === nothing && throw(
            ArgumentError("Target site set does not contain site index $site"),
        )
        position in seen && throw(
            ArgumentError("Target site set maps multiple sparse sites onto position $position"),
        )
        push!(positions, position)
        push!(seen, position)
    end
    issorted(positions) || throw(
        ArgumentError("Sparse site indices must appear in ascending order in the target site set"),
    )
    return positions
end

function _resolve_mpo_sites(
    siteinds_by_tensor::Vector{Vector{Index}},
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index},
)
    length(input_sites) == length(output_sites) || throw(
        DimensionMismatch(
            "Input and output target site sets must have the same length, got $(length(input_sites)) and $(length(output_sites))",
        ),
    )

    sparse_input = Index[]
    sparse_output = Index[]
    positions = Int[]
    seen = Set{Int}()

    for current_sites in siteinds_by_tensor
        length(current_sites) == 2 || throw(
            ArgumentError("Each MPO-like tensor must have exactly two site-like indices"),
        )
        first_site, second_site = current_sites
        first_input = findfirst(==(first_site), input_sites)
        first_output = findfirst(==(first_site), output_sites)
        second_input = findfirst(==(second_site), input_sites)
        second_output = findfirst(==(second_site), output_sites)

        if first_input !== nothing && second_output !== nothing
            first_input == second_output || throw(
                ArgumentError("Input/output target site positions must agree for each MPO-like tensor"),
            )
            push!(sparse_input, first_site)
            push!(sparse_output, second_site)
            push!(positions, first_input)
        elseif second_input !== nothing && first_output !== nothing
            second_input == first_output || throw(
                ArgumentError("Input/output target site positions must agree for each MPO-like tensor"),
            )
            push!(sparse_input, second_site)
            push!(sparse_output, first_site)
            push!(positions, second_input)
        else
            throw(
                ArgumentError(
                    "Each MPO-like tensor must contribute one input and one output site found in the target site sets",
                ),
            )
        end
        last(positions) in seen && throw(
            ArgumentError("Target site set maps multiple sparse operators onto position $(last(positions))"),
        )
        push!(seen, last(positions))
    end

    issorted(positions) || throw(
        ArgumentError("Sparse operator sites must appear in ascending order in the target site set"),
    )
    return positions, sparse_input, sparse_output
end

function _canonicalize_mps_tensor(tensor::Tensor, sparse_position::Int, sparse_length::Int)
    data = tensor.data
    if ndims(data) == 1
        sparse_length == 1 || throw(
            ArgumentError("Only single-site MPS-like TensorTrains may use rank-1 tensors"),
        )
        return Array(reshape(copy(data), 1, size(data, 1), 1))
    elseif ndims(data) == 2
        if sparse_position == 1
            return Array(reshape(copy(data), 1, size(data, 1), size(data, 2)))
        elseif sparse_position == sparse_length
            return Array(reshape(copy(data), size(data, 1), size(data, 2), 1))
        end
    elseif ndims(data) == 3
        return copy(data)
    end
    throw(ArgumentError("Unsupported MPS-like tensor shape $(size(data)) at position $sparse_position"))
end

function _canonicalize_mpo_tensor(tensor::Tensor, sparse_position::Int, sparse_length::Int)
    data = tensor.data
    if ndims(data) == 2
        sparse_length == 1 || throw(
            ArgumentError("Only single-site MPO-like TensorTrains may use rank-2 tensors"),
        )
        return Array(reshape(copy(data), 1, size(data, 1), size(data, 2), 1))
    elseif ndims(data) == 3
        if sparse_position == 1
            return Array(reshape(copy(data), 1, size(data, 1), size(data, 2), size(data, 3)))
        elseif sparse_position == sparse_length
            return Array(reshape(copy(data), size(data, 1), size(data, 2), size(data, 3), 1))
        end
    elseif ndims(data) == 4
        return copy(data)
    end
    throw(ArgumentError("Unsupported MPO-like tensor shape $(size(data)) at position $sparse_position"))
end

function _set_linkdim!(linkdims::Vector{Int}, position::Int, value::Int)
    current = linkdims[position]
    if current == 0
        linkdims[position] = value
    elseif current != value
        throw(ArgumentError("Inconsistent link dimensions encountered while embedding sparse sites"))
    end
    return nothing
end

function _fill_linkdims!(linkdims::Vector{Int})
    while any(==(0), linkdims)
        changed = false
        for position in eachindex(linkdims)
            linkdims[position] != 0 && continue
            if position > 1 && linkdims[position - 1] != 0
                linkdims[position] = linkdims[position - 1]
                changed = true
            elseif position < length(linkdims) && linkdims[position + 1] != 0
                linkdims[position] = linkdims[position + 1]
                changed = true
            end
        end
        changed || throw(ArgumentError("Failed to infer link dimensions for embedded sparse sites"))
    end
    return linkdims
end

function _link_indices_for_embedding(canonical_tensors::Vector{<:AbstractArray}, positions::Vector{Int}, full_length::Int)
    linkdims = zeros(Int, full_length + 1)
    linkdims[1] = 1
    linkdims[end] = 1

    for (n, position) in pairs(positions)
        tensor = canonical_tensors[n]
        _set_linkdim!(linkdims, position, size(tensor, 1))
        _set_linkdim!(linkdims, position + 1, size(tensor, ndims(tensor)))
    end

    _fill_linkdims!(linkdims)
    return [Index(linkdims[position]; tags=["Link", "l=$(position - 1)"]) for position in eachindex(linkdims)]
end

function _missing_mps_tensor(left::Index, site::Index, right::Index)
    dim(left) == dim(right) || throw(
        DimensionMismatch("Missing MPS-like site insertion requires matching left/right link dimensions"),
    )
    data = zeros(Float64, dim(left), dim(site), dim(right))
    for link in 1:dim(left), site_value in 1:dim(site)
        data[link, site_value, link] = 1.0
    end
    return Tensor(data, [left, site, right])
end

function _missing_mpo_tensor(left::Index, input_site::Index, output_site::Index, right::Index)
    dim(left) == dim(right) || throw(
        DimensionMismatch("Missing MPO-like site insertion requires matching left/right link dimensions"),
    )
    dim(input_site) == dim(output_site) || throw(
        DimensionMismatch("Identity MPO insertion requires matching input/output site dimensions"),
    )
    data = zeros(Float64, dim(left), dim(input_site), dim(output_site), dim(right))
    for link in 1:dim(left), site_value in 1:dim(input_site)
        data[link, site_value, site_value, link] = 1.0
    end
    return Tensor(data, [left, input_site, output_site, right])
end

"""
    matchsiteinds(tt, sites)
    matchsiteinds(tt, input_sites, output_sites)

Embed a sparse MPS-like or MPO-like `TensorTrain` into a larger site set by
inserting constant-one MPS legs or identity MPO legs on missing sites.
"""
function matchsiteinds(tt::TensorTrain, sites::AbstractVector{<:Index})
    arity, siteinds_by_tensor = _site_arity(tt)
    arity == 1 || throw(
        ArgumentError("matchsiteinds(tt, sites) expects an MPS-like TensorTrain with one site-like index per tensor"),
    )

    sparse_sites = [only(indices) for indices in siteinds_by_tensor]
    positions = _validate_target_positions(sparse_sites, sites)
    canonical_tensors = [_canonicalize_mps_tensor(tt[n], n, length(tt)) for n in eachindex(tt.data)]
    links = _link_indices_for_embedding(canonical_tensors, positions, length(sites))
    sparse_by_position = Dict(positions[n] => canonical_tensors[n] for n in eachindex(positions))

    tensors = Tensor[]
    for position in eachindex(sites)
        left = links[position]
        right = links[position + 1]
        if haskey(sparse_by_position, position)
            push!(tensors, Tensor(sparse_by_position[position], [left, sites[position], right]))
        else
            push!(tensors, _missing_mps_tensor(left, sites[position], right))
        end
    end

    return TensorTrain(tensors, tt.llim, tt.llim + length(tensors) + 1)
end

function matchsiteinds(
    tt::TensorTrain,
    input_sites::AbstractVector{<:Index},
    output_sites::AbstractVector{<:Index},
)
    arity, siteinds_by_tensor = _site_arity(tt)
    arity == 2 || throw(
        ArgumentError(
            "matchsiteinds(tt, input_sites, output_sites) expects an MPO-like TensorTrain with two site-like indices per tensor",
        ),
    )

    positions, sparse_input, sparse_output = _resolve_mpo_sites(siteinds_by_tensor, input_sites, output_sites)
    canonical_tensors = [_canonicalize_mpo_tensor(tt[n], n, length(tt)) for n in eachindex(tt.data)]
    links = _link_indices_for_embedding(canonical_tensors, positions, length(input_sites))
    sparse_by_position = Dict(positions[n] => canonical_tensors[n] for n in eachindex(positions))

    tensors = Tensor[]
    for position in eachindex(input_sites)
        left = links[position]
        right = links[position + 1]
        if haskey(sparse_by_position, position)
            push!(tensors, Tensor(sparse_by_position[position], [left, input_sites[position], output_sites[position], right]))
        else
            push!(tensors, _missing_mpo_tensor(left, input_sites[position], output_sites[position], right))
        end
    end

    return TensorTrain(tensors, tt.llim, tt.llim + length(tensors) + 1)
end
