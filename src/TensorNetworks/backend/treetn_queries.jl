"""
    dag(tt)

Return a TensorTrain with each site tensor complex-conjugated.
"""
function dag(tt::TensorTrain)
    return TensorTrain([dag(t) for t in tt.data], tt.llim, tt.rlim)
end

"""
    linkinds(tt)
    linkinds(tt, i)

Return the shared bond indices between adjacent tensors in `tt`.
"""
function linkinds(tt::TensorTrain)
    isempty(tt.data) && return Index[]

    links = Index[]
    for i in 1:(length(tt) - 1)
        push!(links, linkinds(tt, i))
    end
    return links
end

function linkinds(tt::TensorTrain, i::Integer)
    1 <= i < length(tt) || throw(BoundsError(tt.data, i))

    shared = commoninds(inds(tt[i]), inds(tt[i + 1]))
    length(shared) == 1 || throw(
        ArgumentError("Expected exactly 1 shared index between sites $i and $(i + 1), got $(length(shared))"),
    )
    return only(shared)
end

"""
    linkdims(tt)

Return the dimensions of the shared bond indices in `tt`.
"""
linkdims(tt::TensorTrain) = [dim(index) for index in linkinds(tt)]

"""
    siteinds(tt)
    siteinds(tt, i)

Return the non-bond indices attached to each tensor in `tt`.
"""
function siteinds(tt::TensorTrain)
    isempty(tt.data) && return Vector{Index}[]

    bond_set = Set{Index}()
    for i in 1:(length(tt) - 1)
        for index in commoninds(inds(tt[i]), inds(tt[i + 1]))
            push!(bond_set, index)
        end
    end

    return [filter(index -> !(index in bond_set), inds(tt[i])) for i in 1:length(tt)]
end

function siteinds(tt::TensorTrain, i::Integer)
    1 <= i <= length(tt) || throw(BoundsError(tt.data, i))

    bond_inds = Set{Index}()
    if i > 1
        for index in commoninds(inds(tt[i]), inds(tt[i - 1]))
            push!(bond_inds, index)
        end
    end
    if i < length(tt)
        for index in commoninds(inds(tt[i]), inds(tt[i + 1]))
            push!(bond_inds, index)
        end
    end

    return filter(index -> !(index in bond_inds), inds(tt[i]))
end

function _extract_mpo_io_indices(tt::TensorTrain)
    input_indices = Index[]
    output_indices = Index[]
    for site_inds in siteinds(tt)
        length(site_inds) == 2 || throw(
            ArgumentError("Expected 2 site indices per MPO tensor, got $(length(site_inds))"),
        )
        push!(output_indices, site_inds[1])
        push!(input_indices, site_inds[2])
    end
    return input_indices, output_indices
end
