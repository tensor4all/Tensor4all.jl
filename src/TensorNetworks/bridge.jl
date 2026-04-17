"""
    SimpleTT.TensorTrain(tt::TensorNetworks.TensorTrain) -> SimpleTT.TensorTrain{T,N}

Convert an indexed `TensorNetworks.TensorTrain` to its raw-array
counterpart `Tensor4all.SimpleTT.TensorTrain{T,N}`.

The number of site indices `s` must be the **same on every tensor** of
`tt` (so 1 for MPS-like inputs, 2 for MPO-like inputs, ...). The
resulting array rank is `N = 2 + s`: each site tensor is permuted into
`(left_link, site_inds..., right_link)` order. Boundary tensors that do
not carry an explicit left or right link in `tt` are padded with a
dim-1 link so that every site tensor has the same rank.

Site index identity is **not** preserved (the SimpleTT layer is
indexless). Use `TensorNetworks.TensorTrain(stt, site_groups)` to lift
back into the indexed layer with caller-supplied indices.

Throws `ArgumentError` if `tt` is empty or if the number of site indices
varies across tensors.
"""
function _Tensor4all_SimpleTT_TensorTrain_from_TN(tt::TensorTrain)
    isempty(tt.data) && throw(ArgumentError("TensorTrain must not be empty for SimpleTT conversion"))

    site_groups = _siteinds_by_tensor(tt)
    n_site = length(first(site_groups))
    all(length(group) == n_site for group in site_groups) || throw(ArgumentError(
        "all tensors must carry the same number of site indices; got $(map(length, site_groups))",
    ))

    n_tensors = length(tt)
    N = n_site + 2
    raw = Vector{Array}(undef, n_tensors)

    for i in 1:n_tensors
        tensor = tt[i]
        site_inds_here = site_groups[i]
        all_inds = inds(tensor)
        link_inds = filter(idx -> !(idx in site_inds_here), all_inds)
        # Identify left vs right link by membership in adjacent tensors.
        left_link = i > 1 ? _shared_index(all_inds, inds(tt[i - 1])) : nothing
        right_link = i < n_tensors ? _shared_index(all_inds, inds(tt[i + 1])) : nothing

        link_inds_seen = filter(!isnothing, [left_link, right_link])
        all(idx in link_inds for idx in link_inds_seen) || throw(ArgumentError(
            "tensor $i carries link indices that are not detectable as bonds; layout may not be a chain",
        ))

        target_order = Index[]
        left_link === nothing || push!(target_order, left_link)
        append!(target_order, site_inds_here)
        right_link === nothing || push!(target_order, right_link)

        if length(target_order) != length(all_inds)
            throw(ArgumentError(
                "tensor $i has $(length(all_inds)) indices but the canonical (link, sites..., link) layout used $(length(target_order)); other indices are unexpected",
            ))
        end

        permuted = Array(tensor, target_order...)

        # Pad boundary dim-1 links so every tensor has rank N.
        if left_link === nothing
            permuted = reshape(permuted, 1, size(permuted)...)
        end
        if right_link === nothing
            permuted = reshape(permuted, size(permuted)..., 1)
        end
        raw[i] = permuted
    end

    elem = promote_type(map(eltype, raw)...)
    typed = [convert(Array{elem,N}, t) for t in raw]
    return SimpleTT.TensorTrain{elem,N}(typed)
end

"""
    TensorNetworks.TensorTrain(stt::SimpleTT.TensorTrain{T,N},
                               site_groups::AbstractVector{<:AbstractVector{<:Index}}) -> TensorTrain
    TensorNetworks.TensorTrain(stt::SimpleTT.TensorTrain{T,3},
                               sites::AbstractVector{<:Index}) -> TensorTrain
    TensorNetworks.TensorTrain(stt::SimpleTT.TensorTrain{T,4},
                               input_inds::AbstractVector{<:Index},
                               output_inds::AbstractVector{<:Index}) -> TensorTrain

Lift a raw-array `Tensor4all.SimpleTT.TensorTrain{T,N}` back into the
indexed `TensorNetworks.TensorTrain` layer. The caller supplies the
site `Index` identities; each `site_groups[i]` lists the site indices
for tensor `i` and must have length `N - 2`.

The N=3 and N=4 overloads are convenience adapters: they form
`site_groups = [[s] for s in sites]` and
`site_groups = [[in_inds[i], out_inds[i]] for ...]` respectively.

Internal link indices are freshly minted as
`Index(bond_dim; tags=["Link", "l=i"])` for each chain bond. Boundary
dim-1 links present in `stt` (which `SimpleTT` requires for rank
uniformity) are dropped from the indexed result.

Throws `ArgumentError` if shapes mismatch the supplied `site_groups`.
"""
function TensorTrain(
    stt::SimpleTT.TensorTrain{T,N},
    site_groups::AbstractVector{<:AbstractVector{<:Index}},
) where {T,N}
    n_tensors = length(stt.sitetensors)
    n_tensors == length(site_groups) || throw(DimensionMismatch(
        "site_groups must have $n_tensors entries (one per tensor), got $(length(site_groups))",
    ))
    n_site = N - 2
    all(length(g) == n_site for g in site_groups) || throw(ArgumentError(
        "each site_groups entry must have length N - 2 = $n_site for SimpleTT.TensorTrain{T,$N}",
    ))

    n_tensors == 0 && return TensorTrain(Tensor[])

    # Materialize new chain link indices using the actual bond dimensions.
    link_indices = Vector{Index}(undef, max(n_tensors - 1, 0))
    for i in 1:(n_tensors - 1)
        bond_dim = size(stt.sitetensors[i], N)
        bond_dim == size(stt.sitetensors[i + 1], 1) || throw(DimensionMismatch(
            "bond dimension mismatch between tensor $i (right link $bond_dim) and tensor $(i + 1) (left link $(size(stt.sitetensors[i + 1], 1)))",
        ))
        link_indices[i] = Index(bond_dim; tags=[_LINK_TAG, "l=$i"])
    end

    tensors = Tensor[]
    for i in 1:n_tensors
        raw = stt.sitetensors[i]
        site_inds_here = site_groups[i]

        size_left = size(raw, 1)
        size_right = size(raw, N)
        for (axis, idx) in enumerate(site_inds_here)
            size(raw, axis + 1) == dim(idx) || throw(DimensionMismatch(
                "tensor $i axis $(axis + 1) has size $(size(raw, axis + 1)) but site index dim is $(dim(idx))",
            ))
        end

        # Decide whether to keep boundary links: drop dim-1 boundaries.
        has_left = i > 1 || size_left != 1
        has_right = i < n_tensors || size_right != 1
        target_inds = Index[]
        has_left && push!(target_inds, i > 1 ? link_indices[i - 1] : Index(size_left; tags=[_LINK_TAG, "l=0"]))
        append!(target_inds, site_inds_here)
        has_right && push!(target_inds, i < n_tensors ? link_indices[i] : Index(size_right; tags=[_LINK_TAG, "l=$n_tensors"]))

        data = if has_left && has_right
            raw
        elseif !has_left && has_right
            reshape(raw, size(raw)[2:end])
        elseif has_left && !has_right
            reshape(raw, size(raw)[1:end - 1])
        else
            reshape(raw, size(raw)[2:end - 1])
        end

        push!(tensors, Tensor(collect(data), target_inds))
    end

    return TensorTrain(tensors)
end

# Convenience: MPS-like (1 site per tensor)
function TensorTrain(
    stt::SimpleTT.TensorTrain{T,3},
    sites::AbstractVector{<:Index},
) where {T}
    return TensorTrain(stt, [[s] for s in sites])
end

# Convenience: MPO-like (2 sites per tensor; (input, output) at each site)
function TensorTrain(
    stt::SimpleTT.TensorTrain{T,4},
    input_inds::AbstractVector{<:Index},
    output_inds::AbstractVector{<:Index},
) where {T}
    length(input_inds) == length(output_inds) || throw(DimensionMismatch(
        "input_inds and output_inds must have the same length, got $(length(input_inds)) and $(length(output_inds))",
    ))
    return TensorTrain(stt, [[input_inds[i], output_inds[i]] for i in 1:length(input_inds)])
end

# Helper: identify the unique Index shared between two index lists, or nothing.
function _shared_index(a::AbstractVector{<:Index}, b::AbstractVector{<:Index})
    shared = Index[]
    for idx in a
        if any(==(idx), b)
            push!(shared, idx)
        end
    end
    if length(shared) == 0
        return nothing
    elseif length(shared) == 1
        return only(shared)
    else
        throw(ArgumentError(
            "expected at most 1 shared index between adjacent tensors, got $(length(shared))",
        ))
    end
end

# Register the SimpleTT-side method *after* the helper above is defined.
# (SimpleTT module is loaded earlier than TensorNetworks, so we extend its
# generic SimpleTT.TensorTrain by adding a method on the TN.TensorTrain type.)
SimpleTT.TensorTrain(tt::TensorTrain) = _Tensor4all_SimpleTT_TensorTrain_from_TN(tt)
