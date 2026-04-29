"""
    PartialContractionSpec(contract_pairs, diagonal_pairs; output_order=nothing)

Describe a partial contraction between two `TensorTrain`s.

`contract_pairs` contains `left => right` site-index pairs that are summed
over and removed from the result. `diagonal_pairs` contains `left => right`
site-index pairs that are identified by diagonal/copy structure while keeping
the left-hand index in the result. `output_order`, when provided, lists the
surviving external site indices in the requested result order.
"""
struct PartialContractionSpec
    contract_pairs::Vector{Pair{Index,Index}}
    diagonal_pairs::Vector{Pair{Index,Index}}
    output_order::Union{Nothing,Vector{Index}}
end

function PartialContractionSpec(
    contract_pairs,
    diagonal_pairs;
    output_order::Union{Nothing,AbstractVector{<:Index}}=nothing,
)
    contract = _index_pair_vector(contract_pairs, "contract_pairs")
    diagonal = _index_pair_vector(diagonal_pairs, "diagonal_pairs")
    _validate_partial_pair_dimensions(contract, "contract_pairs")
    _validate_partial_pair_dimensions(diagonal, "diagonal_pairs")
    _validate_partial_pair_uniqueness(contract, diagonal)

    order = output_order === nothing ? nothing : Index[index for index in output_order]
    if order !== nothing && length(unique(order)) != length(order)
        throw(ArgumentError("output_order must not contain duplicates, got $order"))
    end

    return PartialContractionSpec(contract, diagonal, order)
end

function _index_pair_vector(pairs, label::AbstractString)
    result = Pair{Index,Index}[]
    for pair in pairs
        pair isa Pair || throw(ArgumentError("$label entries must be Pair{Index,Index}, got $(typeof(pair))"))
        left = first(pair)
        right = last(pair)
        left isa Index || throw(ArgumentError("$label left entry must be an Index, got $(typeof(left))"))
        right isa Index || throw(ArgumentError("$label right entry must be an Index, got $(typeof(right))"))
        push!(result, left => right)
    end
    return result
end

function _validate_partial_pair_dimensions(pairs::Vector{Pair{Index,Index}}, label::AbstractString)
    for pair in pairs
        left, right = first(pair), last(pair)
        dim(left) == dim(right) || throw(DimensionMismatch(
            "$label pair dimension mismatch: left $left has dim $(dim(left)), right $right has dim $(dim(right))",
        ))
    end
    return nothing
end

function _validate_partial_pair_uniqueness(
    contract_pairs::Vector{Pair{Index,Index}},
    diagonal_pairs::Vector{Pair{Index,Index}},
)
    seen_left = Set{Index}()
    seen_right = Set{Index}()
    for pair in Iterators.flatten((contract_pairs, diagonal_pairs))
        left, right = first(pair), last(pair)
        left in seen_left && throw(ArgumentError("left index $left appears in more than one partial contraction pair"))
        right in seen_right && throw(ArgumentError("right index $right appears in more than one partial contraction pair"))
        push!(seen_left, left)
        push!(seen_right, right)
    end
    return nothing
end

function _flatten_siteinds(tt::TensorTrain)
    return Index[index for group in siteinds(tt) for index in group]
end

function _validate_partial_contract_inputs(a::TensorTrain, b::TensorTrain, spec::PartialContractionSpec)
    isempty(a.data) && throw(ArgumentError("Left TensorTrain must not be empty for partial_contract"))
    isempty(b.data) && throw(ArgumentError("Right TensorTrain must not be empty for partial_contract"))

    left_sites = Set(_flatten_siteinds(a))
    right_sites = Set(_flatten_siteinds(b))
    for pair in spec.contract_pairs
        first(pair) in left_sites || throw(ArgumentError("contract_pairs left index $(first(pair)) is not a site index of the left TensorTrain"))
        last(pair) in right_sites || throw(ArgumentError("contract_pairs right index $(last(pair)) is not a site index of the right TensorTrain"))
    end
    for pair in spec.diagonal_pairs
        first(pair) in left_sites || throw(ArgumentError("diagonal_pairs left index $(first(pair)) is not a site index of the left TensorTrain"))
        last(pair) in right_sites || throw(ArgumentError("diagonal_pairs right index $(last(pair)) is not a site index of the right TensorTrain"))
    end

    if spec.output_order !== nothing
        contract_left = Set(first(pair) for pair in spec.contract_pairs)
        contract_right = Set(last(pair) for pair in spec.contract_pairs)
        diagonal_right = Set(last(pair) for pair in spec.diagonal_pairs)
        surviving = union(
            setdiff(left_sites, contract_left),
            setdiff(right_sites, union(contract_right, diagonal_right)),
        )
        for index in spec.output_order
            index in surviving || throw(ArgumentError("output_order index $index is not a surviving site index"))
        end
    end

    return nothing
end

function _index_handles(indices::AbstractVector{<:Index})
    return Ptr{Cvoid}[_new_index_handle(index) for index in indices]
end

function _pair_side_handles(pairs::Vector{Pair{Index,Index}}, side::Symbol)
    if side === :left
        return _index_handles(Index[first(pair) for pair in pairs])
    elseif side === :right
        return _index_handles(Index[last(pair) for pair in pairs])
    else
        throw(ArgumentError("unknown pair side $side"))
    end
end

function _release_index_handles!(handles::Vector{Ptr{Cvoid}})
    for handle in reverse(handles)
        _release_index_handle(handle)
    end
    empty!(handles)
    return nothing
end

"""
    partial_contract(a::TensorTrain, b::TensorTrain, spec::PartialContractionSpec;
                     center=1, method=:zipup, threshold=nothing, maxdim=nothing,
                     svd_policy=nothing, nfullsweeps=0, convergence_tol=0.0,
                     factorize_alg=:svd, qr_rtol=0.0) -> TensorTrain

Partially contract two `TensorTrain`s using Rust's TreeTN partial-contraction
kernel. The pair semantics are defined by [`PartialContractionSpec`](@ref).
Truncation and factorization keywords match [`contract`](@ref).
"""
function partial_contract(
    a::TensorTrain,
    b::TensorTrain,
    spec::PartialContractionSpec;
    center::Integer=1,
    method::Symbol=:zipup,
    threshold::Union{Nothing,Real}=nothing,
    maxdim::Union{Nothing,Integer}=nothing,
    svd_policy::Union{Nothing, SvdTruncationPolicy}=nothing,
    nfullsweeps::Integer=0,
    convergence_tol::Real=0.0,
    factorize_alg::Symbol=:svd,
    qr_rtol::Real=0.0,
)
    _validate_partial_contract_inputs(a, b, spec)
    1 <= center <= max(length(a), length(b)) || throw(ArgumentError("center must be in 1:$(max(length(a), length(b))), got $center"))
    threshold_value = _normalize_threshold(threshold)
    maxdim_value = _normalize_maxdim(maxdim)
    nfullsweeps >= 0 || throw(ArgumentError("nfullsweeps must be nonnegative, got $nfullsweeps"))
    convergence_tol >= 0 || throw(ArgumentError("convergence_tol must be nonnegative, got $convergence_tol"))
    qr_rtol >= 0 || throw(ArgumentError("qr_rtol must be nonnegative, got $qr_rtol"))

    method_code = _contract_method_code(method)
    factorize_code = _factorize_alg_code(factorize_alg)
    ffi_policy = _resolve_svd_policy(; threshold=threshold_value, svd_policy)

    scalar_kind = _promoted_scalar_kind(a, b)
    a_handle = _new_treetn_handle(a, scalar_kind)
    b_handle = _new_treetn_handle(b, scalar_kind)
    contract_left_handles = Ptr{Cvoid}[]
    contract_right_handles = Ptr{Cvoid}[]
    diagonal_left_handles = Ptr{Cvoid}[]
    diagonal_right_handles = Ptr{Cvoid}[]
    output_order_handles = Ptr{Cvoid}[]
    result_handle = C_NULL

    try
        contract_left_handles = _pair_side_handles(spec.contract_pairs, :left)
        contract_right_handles = _pair_side_handles(spec.contract_pairs, :right)
        diagonal_left_handles = _pair_side_handles(spec.diagonal_pairs, :left)
        diagonal_right_handles = _pair_side_handles(spec.diagonal_pairs, :right)
        if spec.output_order !== nothing
            output_order_handles = _index_handles(spec.output_order)
        end

        out = Ref{Ptr{Cvoid}}(C_NULL)
        status = _with_svd_policy_ptr(ffi_policy) do policy_ptr
            ccall(
                _t4a(:t4a_treetn_partial_contract),
                Cint,
                (
                    Ptr{Cvoid},
                    Ptr{Cvoid},
                    Csize_t,
                    Ptr{Ptr{Cvoid}},
                    Ptr{Ptr{Cvoid}},
                    Csize_t,
                    Ptr{Ptr{Cvoid}},
                    Ptr{Ptr{Cvoid}},
                    Csize_t,
                    Ptr{Ptr{Cvoid}},
                    Csize_t,
                    Cint,
                    Ptr{Cvoid},
                    Csize_t,
                    Csize_t,
                    Cdouble,
                    Cint,
                    Cdouble,
                    Ref{Ptr{Cvoid}},
                ),
                a_handle,
                b_handle,
                Csize_t(length(spec.contract_pairs)),
                contract_left_handles,
                contract_right_handles,
                Csize_t(length(spec.diagonal_pairs)),
                diagonal_left_handles,
                diagonal_right_handles,
                Csize_t(spec.output_order === nothing ? 0 : length(spec.output_order)),
                output_order_handles,
                Csize_t(center - 1),
                method_code,
                policy_ptr,
                Csize_t(maxdim_value),
                Csize_t(nfullsweeps),
                float(convergence_tol),
                factorize_code,
                float(qr_rtol),
                out,
            )
        end
        _check_backend_status(status, "partially contracting TensorTrains")
        result_handle = out[]
        return _treetn_from_handle(result_handle)
    finally
        _release_treetn_handle(result_handle)
        _release_index_handles!(output_order_handles)
        _release_index_handles!(diagonal_right_handles)
        _release_index_handles!(diagonal_left_handles)
        _release_index_handles!(contract_right_handles)
        _release_index_handles!(contract_left_handles)
        _release_treetn_handle(b_handle)
        _release_treetn_handle(a_handle)
    end
end

"""
    elementwise_product(a::TensorTrain, b::TensorTrain; kwargs...) -> TensorTrain

Compute the pointwise product of two TensorTrains by diagonal-pairing matching
site indices and calling [`partial_contract`](@ref). The result keeps the site
indices of `a`.
"""
function elementwise_product(a::TensorTrain, b::TensorTrain; kwargs...)
    left_groups = siteinds(a)
    right_groups = siteinds(b)
    length(left_groups) == length(right_groups) || throw(DimensionMismatch(
        "elementwise_product requires equal numbers of site tensors, got $(length(left_groups)) and $(length(right_groups))",
    ))

    diagonal_pairs = Pair{Index,Index}[]
    output_order = Index[]
    for i in eachindex(left_groups)
        left_group = left_groups[i]
        right_group = right_groups[i]
        length(left_group) == length(right_group) || throw(DimensionMismatch(
            "elementwise_product requires matching site group lengths at tensor $i, got $(length(left_group)) and $(length(right_group))",
        ))
        for (left, right) in zip(left_group, right_group)
            dim(left) == dim(right) || throw(DimensionMismatch(
                "elementwise_product site dimension mismatch at tensor $i: $left has dim $(dim(left)), $right has dim $(dim(right))",
            ))
            push!(diagonal_pairs, left => right)
            push!(output_order, left)
        end
    end

    spec = PartialContractionSpec(Pair{Index,Index}[], diagonal_pairs; output_order)
    return partial_contract(a, b, spec; kwargs...)
end
